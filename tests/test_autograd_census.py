"""Graph identity, hidden graphs, execution counts and non-owning observations."""
import gc
import weakref

import pytest
import torch

from tidegraph.autograd_census import Census, current, register_private_graph, tensor_roots


def test_all_roots_shared_edges_and_repeat_registration_are_deduplicated():
    x = torch.tensor([2., 3.], requires_grad=True)
    with Census() as census:
        y = x*x
        z = y.sum()
        census.register([z, y, x])
        census.register([y], label="second-view")
        report = census.snapshot()
        assert report["total"]["unique_nodes"] == 3
        assert report["total"]["connected_edges"] == 3  # Mul has two edges to x.
        assert report["groups"]["second-view"]["unique_nodes"] == 2
        assert report["total"]["node_executions"] == 0
        z.backward(retain_graph=True)
        z.backward()
        assert census.snapshot()["total"]["node_executions"] == 6
    torch.testing.assert_close(x.grad, 4*x)


def test_saved_occurrences_differ_from_alias_storage_bytes():
    x = torch.arange(8., requires_grad=True)
    with Census() as census:
        y = x[:4]*x[4:]
        census.register([y])
        report = census.snapshot()["total"]
        assert report["saved_tensor_occurrences"] == 2
        assert report["saved_logical_bytes"] == 2*4*4
        assert report["saved_unique_storages"] == 1
        assert report["saved_unique_storage_bytes"] == 8*4
        census.register([y])
        assert census.snapshot()["total"] == report
        y.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([4., 5., 6., 7., 0., 1., 2., 3.]))


class _Private(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.enable_grad():
            leaf = x.detach().requires_grad_()
            ref = leaf.square()
        register_private_graph("example", [ref])
        ctx.save_for_backward(leaf, ref)
        return ref.detach().clone()

    @staticmethod
    def backward(ctx, bar):
        leaf, ref = ctx.saved_tensors
        return torch.autograd.grad(ref, leaf, bar)[0]


def test_private_graph_is_visible_only_by_explicit_registration():
    x = torch.tensor([2., 3.], requires_grad=True)
    with Census() as census:
        y = _Private.apply(x)
        census.register([y], label="outer")
        report = census.snapshot()
        assert report["groups"]["outer"]["unique_nodes"] == 2
        assert report["groups"]["private/example"]["unique_nodes"] == 2
        assert report["total"]["unique_nodes"] == 4
        y.sum().backward()
        assert census.snapshot()["groups"]["private/example"]["node_executions"] >= 1
    torch.testing.assert_close(x.grad, 2*x)


def test_no_saved_hook_override_or_hidden_version_change():
    x = torch.tensor([2.], requires_grad=True)
    with Census() as census:
        y = x.square()
        version = x._version
        census.register([y])
        assert x._version == version
        with torch.no_grad():
            x.add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.backward()


def test_observer_does_not_keep_tensor_or_storage_alive():
    with Census() as census:
        x = torch.ones(4, requires_grad=True)
        ref = weakref.ref(x)
        y = x.square().sum()
        census.register([y])
        y.backward()
        del x, y
        gc.collect()
        assert ref() is None
        assert census.snapshot()["total"]["observed_storage_bytes_still_alive"] == 0
    assert census.snapshot()["total"]["saved_unique_storage_bytes"] == 16


def test_nograd_and_disabled_registration_have_no_graph():
    assert current() is None
    register_private_graph("disabled", object())  # Does not traverse or validate.
    with Census() as census, torch.no_grad():
        y = torch.ones(3).square()
        census.register([y])
        report = census.snapshot()
    assert report["total"]["unique_nodes"] == 0
    assert report["groups"]["outer"]["nondifferentiable_roots"] == 1
    assert current() is None


def test_closed_scope_removes_hooks_without_releasing_retained_graph():
    x = torch.ones(3, requires_grad=True)
    with Census() as census:
        y = x.square().sum()
        census.register([y])
    y.backward(retain_graph=True)
    y.backward()
    assert census.snapshot()["total"]["node_executions"] == 0
    torch.testing.assert_close(x.grad, 4*x)


def test_disconnected_and_zero_gradients_unchanged():
    x, unused = (torch.ones(3, requires_grad=True) for _ in range(2))
    with Census() as census:
        y = (x*0).sum()
        census.register([y, unused])
        y.backward()
    assert unused.grad is None
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


def test_scope_validation_and_cleanup_after_exception():
    census = Census()
    with pytest.raises(RuntimeError, match="active census"):
        census.register([])
    with pytest.raises(ValueError, match="sentinel"):
        with census:
            with pytest.raises(RuntimeError, match="cannot nest"):
                with Census():
                    pass
            raise ValueError("sentinel")
    assert current() is None
    with pytest.raises(RuntimeError, match="single-use"):
        with census:
            pass


def test_record_roots_do_not_construct_new_objectives():
    from tidegraph.records import State
    x = torch.ones(2, requires_grad=True)
    y = x.square()
    record = {"state": State(y, slots={"times": torch.arange(2)}), "output": [y]}
    roots = list(tensor_roots(record))
    assert len(roots) == 3 and roots[0] is roots[2] is y
    with Census() as census:
        census.register(roots)
        assert census.snapshot()["total"]["unique_nodes"] == 2


def test_real_query_private_graph_and_gradients():
    from test_weighted_query_vjp import independent
    from tidegraph.weighted.query_vjp import evaluate
    from tidegraph.compare import equivalent
    from isolated_cases import vjp
    w, requests, roots = independent(torch.float64, "gated_delta", "state")
    rows = [requests[1], requests[1]]
    expected = [w.read_program.step(w, r) for r in rows]
    with Census() as census:
        result = evaluate(w.read_program, w, rows)
        loss = sum(row["score"].square() for row in result)
        census.register([loss])
        before = census.snapshot()
        assert before["groups"]["private/query.read"]["unique_nodes"] > 0
        assert before["total"]["unique_nodes"] > before["groups"]["outer"]["unique_nodes"]
        actual = vjp(loss, roots)
        assert census.snapshot()["groups"]["private/query.read"]["node_executions"] > 0
    equivalent(actual, vjp(sum(row["score"].square() for row in expected), roots))
