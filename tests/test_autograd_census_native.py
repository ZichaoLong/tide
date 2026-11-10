"""Native outer/private graph counts; requires a build with census bindings."""
import gc
import weakref

import torch
import _tide_native as core


def test_native_union_counts_executions_and_nonowning_scope():
    x = torch.tensor([2., 3.], requires_grad=True)
    ref = weakref.ref(x)
    core.reset_autograd_census(True)
    try:
        y = x*x
        z = y.sum()
        core.register_autograd_roots("outer", [z, y, x])
        core.register_autograd_roots("overlap", [y])
        report = core.autograd_census_metrics()
        assert report["total/unique_nodes"] == 3
        assert report["total/connected_edges"] == 3
        assert report["graphs/overlap/unique_nodes"] == 2
        assert report["total/node_executions"] == 0
        z.backward(retain_graph=True)
        z.backward()
        assert core.autograd_census_metrics()["total/node_executions"] == 6
        torch.testing.assert_close(x.grad, 4*x)
        del x, y, z
        gc.collect()
        assert ref() is None
    finally:
        core.reset_autograd_census(False)
    assert core.autograd_census_metrics() == {}


def test_native_private_query_saved_scope_and_gradients():
    from test_weighted_query_vjp import independent, native_query
    from tidegraph.compare import equivalent
    from isolated_cases import vjp
    w, requests, roots = independent(torch.float64, "gated_delta", "state")
    rows = [requests[1], requests[1]]
    expected = [w.read_program.step(w, r) for r in rows]
    core.reset_autograd_census(True)
    try:
        result = native_query(w.read_program, w, rows)
        loss = sum(row["score"].square() for row in result)
        core.register_autograd_roots("outer", [loss])
        report = core.autograd_census_metrics()
        assert report["coverage/saved_tensors_explicit_only"] == 1
        assert report["graphs/private/query.read/unique_nodes"] > 0
        assert report["total/unique_nodes"] > report["graphs/outer/unique_nodes"]
        assert report["explicit_saved/occurrences"] > 0
        assert report["explicit_saved/unique_storage_bytes"] > 0
        actual = vjp(loss, roots)
        assert core.autograd_census_metrics()["graphs/private/query.read/node_executions"] > 0
        equivalent(actual, vjp(sum(row["score"].square() for row in expected), roots))
    finally:
        core.reset_autograd_census(False)


def test_native_nograd_counts_are_explicit_zeros():
    core.reset_autograd_census(True)
    try:
        with torch.no_grad():
            y = torch.ones(3).square()
        core.register_autograd_roots("outer", [y])
        report = core.autograd_census_metrics()
        assert report["total/unique_nodes"] == report["total/connected_edges"] == 0
        assert report["total/node_executions"] == 0
        assert report["graphs/outer/nondifferentiable_roots"] == 1
        assert report["explicit_saved/unique_storage_bytes"] == 0
    finally:
        core.reset_autograd_census(False)
