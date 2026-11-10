"""Literal transpose and structural counterexamples, independent of graph replay."""
import pytest
import torch
from tidegraph.weighted.query_roots import coalesce
from tidegraph.weighted.packing_observation import observe
from tidegraph.compare import equivalent


def test_complete_partition_transpose_reordered_roots_and_shared_parameters(dtype):
    x = torch.arange(12, dtype=dtype).reshape(4, 3).requires_grad_()
    weight = torch.tensor([.1, -.3, .2], dtype=dtype, requires_grad=True)
    batch = x*weight
    roots = [batch[i] for i in (2, 0, 3, 1)]
    bars = [torch.full_like(t, float(i+1)) for i, t in enumerate(roots)]
    with observe() as observation:
        packed, adjoints = coalesce(roots, bars)
    assert len(packed) == 1 and packed[0] is batch
    assert observation.metrics()["packing/projection/query_vjp_root_partition/rows"] == 4
    actual = torch.autograd.grad(packed, (x, weight), adjoints, retain_graph=True)
    expected = torch.autograd.grad(roots, (x, weight), bars, retain_graph=True)
    equivalent(actual, expected, atol=0, rtol=0)
    literal = torch.tensor([2., 4., 1., 3.], dtype=dtype)[:, None].expand(4, 3)
    equivalent(actual, (literal*weight, (literal*x).sum(0)), atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["partial", "overlap", "strided", "empty"])
def test_incomplete_partitions_keep_exact_roots(dtype, kind):
    leaf = torch.arange(12, dtype=dtype).requires_grad_()
    batch = leaf.square().reshape(4, 3)
    roots = {"partial": [batch[0], batch[2]], "overlap": [batch[0], batch[0], batch[2:]],
             "strided": list(batch.t().unbind()), "empty": [batch[:0], batch]}[kind]
    bars = [torch.ones_like(t) for t in roots]
    packed, adjoints = coalesce(roots, bars)
    assert len(packed) == len(roots) and all(a is b for a, b in zip(roots, packed))
    equivalent(torch.autograd.grad(packed, leaf, adjoints, retain_graph=True),
               torch.autograd.grad(roots, leaf, bars), atol=0, rtol=0)


def test_independent_field_absence_and_connected_zero_are_preserved(dtype):
    x = torch.ones(3, dtype=dtype, requires_grad=True)
    y = torch.ones(3, dtype=dtype, requires_grad=True)
    packed, bars = coalesce(list((x*2).unbind()), [torch.zeros((), dtype=dtype)]*3)
    a, b = torch.autograd.grad(packed, (x, y), bars, allow_unused=True)
    assert a is not None and torch.equal(a, torch.zeros_like(x)) and b is None


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_full_private_backward_reaches_a_batched_root(dtype, native):
    from test_weighted_dense_query import independent
    from test_weighted_query_vjp import native_query
    from tidegraph.weighted.query_vjp import evaluate
    from tidegraph.full import FullInput
    from isolated_cases import vjp
    w, requests, roots = independent(dtype)
    rows = [requests[1], requests[2], requests[5], requests[1]]
    rows = [FullInput(r.state, r.time, r.content, r.content.value.new_tensor(.3)) for r in rows]
    expected = [w.full_program.step(w, r, 1, "hard", 1.).value for r in rows]
    w.full_program.query_execution = "packed-v2"
    if native:
        import _tide_native as core
        core.reset_work(True)
    try:
        with observe() as observation:
            result = (native_query if native else evaluate)(w.full_program, w, rows, kind="full", slots=1)
            actual = vjp(sum(r.value.square().sum() for r in result), roots)
        metrics = core.work_metrics() if native else observation.metrics()
        assert metrics["packing/projection/query_vjp_root_partition/rows"] >= 2
        equivalent(actual, vjp(sum(t.square().sum() for t in expected), roots))
    finally:
        if native:
            core.reset_work(False)
