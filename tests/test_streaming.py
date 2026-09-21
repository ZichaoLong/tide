import dataclasses
import pytest
import torch
from cases import gradients, ring
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model, emit
from tidegraph.reference import run
from tidegraph.records import Result


@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("workers,packed", [(1, False), (1, True), (3, False), (3, True)])
@pytest.mark.parametrize("root", ["output", "state", "pending"])
def test_native_forward_and_vjp(dtype, mode, workers, packed, root):
    g, m, q, xs, x, initial = ring(dtype)
    expected = run(g, m, q, xs, 7, sealed_until=7, mode=mode)
    expected_grad = gradients(objective(expected, root), m, x, initial)
    g, m, q, xs, x, initial = ring(dtype)
    actual = Native(g, m, mode=mode, workers=workers, packed=packed).run(q, xs, 7, sealed_until=7)
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual, root), m, x, initial))
    assert actual.stats["visited_edges"] == len(actual.messages)
    assert len(q.pending) == 0 and q.cut == 0  # Source continuation was not mutated.


@pytest.mark.parametrize("cuts", [(0, 0, 1, 2, 4, 7), (3, 6, 7), (7,)])
@pytest.mark.parametrize("native", [False, True])
def test_chunk_composition_and_vjp(dtype, cuts, native):
    g, m, q, xs, x, initial = ring(dtype)
    whole = run(g, m, q, xs, 7, sealed_until=7, mode="hst")
    expected_grad = gradients(objective(whole), m, x, initial)
    g, m, q, xs, x, initial = ring(dtype)
    engine = Native(g, m, mode="hst", workers=3, packed=True) if native else None
    events, outputs, messages = [], [], []
    for stop in cuts:
        inputs = [a for a in xs if q.cut <= a.time < stop]
        piece = engine.run(q, inputs, stop, sealed_until=stop) if native else run(g, m, q, inputs, stop, sealed_until=stop, mode="hst")
        events += piece.trace; outputs += piece.outputs; messages += piece.messages
        q = piece.continuation
    stitched = Result(q, events, outputs, messages, {})
    equivalent(whole, stitched)
    equivalent(expected_grad, gradients(objective(stitched), m, x, initial))


def test_native_csr_csc_and_lazy_sparse_work(dtype):
    n = 10000
    g = Graph(tuple(Node(0) for _ in range(n)), (Edge(0, n - 1, 1000000),), (Region(1),), (0,), (n - 1,))
    # Share one node module: graph cardinality must not require distinct weights.
    small = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    m = Model(small, width=1, dtype=dtype)
    m.nodes = torch.nn.ModuleList([m.nodes[0]] * n)
    m.agg_scale = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((), dtype=dtype))])
    m.edge_scale = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((), dtype=dtype))])
    engine = Native(g, m, workers=2, packed=True, trace=False)
    assert (engine.compiled.csr.offsets, engine.compiled.csr.edges) == g.adjacency()
    assert (engine.compiled.csc.offsets, engine.compiled.csc.edges) == g.adjacency(False)
    result = engine.run(Continuation(g.identity, 1), [External(0, 0, 0, 0, torch.ones(1, dtype=dtype))],
                        1000001, sealed_until=1000001)
    assert result.stats["candidate_events"] == 2 and result.stats["logical_times"] == 2
    assert len(result.continuation.states) == 2 and len(result.trace) == 0


def test_native_preserves_no_grad_mode(dtype):
    g, m, q, xs, _, _ = ring(dtype)
    engine = Native(g, m, workers=3, packed=True)
    with torch.no_grad():
        result = engine.run(q, xs, 7, sealed_until=7)
    assert all(not o[3].requires_grad for o in result.outputs)
    assert all(not s.value.requires_grad for s in result.continuation.states.values())


def test_native_rejects_unsealed_window(dtype):
    g, m, q, xs, _, _ = ring(dtype)
    with pytest.raises(ValueError, match="unsealed"):
        Native(g, m).run(q, xs, 7, sealed_until=6)


def test_native_emit_analytic_vjp(dtype):
    import _tide_native
    h = torch.tensor([0.2, 0.4], dtype=dtype, requires_grad=True)
    g = torch.tensor([0.8, -0.1], dtype=dtype, requires_grad=True)
    p = torch.tensor(0.3, dtype=dtype, requires_grad=True)
    value = _tide_native.emit(h, g, p, "hst", 0.7)
    assert torch.equal(value, g)
    dh, dg, dp = torch.autograd.grad(value.sum(), (h, g, p))
    equivalent(dh, torch.zeros_like(h)); equivalent(dg, torch.ones_like(g))
    equivalent(dp, (g - h).sum() * 0.7)
