import dataclasses
import pytest
import torch
from tidegraph import Atom, Continuation, Edge, External, Graph, Node, Region
from tidegraph.ops import Model, emit
from tidegraph.reference import run
from tidegraph.compare import equivalent


def test_hand_computed_delayed_loop(dtype):
    graph = Graph((Node(0),), (Edge(0, 0, 2),), (Region(1),), (0,), (0,))
    model = Model(graph, width=1, dtype=dtype)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for scales in (model.input_scale, model.agg_scale, model.edge_scale, model.output_scale):
            for p in scales:
                p.fill_(1)
    x = torch.tensor([2.0], dtype=dtype, requires_grad=True)
    result = run(graph, model, Continuation(graph.identity, 1), [External(0, 0, 0, 0, x)], 5, sealed_until=5)
    assert [o[1] for o in result.outputs] == [0, 2, 4]
    equivalent([o[3] for o in result.outputs], [x, x, x])
    equivalent(result.continuation.states[0, 0].value, x * 1.75)
    assert [(a.time, a.position, a.source) for a in result.continuation.pending] == [(6, 4, 0)]
    grad, = torch.autograd.grad(result.continuation.states[0, 0].value.sum(), x)
    equivalent(grad, torch.full_like(x, 1.75))


@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
def test_emit_local_vjp(dtype, mode):
    h = torch.tensor([[0.2, -0.3], [0.4, 0.8]], dtype=dtype, requires_grad=True)
    g = torch.tensor([[0.9, -0.2], [0.1, 0.5]], dtype=dtype, requires_grad=True)
    p = torch.tensor([0.4, 0.8], dtype=dtype, requires_grad=True)
    u = torch.tensor([[0.5, 0.7], [-0.1, 0.3]], dtype=dtype)
    value = emit(h, g, p, mode, 0.6)
    dh, dg, dp = torch.autograd.grad((value * u).sum(), (h, g, p), allow_unused=True)
    if mode == "hard":
        assert torch.equal(value, g) and dh is None and dp is None
        equivalent(dg, u)
    elif mode == "hst":
        assert torch.equal(value, g)
        equivalent(dh, torch.zeros_like(h)); equivalent(dg, u)
        equivalent(dp, 0.6 * (u * (g - h)).sum(-1))
    else:
        equivalent(dh, (1 - p[:, None]) * u); equivalent(dg, p[:, None] * u)
        equivalent(dp, (u * (g - h)).sum(-1))


def test_graph_rejects_invalid_and_preserves_parallel_edges():
    with pytest.raises(ValueError):
        Graph((Node(0),), (Edge(0, 0, 0),), (Region(1),), (0,), ())
    g = Graph((Node(0), Node(1)), (Edge(0, 1, 1), Edge(0, 1, 3)), (Region(1), Region(1)), (0,), ())
    assert g.adjacency() == ([0, 2, 2], [0, 1])
    assert g.adjacency(False) == ([0, 0, 2], [0, 1])
    assert g.topological_order() == [0, 1]
    with pytest.raises(ValueError):
        dataclasses.replace(g, edges=g.edges + (Edge(1, 0, 1),)).topological_order()


def test_seals_and_pending_validation(dtype):
    g = Graph((Node(0),), (Edge(0, 0, 2),), (Region(1),), (0,), ())
    m, q = Model(g, dtype=dtype), Continuation(g.identity, 1)
    with pytest.raises(ValueError, match="seal"):
        run(g, m, q, [], 3, sealed_until=2)
    q.cut = 1
    q.pending = [Atom(0, 0, 2, 1, 0, 0, torch.ones(3, dtype=dtype))] * 2
    with pytest.raises(ValueError, match="pending"):
        run(g, m, q, [], 3, sealed_until=3)
