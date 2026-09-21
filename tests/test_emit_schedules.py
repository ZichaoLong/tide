import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, PortLayout, Region
from tidegraph.blocks import canonicalize
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.records import Result
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from isolated_cases import vjp


def topology_case(dtype, topology):
    count = 1 if topology == "self_loop" else 2
    nodes = tuple(Node(n, emission="slot_affine", emit_period=2,
                       emit_phases=(0, -1) if count == 1 else ((0,) if n == 0 else (-1,))) for n in range(count))
    g = Graph(nodes, (Edge(0, count-1, 1),), (Region(1),) * count, (0,), (count-1,))
    m = Model(g, dtype=dtype); q = Continuation(g.identity, 2)
    x = (torch.arange(18, dtype=dtype).reshape(2, 3, 3) / 30).requires_grad_()
    xs = [External(b, 0, p, time, x[b, p]) for b in range(2) for p, time in enumerate((0, 1, 3))]
    return g, m, q, xs, dict(m.named_parameters()) | {"input": x}


@pytest.mark.parametrize("topology", ["self_loop", "chain"])
@pytest.mark.parametrize("implementation", ["python", "native", "cursor"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
def test_sparse_phase_topology_anchors_and_chunked_cursor(dtype, topology, implementation, mode):
    g, m, q, xs, variables = topology_case(dtype, topology)
    expected = run(g, m, q, xs, 6, sealed_until=6, mode=mode)
    g, m, q, xs, actual_variables = topology_case(dtype, topology)
    if implementation == "python":
        actual = specialized(g, m, q, xs, 6, sealed_until=6, topology=topology, mode=mode)
    elif implementation == "native":
        actual = Native(g, m, algorithm=topology, packed=True, workers=3, mode=mode).run(q, xs, 6, sealed_until=6)
    else:
        cursor = Native(g, m, packed=True, workers=3, mode=mode).cursor(q)
        trace, outputs, messages = [], [], []
        for stop in (1, 3, 6):
            part = cursor.advance([a for a in xs if cursor.cut <= a.time < stop], stop, sealed_until=stop)
            if stop == 1:
                assert cursor.snapshot().pending
            trace += part.trace; outputs += part.outputs; messages += part.messages
        actual = canonicalize(g, Result(cursor.snapshot(), trace, outputs, messages, {}))
    equivalent(expected, actual)
    for a, b in ((objective(expected), objective(actual)), (expected.outputs[0][-1], actual.outputs[0][-1])):
        equivalent(vjp(a, variables), vjp(b, actual_variables))


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
def test_slot_affine_settle_embedding_with_sharing_and_permuted_slots(dtype, implementation, mode):
    def fixture():
        node = Node(0, memory="ssm", emission="slot_affine", emit_period=2, emit_phases=(-1, 0))
        g = Graph((node, node, Node(1, emission="slot_affine")), (Edge(0, 2, 1), Edge(1, 2, 1)),
                  (Region(2), Region(1)), (0, 1), (0, 1, 2), PortLayout((0, 1), (0, 1), (0, 0), (1, 0, 0)))
        m = Model(g, dtype=dtype); m.nodes[1] = m.nodes[0]
        x = (torch.arange(18, dtype=dtype).reshape(2, 3, 3) / 30).requires_grad_()
        return g, m, x, dict(m.named_parameters()) | {"input": x}
    g, m, x, variables = fixture(); spec = SettleGraph(g, (1, 2))
    expected = settle(spec, m, Continuation(g.identity, 2), x, mode=mode)
    g, m, x, actual_variables = fixture(); spec = SettleGraph(g, (1, 2)); eg, em = spec.embed(m)
    assert em.nodes[0] is em.nodes[1]
    xs = spec.external(x, encoded=True); stop = spec.stride * x.shape[1]; q = Continuation(eg.identity, 2)
    result = frontier(eg, em, q, xs, stop, sealed_until=stop, mode=mode) if implementation == "python" else Native(
        eg, em, algorithm="frontier", packed=True, workers=3, mode=mode).run(q, xs, stop, sealed_until=stop)
    actual = spec.project(result)
    equivalent(expected, actual)
    for a, b in ((objective(expected), objective(actual)), (expected.outputs[0][-1], actual.outputs[0][-1])):
        grad = vjp(a, variables); equivalent(grad, vjp(b, actual_variables))
        assert grad["nodes.0.extra.emit_w_1"] is None


def test_phase_emission_settle_fixed_chain_anchor(dtype):
    g, m, q, _, variables = topology_case(dtype, "chain"); x = variables["input"]
    spec = SettleGraph(g, (1, 2))
    # Choose an always-on first slot so the odd-rank body actually emits.
    from dataclasses import replace
    g = replace(g, nodes=(replace(g.nodes[0], emit_phases=(-1,)), g.nodes[1]))
    m = Model(g, dtype=dtype); variables = dict(m.named_parameters()) | {"input": x}
    spec = SettleGraph(g, (1, 2)); q = Continuation(g.identity, 2)
    expected = settle_chain(spec, m, q, x, mode="hst")
    actual = settle(spec, m, q, x, mode="hst")
    equivalent(expected, actual); equivalent(vjp(objective(expected), variables), vjp(objective(actual), variables))
