import pytest
import torch
from dataclasses import replace
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.blocks import canonicalize
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.records import Result
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from aggregate_cases import fixture
from isolated_cases import vjp


@pytest.mark.parametrize("kind", ["mean", "weighted_mean", "active_softmax", "all_softmax"])
@pytest.mark.parametrize("implementation", ["python", "python-step", "native-stream", "native-serial", "native-frontier", "native-step"])
@pytest.mark.parametrize("budget", [1, 2])
def test_complete_trace_state_and_isolated_source_vjps(dtype, kind, implementation, budget):
    g, m, q, xs, variables = fixture(dtype, kind, budget)
    expected = run(g, m, q, xs, 5, sealed_until=5, mode="hst")
    g, m, q, xs, actual_variables = fixture(dtype, kind, budget)
    if implementation.startswith("python"):
        actual = frontier(g, m, q, xs, 5, sealed_until=5, mode="hst", prefill=implementation == "python")
    else:
        actual = Native(g, m, packed=implementation != "native-serial", workers=1 if implementation == "native-serial" else 3,
                        algorithm="streaming" if implementation in {"native-stream", "native-serial"} else "frontier",
                        prefill=implementation != "native-step", mode="hst").run(q, xs, 5, sealed_until=5)
    equivalent(expected, actual)
    for a, b in ((objective(expected), objective(actual)), (expected.outputs[0][-1], actual.outputs[0][-1]),
                 (expected.trace[0]["contributions"][1], actual.trace[0]["contributions"][1]),
                 (expected.continuation.pending[0].value, actual.continuation.pending[0].value)):
        for zero in (False, True):
            equivalent(vjp(a, variables, zero), vjp(b, actual_variables, zero))


@pytest.mark.parametrize("kind", ["weighted_mean", "active_softmax", "all_softmax"])
@pytest.mark.parametrize("topology", ["self_loop", "chain"])
@pytest.mark.parametrize("implementation", ["python", "native", "cursor"])
def test_topology_anchors_and_cursor_cuts(dtype, kind, topology, implementation):
    n = 1 if topology == "self_loop" else 2
    g = Graph(tuple(Node(v, aggregation=kind) for v in range(n)), (Edge(0, n-1, 1),), (Region(1),)*n, (0,), (n-1,))
    def case():
        m = Model(g, dtype=dtype); x = torch.ones((2, 3, 3), dtype=dtype, requires_grad=True)
        xs = [External(b, 0, p, t, x[b, p]) for b in range(2) for p, t in enumerate((0, 1, 3))]
        return m, Continuation(g.identity, 2), xs, dict(m.named_parameters()) | {"input": x}
    m, q, xs, variables = case(); expected = run(g, m, q, xs, 5, sealed_until=5, mode="softp")
    m, q, xs, av = case()
    if implementation == "python":
        actual = specialized(g, m, q, xs, 5, sealed_until=5, topology=topology, mode="softp")
    elif implementation == "native":
        actual = Native(g, m, algorithm=topology, packed=True, workers=3, mode="softp").run(q, xs, 5, sealed_until=5)
    else:
        cursor = Native(g, m, packed=True, workers=3, mode="softp").cursor(q)
        trace, outputs, messages = [], [], []
        for stop in (1, 2, 5):
            part = cursor.advance([a for a in xs if cursor.cut <= a.time < stop], stop, sealed_until=stop)
            trace += part.trace; outputs += part.outputs; messages += part.messages
        actual = canonicalize(g, Result(cursor.snapshot(), trace, outputs, messages, {}))
    equivalent(expected, actual)
    equivalent(vjp(objective(expected), variables), vjp(objective(actual), av))
    equivalent(vjp(expected.outputs[0][-1], variables), vjp(actual.outputs[0][-1], av))


@pytest.mark.parametrize("kind", ["mean", "weighted_mean", "active_softmax", "all_softmax"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_shared_source_program_settle_embedding_with_mixed_fiber(dtype, kind, implementation):
    g = Graph((Node(0, aggregation=kind), Node(0, aggregation=kind), Node(1, aggregation=kind)),
              (Edge(0, 2, 1), Edge(1, 2, 1)), (Region(2), Region(1)), (0, 0, 1, 1, 2), (0, 1, 2))
    def case():
        m = Model(g, dtype=dtype); m.nodes[1] = m.nodes[0]
        x = (torch.arange(18, dtype=dtype).reshape(2, 3, 3)/30).requires_grad_()
        return m, x, dict(m.named_parameters()) | {"input": x}
    spec = SettleGraph(g, (1, 2)); m, x, variables = case()
    expected = settle(spec, m, Continuation(g.identity, 2), x, mode="hst")
    m, x, av = case(); eg, em = spec.embed(m)
    assert em.nodes[0] is em.nodes[1]
    q = Continuation(eg.identity, 2); xs = spec.external(x, encoded=True); stop = spec.stride*x.shape[1]
    result = frontier(eg, em, q, xs, stop, sealed_until=stop, mode="hst") if implementation == "python" else Native(
        eg, em, algorithm="frontier", packed=True, workers=3, mode="hst").run(q, xs, stop, sealed_until=stop)
    actual = spec.project(result); equivalent(expected, actual)
    for a, b in ((objective(expected), objective(actual)), (expected.outputs[0][-1], actual.outputs[0][-1])):
        equivalent(vjp(a, variables), vjp(b, av))


@pytest.mark.parametrize("kind", ["weighted_mean", "active_softmax", "all_softmax"])
@pytest.mark.parametrize("policy", ["replay", "batched"])
def test_aggregate_checkpoint_roundtrip_and_profile_rejection(dtype, kind, policy, tmp_path):
    g, m, q, xs, _ = fixture(dtype, kind)
    first = Native(g, m, packed=True, aggregate_autograd=policy).run(q, [x for x in xs if x.time < 2], 2, sealed_until=2)
    path = tmp_path / "aggregate.pt"; save(path, g, m, first.continuation)
    restored = Model(g, width=4, dtype=dtype, seed=123); resumed = load(path, g, restored)
    later = [x for x in xs if x.time >= 2]
    expected = run(g, m, first.continuation.detach(), later, 5, sealed_until=5)
    actual = Native(g, restored, algorithm="frontier", packed=True, workers=3, aggregate_autograd=policy).run(resumed, later, 5, sealed_until=5)
    equivalent(expected, actual)
    equivalent(vjp(objective(expected), dict(m.named_parameters())), vjp(objective(actual), dict(restored.named_parameters())))
    changed = replace(g, nodes=tuple(replace(n, aggregation="mean") for n in g.nodes))
    new = Model(changed, width=4, dtype=dtype); before = {k: v.clone() for k, v in new.state_dict().items()}
    with pytest.raises(ValueError, match="graph"):
        load(path, changed, new)
    equivalent(before, new.state_dict())


@pytest.mark.parametrize("kind", ["weighted_mean", "active_softmax", "all_softmax"])
def test_source_program_settle_chain_specialization(dtype, kind):
    g = Graph((Node(0, aggregation=kind), Node(1, aggregation=kind)), (Edge(0, 1, 1),),
              (Region(1),)*2, (0,), (1,))
    spec = SettleGraph(g, (1, 2)); m = Model(g, dtype=dtype)
    x = (torch.arange(18, dtype=dtype).reshape(2, 3, 3)/30).requires_grad_()
    q = Continuation(g.identity, 2); variables = dict(m.named_parameters()) | {"input": x}
    expected = settle_chain(spec, m, q, x, mode="softp")
    actual = settle(spec, m, q, x, mode="softp")
    equivalent(expected, actual)
    equivalent(vjp(objective(expected), variables), vjp(objective(actual), variables))


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_inference_aggregate_packing_has_no_semantic_replay(dtype, implementation, context):
    with context():
        g, m, q, xs, _ = fixture(dtype, "all_softmax")
        expected = run(g, m, q, xs, 5, sealed_until=5)
        actual = frontier(g, m, q, xs, 5, sealed_until=5) if implementation == "python" else Native(
            g, m, algorithm="frontier", packed=True, workers=3).run(q, xs, 5, sealed_until=5)
    equivalent(expected, actual)
    assert actual.stats["aggregate_calls"] > 0
    assert actual.stats.get("semantic_aggregate_replays", 0) == 0
    assert all(not e["content"].requires_grad for e in actual.trace)
