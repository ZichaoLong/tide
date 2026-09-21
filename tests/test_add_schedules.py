import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from add_cases import PROFILE, fixture, physical
from isolated_cases import vjp
from read_cases import execute


@pytest.mark.parametrize("policy", ["all", "selected", "clear", "blend", "old"])
@pytest.mark.parametrize("implementation", ["python-frontier", "python-causal", "native-serial", "native-packed",
                                           "native-frontier", "native-frontier-causal"])
def test_add_schedule_public_roots(dtype, policy, implementation):
    g, m, q, xs, leaves = fixture(dtype, policy)
    expected = execute("reference", g, m, q, xs, stop=7)
    actual = Native(g, m, algorithm="frontier", packed=True, workers=3, prefill=False, mode="hst").run(
        q, xs, 7, sealed_until=7) if implementation == "native-frontier-causal" else execute(
            implementation, g, m, q, xs, stop=7)
    equivalent(expected, actual)
    ep, ap = physical(expected, m), physical(actual, m, implementation.startswith("native"))
    equivalent(ep, ap)
    for root in ("output", "pending", "state"):
        equivalent(vjp(objective(expected, root), leaves), vjp(objective(actual, root), leaves))
    equivalent(vjp(ep[0, 0], leaves), vjp(ap[0, 0], leaves))
    # Never-observed samples retain encoded state, even though decoded hidden decays.
    assert actual.continuation.states[2, 0].last_time == -1
    if implementation.endswith("frontier") and policy in {"all", "old"}:
        assert actual.stats["state_scalar_sequence_steps"] > 0


@pytest.mark.parametrize("workers,packed", [(1, False), (1, True), (3, True)])
def test_add_cyclic_streaming(dtype, workers, packed):
    g, m, q, xs, leaves = fixture(dtype, "clear", cyclic=True)
    expected = execute("reference", g, m, q, xs, stop=9)
    actual = Native(g, m, workers=workers, packed=packed, mode="hst").run(q, xs, 9, sealed_until=9)
    equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))


@pytest.mark.parametrize("topology", ["self_loop", "chain"])
@pytest.mark.parametrize("clear", [False, True])
def test_add_independent_topology_specializations(dtype, topology, clear):
    count = 1 if topology == "self_loop" else 3
    g = Graph(tuple(Node(i, memory=PROFILE, clear=clear) for i in range(count)),
              (Edge(0, 0, 2),) if count == 1 else (Edge(0, 1, 2), Edge(1, 2, 3)),
              (Region(1),)*count, (0,), (count-1,))
    m = Model(g, width=2, dtype=dtype); q = Continuation(g.identity, 2)
    x = torch.tensor([[.2, -.3], [.4, .1]], dtype=dtype, requires_grad=True)
    xs = [External(b, 0, 0, 3+b, x[b]) for b in range(2)]
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = run(g, m, q, xs, 11, sealed_until=11, mode="softp")
    for actual in (specialized(g, m, q, xs, 11, sealed_until=11, topology=topology, mode="softp"),
                   Native(g, m, algorithm=topology, mode="softp").run(q, xs, 11, sealed_until=11)):
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))


@pytest.mark.parametrize("prefill", [False, True])
def test_add_settle_clock_embedding_and_chain(dtype, prefill):
    g = Graph((Node(0, memory=PROFILE), Node(1, memory=PROFILE)), (Edge(0, 1, 2),),
              (Region(1), Region(1)), (0,), (1,))
    m = Model(g, width=2, dtype=dtype); spec = SettleGraph(g, (1, 3)); eg, em = spec.embed(m)
    value = torch.tensor([.7, -.4], dtype=dtype, requires_grad=True)
    q = Continuation(g.identity, 2, states={(0, 0): State(value)})
    x = (torch.arange(12, dtype=dtype).reshape(2, 3, 2)/30).requires_grad_()
    leaves = dict(m.named_parameters()) | {"input": x, "initial": value}
    expected = settle(spec, m, q, x, mode="hst", prefill=prefill)
    encoded = Native(eg, em, algorithm="frontier", workers=3, packed=True, prefill=prefill, mode="hst").run(
        spec.embed_initial(q, eg), spec.external(x, encoded=True), 3*spec.stride, sealed_until=3*spec.stride)
    for actual in (spec.project(encoded), settle_chain(spec, m, q, x, mode="hst")):
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
        equivalent(physical(expected, m), physical(actual, m, native=True))
