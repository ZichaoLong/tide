import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from fiber_cases import fixture, physical
from fiber_pool_cases import POOLS, profile
from isolated_cases import vjp
from read_cases import execute


@pytest.mark.parametrize("kind", POOLS)
@pytest.mark.parametrize("policy", ["all", "selected", "clear", "old"])
@pytest.mark.parametrize("implementation", ["python-frontier", "native-serial", "native-packed", "native-frontier"])
def test_pooling_schedulers_public_roots(dtype, kind, policy, implementation):
    g, m, q, xs, leaves = fixture(dtype, policy, profile=profile(kind))
    expected = execute("reference", g, m, q, xs, stop=7)
    actual = execute(implementation, g, m, q, xs, stop=7)
    equivalent(expected, actual); equivalent(physical(expected, m), physical(actual, m))
    for root in ("output", "pending", "state"):
        equivalent(vjp(objective(expected, root), leaves), vjp(objective(actual, root), leaves))
    for name, value in expected.continuation.states[0, 0].slots.items():
        equivalent(vjp(value, leaves), vjp(actual.continuation.states[0, 0].slots[name], leaves))


@pytest.mark.parametrize("kind", POOLS)
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
def test_pooling_cyclic_node_parallel_batch(dtype, kind, mode):
    g, m, q, xs, leaves = fixture(dtype, "clear", cyclic=True, profile=profile(kind))
    expected = execute("reference", g, m, q, xs, stop=9, mode=mode)
    actual = execute("native-packed", g, m, q, xs, stop=9, mode=mode)
    equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))


@pytest.mark.parametrize("kind", POOLS)
@pytest.mark.parametrize("topology", ["self_loop", "chain"])
def test_pooling_independent_specializations(dtype, kind, topology):
    count = 1 if topology == "self_loop" else 3
    g = Graph(tuple(Node(i, memory=profile(kind)) for i in range(count)),
              (Edge(0, 0, 2),) if count == 1 else (Edge(0, 1, 2), Edge(1, 2, 3)),
              (Region(1),)*count, (0,), (count-1,))
    m = Model(g, width=2, dtype=dtype); q = Continuation(g.identity, 2)
    x = torch.tensor([[.2, -.3], [.4, .1]], dtype=dtype, requires_grad=True)
    xs = [External(b, 0, pos, t+b, x[b]) for b in range(2) for pos, t in enumerate((3, 5))]
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = execute("reference", g, m, q, xs, stop=11, mode="softp")
    for actual in (specialized(g, m, q, xs, 11, sealed_until=11, topology=topology, mode="softp"),
                   Native(g, m, algorithm=topology, mode="softp").run(q, xs, 11, sealed_until=11)):
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))


@pytest.mark.parametrize("kind", POOLS)
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("ports", [1, 2])
def test_pooling_settle_slot_domain_embedding(dtype, kind, prefill, ports):
    g = Graph((Node(0, memory=profile(kind)), Node(1, memory=profile(kind))), (Edge(0, 1, 2),),
              (Region(1), Region(1)), (0,)*ports, (1,))
    m = Model(g, width=2, dtype=dtype)
    with torch.no_grad():
        for w in m.nodes:
            if "fiber_pool" in w.extra:
                w.extra["fiber_pool"].copy_(torch.arange(len(w.extra["fiber_pool"]), dtype=dtype)+1)
    spec = SettleGraph(g, (1, 3)); eg, em = spec.embed(m)
    q = Continuation(g.identity, 2)
    x = (torch.arange(12, dtype=dtype).reshape(2, 3, 2)/30).requires_grad_()
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = settle(spec, m, q, x, mode="hst", prefill=prefill)
    encoded = Native(eg, em, algorithm="frontier", workers=3, packed=True, prefill=prefill, mode="hst").run(
        spec.embed_initial(q, eg), spec.external(x, encoded=True), 3*spec.stride, sealed_until=3*spec.stride)
    results = [spec.project(encoded)]
    if ports == 1:
        results.append(settle_chain(spec, m, q, x, mode="hst"))
    for actual in results:
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
