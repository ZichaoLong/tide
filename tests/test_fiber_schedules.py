import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from fiber_cases import PROFILE, fixture, physical
from isolated_cases import vjp
from read_cases import execute


@pytest.mark.parametrize("policy", ["all", "selected", "clear", "old"])
@pytest.mark.parametrize("implementation", ["python-frontier", "python-causal", "native-serial", "native-packed", "native-frontier"])
def test_fiber_schedules_and_isolated_public_roots(dtype, policy, implementation):
    g, m, q, xs, leaves = fixture(dtype, policy)
    expected = execute("reference", g, m, q, xs, stop=7)
    actual = execute(implementation, g, m, q, xs, stop=7)
    equivalent(expected, actual)
    equivalent(physical(expected, m), physical(actual, m, implementation.startswith("native")))
    roots = lambda r: {"output": objective(r, "output"), "pending": objective(r, "pending"),
                       "state": r.continuation.states[0, 0].value,
                       **r.continuation.states[0, 0].slots}
    for key, root in roots(expected).items():
        equivalent(vjp(root, leaves), vjp(roots(actual)[key], leaves), key)
    if policy == "clear":
        cleared = [e for e in actual.trace if e["active"]]
        assert cleared and all(len(e["next_slots"]["key"]) == 0 for e in cleared)
        assert all(len(e["comparison_slots"]["key"]) > 0 for e in cleared)
    # Entirely idle samples keep their representation and observation count.
    assert actual.continuation.states[2, 0].observations == 0
    if implementation == "native-frontier" and policy in {"all", "old"}:
        assert actual.stats["state_scalar_sequence_steps"] > 0


@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("clear", [False, True])
def test_fiber_cyclic_serial_parallel_and_batch_fallback(dtype, mode, clear):
    g, m, q, xs, leaves = fixture(dtype, "clear" if clear else "all", cyclic=True)
    expected = execute("reference", g, m, q, xs, stop=9, mode=mode)
    for workers, packed in ((1, False), (1, True), (3, True)):
        actual = Native(g, m, workers=workers, packed=packed, mode=mode).run(q, xs, 9, sealed_until=9)
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))


@pytest.mark.parametrize("topology", ["self_loop", "chain"])
@pytest.mark.parametrize("clear", [False, True])
def test_fiber_independent_topology_specializations(dtype, topology, clear):
    count = 1 if topology == "self_loop" else 3
    g = Graph(tuple(Node(i, memory=PROFILE, clear=clear) for i in range(count)),
              (Edge(0, 0, 2),) if count == 1 else (Edge(0, 1, 2), Edge(1, 2, 3)),
              (Region(1),)*count, (0,), (count-1,))
    m = Model(g, width=2, dtype=dtype); q = Continuation(g.identity, 2)
    x = torch.tensor([[.2, -.3], [.4, .1]], dtype=dtype, requires_grad=True)
    # At t=5+b, self-loop feedback and a fresh external row form a real fiber.
    xs = [External(b, 0, pos, t+b, x[b]) for b in range(2) for pos, t in enumerate((3, 5))]
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = run(g, m, q, xs, 11, sealed_until=11, mode="softp")
    for actual in (specialized(g, m, q, xs, 11, sealed_until=11, topology=topology, mode="softp"),
                   Native(g, m, algorithm=topology, mode="softp").run(q, xs, 11, sealed_until=11)):
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("ports", [1, 2])
def test_fiber_settle_clock_and_source_slot_embedding(dtype, prefill, ports):
    g = Graph((Node(0, memory=PROFILE), Node(1, memory=PROFILE)), (Edge(0, 1, 2),),
              (Region(1), Region(1)), (0,)*ports, (1,))
    m = Model(g, width=2, dtype=dtype); spec = SettleGraph(g, (1, 3)); eg, em = spec.embed(m)
    q = Continuation(g.identity, 2)
    x = (torch.arange(12, dtype=dtype).reshape(2, 3, 2)/30).requires_grad_()
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = settle(spec, m, q, x, mode="hst", prefill=prefill)
    encoded = Native(eg, em, algorithm="frontier", workers=3, packed=True, prefill=prefill, mode="hst").run(
        spec.embed_initial(q, eg), spec.external(x, encoded=True), 3*spec.stride, sealed_until=3*spec.stride)
    results = [spec.project(encoded)]
    if ports == 1:  # The independent chain specialization declares a single input.
        results.append(settle_chain(spec, m, q, x, mode="hst"))
    for actual in results:
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
        equivalent(physical(expected, m), physical(actual, m, native=True))


@pytest.mark.parametrize("clear", [False, True])
def test_fiber_lh_rms_composition_directional_vjp(dtype, clear):
    # A nonzero projection offset keeps this integrated FP32 check well conditioned.
    # The retained first development failure records near-zero RMS cancellation.
    g, m, q, xs, leaves = fixture(dtype, "clear" if clear else "all", cyclic=True, full="lh-silu-rms-v1")
    with torch.no_grad():
        for w in m.nodes:
            w.extra["fiber_out_bias"].fill_(.5)
    expected = execute("reference", g, m, q, xs, stop=9)
    actual = execute("native-packed", g, m, q, xs, stop=9)
    equivalent(expected, actual)
    # Nonuniform output cotangents exercise the normalized direction rather than
    # the almost-constant squared norm of a normalized vector.
    def root(r):
        return sum((x*x.new_tensor([.2, -.3, .7, -.1])).sum() for _, _, _, x in r.outputs)
    equivalent(vjp(root(expected), leaves), vjp(root(actual), leaves))
