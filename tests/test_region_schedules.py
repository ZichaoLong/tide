from dataclasses import replace
import math
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from isolated_cases import vjp
from isolated_cases import fixture as memory_fixture
from read_cases import execute
from region_cases import fixture, linear_vjp
from tidegraph.region import program as region_program


IMPLEMENTATIONS = ["reference", "python-frontier", "python-causal", "native-serial", "native-packed", "native-frontier"]


@pytest.mark.parametrize("kind", ["ssm", "linear", "delta", "attention"])
@pytest.mark.parametrize("implementation", ["python-frontier", "native-packed", "native-frontier"])
def test_history_roots_through_packed_memory_profiles(dtype, kind, implementation):
    g, m, q, xs, leaves = memory_fixture(dtype, kind)
    g = replace(g, regions=tuple(replace(r, selector="tensor-history-v1") for r in g.regions))
    q.identity = g.identity
    m.regions = torch.nn.ModuleList(region_program(layout, dtype) for layout in g.region_layouts)
    leaves.update(dict(m.named_parameters()))
    expected = execute("reference", g, m, q, xs, stop=10)
    actual = execute(implementation, g, m, q, xs, stop=10)
    equivalent(expected, actual)
    for root in (lambda r: r.continuation.history[0, 0].tensors["memory"], lambda r: r.outputs[0][-1], lambda r: objective(r)):
        equivalent(vjp(root(expected), leaves), vjp(root(actual), leaves))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_history_analytic_values_routes_and_vjps(dtype, implementation):
    g, m, q, xs, leaves = fixture(dtype)
    result = execute(implementation, g, m, q, xs)
    assert [(e["time"], e["node"]) for e in result.trace if e["batch"] == 0 and e["active"]] == [(1, 1), (4, 0)]
    history = result.continuation.history[0, 0]
    equivalent(history.tensors["memory"], torch.tensor(4.5, dtype=dtype))
    assert history.last_time == 4 and history.node_maps == {"selected": {0: 1, 1: 1}}
    equivalent(result.continuation.history[2, 0], q.history[2, 0])
    assert all(r == 0 for _, r in result.continuation.history)
    first = next(e for e in result.trace if (e["batch"], e["node"], e["time"]) == (0, 0, 1))
    last = next(e for e in result.trace if (e["batch"], e["node"], e["time"]) == (0, 0, 4))
    equivalent(first["control"], torch.tensor(1/(1+math.exp(1)), dtype=dtype))
    p = 1/(1+math.exp(-.5)); slope = p*(1-p)
    equivalent(last["control"], torch.tensor(p, dtype=dtype))
    grads = linear_vjp(history.tensors["memory"], leaves)
    expected = {"history.0": .25, "regions.0.alpha": 6., "input.0.0.0": .5, "input.0.1.0": .5,
                "input.0.0.1": 1., "input.0.1.1": 1., "nodes.0.read": .5, "nodes.1.read": 3.5}
    for name, value in expected.items():
        equivalent(grads[name], torch.full_like(leaves[name], value))
    assert grads["regions.0.bias"] is None
    controls = linear_vjp(last["control"], leaves)
    expected = {"history.0": .25, "regions.0.alpha": 1., "input.0.0.0": .5, "input.0.1.0": .5,
                "input.0.0.1": 1., "input.0.1.1": -1., "nodes.0.read": .5, "nodes.1.read": -.5}
    for name, factor in expected.items():
        equivalent(controls[name], torch.full_like(leaves[name], factor*slope))
    equivalent(controls["regions.0.bias"], torch.tensor([5*slope, -5*slope, 0], dtype=dtype))
    for gradients in (grads, controls):
        for name in leaves:
            if name.startswith(("nodes.2.", "nodes.3.", "regions.1.", "input.1.", "history.1", "history.2")):
                assert gradients[name] is None
    assert vjp(first["control"], leaves)["regions.0.alpha"] is None


@pytest.mark.parametrize("policy", ["all", "selected", "clear", "blend"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS[1:])
def test_history_all_schedules_roots_and_inference(dtype, policy, implementation):
    g, m, q, xs, leaves = fixture(dtype, policy=policy)
    expected = execute("reference", g, m, q, xs)
    actual = execute(implementation, g, m, q, xs)
    equivalent(expected, actual)
    for root in (lambda r: objective(r), lambda r: objective(r, "history"), lambda r: r.outputs[0][-1],
                 lambda r: r.continuation.states[0, 0].value):
        equivalent(vjp(root(expected), leaves), vjp(root(actual), leaves))
    with torch.no_grad():
        equivalent(actual, execute(implementation, g, m, q, xs))
    if "frontier" in implementation:
        assert actual.stats["region_steps"] == 4
        assert bool(actual.stats["state_blocks"]) == (policy == "all")


@pytest.mark.parametrize("topology", ["self_loop", "chain"])
@pytest.mark.parametrize("profile", ["tensor-history-v1", "positive-v1"])
def test_region_specializations_cycles_and_settle(dtype, topology, profile):
    n = 1 if topology == "self_loop" else 2
    edges = (Edge(0, 0, 1),) if n == 1 else (Edge(0, 1, 1),)
    g = Graph(tuple(Node(v) for v in range(n)), edges, tuple(Region(1, selector=profile) for _ in range(n)), (0,), (n-1,))
    m = Model(g, dtype=dtype)
    if profile == "positive-v1":
        with torch.no_grad():
            for w in m.nodes:
                w.read.fill_(-1)
    q = Continuation(g.identity, 2)
    x = torch.ones((2, 2, 3), dtype=dtype, requires_grad=True)
    leaves = dict(m.named_parameters()) | {"input": x}
    xs = [External(b, 0, i, i*3, x[b, i]) for b in range(2) for i in range(2)]
    expected = run(g, m, q, xs, 6, sealed_until=6, mode="hst")
    for actual in (specialized(g, m, q, xs, 6, sealed_until=6, topology=topology, mode="hst"),
                   Native(g, m, algorithm=topology, packed=True, workers=2, mode="hst").run(q, xs, 6, sealed_until=6)):
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    if profile == "positive-v1":
        assert not expected.outputs and not expected.messages and not expected.continuation.pending
        assert len(expected.trace) == 4 and all(not e["active"] for e in expected.trace)
        assert all(v == 0 for _, v in expected.continuation.states)
    if topology == "chain":
        spec = SettleGraph(g, (1, 2)); eg, em = spec.embed(m)
        assert em.regions[0] is m.regions[0] and em.regions[-1] is not em.regions[-2]
        expected = settle(spec, m, q, x, mode="hst")
        for actual in (settle_chain(spec, m, q, x, mode="hst"),
                       spec.project(frontier(eg, em, Continuation(eg.identity, 2), spec.external(x, encoded=True), 8,
                                             sealed_until=8, mode="hst")),
                       spec.project(Native(eg, em, algorithm="frontier", packed=True, workers=2, mode="hst").run(
                           Continuation(eg.identity, 2), spec.external(x, encoded=True), 8, sealed_until=8))):
            equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
