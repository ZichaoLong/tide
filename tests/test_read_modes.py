import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from tidegraph.frontier import run as frontier
from isolated_cases import vjp
from read_cases import execute, fixture


IMPLEMENTATIONS = ["reference", "python-frontier", "python-causal", "native-serial", "native-packed", "native-frontier"]


@pytest.mark.parametrize("read_mode", ["content", "old", "proposal"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_read_modes_have_distinct_routes_and_hand_vjps(dtype, read_mode, implementation):
    g, m, q, xs, leaves = fixture(dtype, read_mode)
    result = execute(implementation, g, m, q, xs)
    route = {"content": (1, 0), "old": (0, 2), "proposal": (2, 0)}[read_mode]
    for b in range(2):
        assert [e["node"] for e in result.trace if e["batch"] == b and e["active"]] == list(route)
        equivalent(q.states[b, 3], result.continuation.states[b, 3])
    first = [e for e in result.trace if e["batch"] == 0 and e["time"] == 2]
    equivalent([e["proposal"].item() for e in first], [2., 2., 2.5])
    expected = {"content": [0., 2., 1.], "old": [4., 0., 3.], "proposal": [2., 2., 2.5]}[read_mode]
    equivalent([e["descriptor"].item() for e in first], expected)
    names = ["input.0.0.0", "initial.0.0", "nodes.0.decay", "nodes.0.read", "input.1.0.0", "input.0.0.1"]
    actual = dict(zip(names, torch.autograd.grad(first[0]["descriptor"], [leaves[name] for name in names],
                                                allow_unused=True, retain_graph=True)))
    coefficients = {"content": [1., None, None, 0., None, None],
                    "old": [None, 1., None, 4., None, None],
                    "proposal": [1., .5, 1., 2., None, None]}[read_mode]
    for name, value in zip(names, coefficients):
        equivalent(actual[name], None if value is None else torch.full_like(leaves[name], value))
    assert all(e["node"] != 3 for e in result.trace)
    if "frontier" in implementation:
        assert result.stats["state_blocks"] == 6


@pytest.mark.parametrize("read_mode", ["content", "old", "proposal"])
@pytest.mark.parametrize("policy", ["all", "selected", "clear"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS[1:])
def test_read_mode_schedules_gradients_and_inference(dtype, read_mode, policy, implementation):
    kwargs = dict(observe_all=policy != "selected", clear=policy == "clear")
    g, m, q, xs, leaves = fixture(dtype, read_mode, **kwargs)
    expected = execute("reference", g, m, q, xs)
    g, m, q, xs, actual_leaves = fixture(dtype, read_mode, **kwargs)
    actual = execute(implementation, g, m, q, xs)
    equivalent(expected, actual)
    for root in (lambda r: objective(r), lambda r: r.trace[6]["descriptor"], lambda r: r.outputs[0][-1]):
        equivalent(vjp(root(expected), leaves), vjp(root(actual), actual_leaves))
    with torch.no_grad():
        inferred = execute(implementation, g, m, q, xs)
    equivalent(actual, inferred)
    assert inferred.stats.get("semantic_read_replays", 0) == 0
    if "frontier" in implementation and policy != "all":
        assert actual.stats.get("state_blocks", 0) == 0


@pytest.mark.parametrize("read_mode", ["content", "old", "proposal"])
@pytest.mark.parametrize("topology", ["self_loop", "chain"])
def test_read_modes_fixed_topology_and_settle_anchors(dtype, read_mode, topology):
    n = 1 if topology == "self_loop" else 2
    edges = (Edge(0, 0, 1),) if n == 1 else (Edge(0, 1, 1),)
    g = Graph(tuple(Node(v) for v in range(n)), edges,
              tuple(Region(1, read_mode=read_mode) for _ in range(n)), (0,), (n-1,))
    m = Model(g, dtype=dtype)
    x = torch.ones((2, 3, 3), dtype=dtype, requires_grad=True)
    xs = [External(b, 0, i, i*3, x[b, i]) for b in range(2) for i in range(3)]
    q = Continuation(g.identity, 2); leaves = dict(m.named_parameters()) | {"input": x}
    expected = run(g, m, q, xs, 9, sealed_until=9, mode="hst")
    for actual in (specialized(g, m, q, xs, 9, sealed_until=9, topology=topology, mode="hst"),
                   Native(g, m, algorithm=topology, workers=2, packed=True, mode="hst").run(q, xs, 9, sealed_until=9)):
        equivalent(expected, actual)
        equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    if topology == "chain":
        spec = SettleGraph(g, (1, 2))
        expected = settle(spec, m, q, x, mode="hst")
        eg, em = spec.embed(m)
        for actual in (settle_chain(spec, m, q, x, mode="hst"),
                       spec.project(frontier(eg, em, Continuation(eg.identity, 2), spec.external(x, encoded=True),
                                             12, sealed_until=12, mode="hst"))):
            equivalent(expected, actual)
            equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
