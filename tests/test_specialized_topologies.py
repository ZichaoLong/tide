"""Fixed topology schedules versus the independent streaming/Settle oracles."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.records import Result
from tidegraph.reference import run as streaming
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run, settle_layered
from isolated_cases import vjp


def loss_vjp(loss, variables):
    """Differentiate the already declared scalar objective without squaring twice."""
    grads = torch.autograd.grad(loss, list(variables.values()), allow_unused=True, retain_graph=True)
    return dict(zip(variables, grads))


def fixture(dtype, topology, kind="ema", clear=False, selector="count-v1"):
    ring = topology == "ring"
    paired = topology == "diamond-region"
    owners = (0, 1, 2) if ring else (0, 1, 1, 2) if paired else (0, 1, 2, 3)
    edges = ((Edge(0, 1, 1), Edge(1, 2, 2), Edge(2, 0, 1)) if ring else
             (Edge(0, 1, 1), Edge(0, 2, 2), Edge(1, 3, 2), Edge(2, 3, 1)))
    regions = tuple(Region(1, observe_all=not clear, selector=selector) for _ in range(max(owners)+1))
    graph = Graph(tuple(Node(r, clear=clear, memory=kind, query_heads=2, kv_heads=1, window=3) for r in owners),
                  edges, regions, (0,), (len(owners)-1,))
    model = Model(graph, width=4, dtype=dtype)
    model.nodes[1].weight = model.nodes[0].weight
    model.agg_scale[1] = model.input_scale[0]
    q = Continuation(graph.identity, 2)
    variables = dict(model.named_parameters())
    for b in range(2):
        for n in range(len(owners)):
            state = model.nodes[n].initial()
            state.value = torch.full_like(state.value, .13 + b*.02, requires_grad=True)
            state.slots = {k: torch.zeros_like(t, requires_grad=True) for k, t in state.slots.items()}
            q.states[b, n] = state
            variables[f"initial.{b}.{n}"] = state.value
            variables.update({f"initial.{b}.{n}.{k}": t for k, t in state.slots.items()})
    xs = []
    for b, times in enumerate(((0, 2, 5), (1, 4))):
        for p, t in enumerate(times):
            x = (torch.arange(4, dtype=dtype) * .02 + .08 + b*.03 + p*.01).requires_grad_()
            variables[f"input.{b}.{p}"] = x
            xs.append(External(b, 0, p, t, x))
    return graph, model, q, xs, variables


def execute(implementation, topology, graph, model, q, xs, stop, mode):
    algorithm = "diamond" if topology.startswith("diamond") else topology
    if implementation == "python":
        return run(graph, model, q, xs, stop, sealed_until=stop, topology=algorithm, mode=mode)
    return Native(graph, model, algorithm=algorithm, workers=3 if implementation == "packed" else 1,
                  packed=implementation == "packed", mode=mode).run(q, xs, stop, sealed_until=stop)


@pytest.mark.parametrize("topology", ["ring", "diamond", "diamond-region"])
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("implementation", ["python", "serial", "packed"])
def test_specialized_ring_diamond_trace_and_roots(dtype, topology, mode, clear, implementation):
    g, m, q, xs, variables = fixture(dtype, topology, clear=clear, selector="tensor-history-v1")
    expected = streaming(g, m, q, xs, 9, sealed_until=9, mode=mode)
    actual = execute(implementation, topology, g, m, q, xs, 9, mode)
    equivalent(expected, actual)
    for root in ("all", "state", "history"):
        equivalent(loss_vjp(objective(expected, root), variables), loss_vjp(objective(actual, root), variables))
    if expected.outputs:
        equivalent(vjp(expected.outputs[0][-1], variables), vjp(actual.outputs[0][-1], variables))
    if expected.continuation.pending:
        equivalent(vjp(objective(expected, "pending"), variables, True),
                   vjp(objective(actual, "pending"), variables, True))


@pytest.mark.parametrize("topology", ["ring", "diamond-region"])
@pytest.mark.parametrize("kind", ["attention", "ssm", "linear", "delta"])
@pytest.mark.parametrize("implementation", ["python", "packed"])
def test_representative_modules_and_chunk_cuts(dtype, topology, kind, implementation):
    g, m, q, xs, variables = fixture(dtype, topology, kind=kind)
    expected = streaming(g, m, q, xs, 9, sealed_until=9, mode="hst")
    actual = execute(implementation, topology, g, m, q, xs, 9, "hst")
    equivalent(expected, actual)
    equivalent(loss_vjp(objective(expected), variables), loss_vjp(objective(actual), variables))
    events, outputs, messages = [], [], []
    for end in (1, 3, 3, 6, 9):
        inputs = [x for x in xs if q.cut <= x.time < end]
        piece = execute(implementation, topology, g, m, q, inputs, end, "hst")
        q = piece.continuation
        events += piece.trace; outputs += piece.outputs; messages += piece.messages
    chunk = Result(q, events, outputs, messages, {})
    # Chunks are already ordered by time; canonicalize edge delivery order.
    from tidegraph.blocks import canonicalize
    equivalent(expected, canonicalize(g, chunk))
    equivalent(loss_vjp(objective(expected), variables), loss_vjp(objective(chunk), variables))


@pytest.mark.parametrize("topology", ["ring", "diamond-region"])
@pytest.mark.parametrize("implementation", ["python", "packed"])
def test_empty_candidates_and_empty_selection(dtype, topology, implementation):
    g, m, q, xs, _ = fixture(dtype, topology, selector="positive-v1")
    with torch.no_grad():
        for weights in m.nodes:
            weights.read.fill_(-1)
    expected = streaming(g, m, q, xs, 8, sealed_until=8)
    actual = execute(implementation, topology, g, m, q, xs, 8, "hard")
    equivalent(expected, actual)
    assert not actual.outputs and not actual.messages
    empty = execute(implementation, topology, g, m, q, [], 0, "hard")
    equivalent(q, empty.continuation)


def layered_fixture(dtype, kind, clear):
    graph = Graph(tuple(Node(r, memory=kind, clear=clear, query_heads=2, kv_heads=1, window=3)
                        for r in (0, 0, 1, 1, 2)),
                  tuple(Edge(a, b, 1) for a, b in ((0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4))),
                  (Region(1, observe_all=not clear, selector="tensor-history-v1"),
                   Region(1, observe_all=not clear), Region(1)), (0, 1), (4,))
    m = Model(graph, width=4, dtype=dtype)
    m.nodes[1] = m.nodes[0]
    x = (torch.arange(32, dtype=dtype).reshape(2, 4, 4) / 90).requires_grad_()
    return SettleGraph(graph, (1, 2, 3)), m, Continuation(graph.identity, 2), x


@pytest.mark.parametrize("kind", ["ema", "attention", "ssm", "linear", "delta"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
def test_layered_settle_independent_schedule(dtype, kind, clear, mode):
    spec, m, q, x = layered_fixture(dtype, kind, clear)
    variables = dict(m.named_parameters()) | {"input": x}
    expected = settle(spec, m, q, x, mode=mode)
    actual = settle_layered(spec, m, q, x, mode=mode)
    equivalent(expected, actual)
    for root in ("all", "state", "history"):
        equivalent(loss_vjp(objective(expected, root), variables), loss_vjp(objective(actual, root), variables))
    eg, em = spec.embed(m)
    native = Native(eg, em, algorithm="frontier", packed=True, workers=3, mode=mode).run(
        spec.embed_initial(q, eg), spec.external(x, encoded=True), 4*spec.stride, sealed_until=4*spec.stride)
    equivalent(actual, spec.project(native))
    equivalent(loss_vjp(objective(actual), variables), loss_vjp(objective(spec.project(native)), variables))


@pytest.mark.parametrize("implementation", ["python", "packed"])
def test_wrong_fixed_topology_is_rejected(dtype, implementation):
    g, m, q, xs, _ = fixture(dtype, "diamond-region")
    with pytest.raises(ValueError, match="specialization|topology|region"):
        execute(implementation, "ring", g, m, q, xs, 9, "hard")
    spec, m, q, x = layered_fixture(dtype, "ema", False)
    bad = SettleGraph(replace(spec.graph, edges=spec.graph.edges[1:]), spec.ranks)
    with pytest.raises(ValueError, match="complete adjacent layers"):
        settle_layered(bad, Model(bad.graph, width=4, dtype=dtype), Continuation(bad.graph.identity, 2), x)


def test_layered_initial_state_slots_and_chunk_history(dtype):
    spec, m, q, x = layered_fixture(dtype, "ssm", False)
    variables = dict(m.named_parameters()) | {"input": x}
    for b in range(q.batch_size):
        for n, weights in enumerate(m.nodes):
            state = weights.initial()
            state.value = torch.full_like(state.value, .04, requires_grad=True)
            state.slots = {k: torch.full_like(v, .03, requires_grad=True) for k, v in state.slots.items()}
            q.states[b, n] = state
            variables[f"initial.{b}.{n}"] = state.value
            variables.update({f"initial.{b}.{n}.{k}": v for k, v in state.slots.items()})
    expected = settle(spec, m, q, x, mode="hst")
    events, outputs, messages, start = [], [], [], 0
    for end in (1, 1, 3, 4):
        piece = settle_layered(spec, m, q, x[:, start:end], mode="hst")
        q = piece.continuation; start = end
        events += piece.trace; outputs += piece.outputs; messages += piece.messages
    actual = Result(q, events, outputs, messages, {})
    equivalent(expected, actual)
    for root in ("all", "state", "history"):
        equivalent(loss_vjp(objective(expected, root), variables), loss_vjp(objective(actual, root), variables))
