"""General online batching versus independent scalar semantics, never route replay."""
import random
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.blocks import canonicalize
from tidegraph.compare import equivalent, objective
from tidegraph.greedy import run as greedy
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.records import Result
from tidegraph.reference import run as scalar
from tidegraph.streaming import run as streaming


def execute(implementation, graph, model, q, xs, stop, **kwargs):
    if implementation == "python":
        return greedy(graph, model, q, xs, stop, sealed_until=stop, **kwargs)
    engine = Native(graph, model, algorithm="greedy", packed=True, workers=2, **kwargs)
    return engine.run(q, xs, stop, sealed_until=stop)


def generated(dtype, seed, memory="ema"):
    rng = random.Random(seed)
    regions = tuple(Region(1, observe_all=(seed+r)%3 != 0, count_priority=(seed+r)%2 == 0)
                    for r in range(3))
    nodes = tuple(Node((n+seed)%3, clear=(seed+n)%4 == 0, memory=memory,
                       full="swiglu", query_heads=2, kv_heads=1, window=3) for n in range(5))
    edges = [Edge(n, (n+1)%5, rng.randrange(1, 4)) for n in range(5)]
    edges += [Edge(rng.randrange(5), rng.randrange(5), rng.randrange(1, 5)) for _ in range(4)]
    edges += [edges[0]]  # preserve distinct parallel-wire identities
    graph = Graph(nodes, tuple(edges), regions, (0, 3), (1, 4))
    model = Model(graph, width=4, dtype=dtype, seed=seed+41)
    variables = dict(model.named_parameters())
    xs = []
    for b in range(2):
        for p in range(2):
            for position, time in enumerate((0, 2, 4)):
                value = (torch.sin(torch.arange(4, dtype=dtype)*.31+(b+p+time+seed)*.27)*.2).requires_grad_()
                variables[f"input.{b}.{p}.{position}"] = value
                xs.append(External(b, p, position, time, value))
    return graph, model, Continuation(graph.identity, 2), xs, variables


def gradients(root, variables, zero=False):
    # Fixed external binary fractions, independent of the candidate output.
    weights = ((torch.arange(root.numel(), device=root.device).remainder(7)-3)/8).to(root.dtype).reshape(root.shape)
    loss = (root * weights).sum() * (0 if zero else 1)
    values = torch.autograd.grad(loss, list(variables.values()), allow_unused=True, retain_graph=True)
    return dict(zip(variables, values))


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
def test_feedback_full_observables_and_isolated_vjps(dtype, implementation, seed, mode):
    graph, model, q, xs, variables = generated(dtype, seed)
    expected = scalar(graph, model, q, xs, 8, sealed_until=8, mode=mode)
    actual = execute(implementation, graph, model, q, xs, 8, mode=mode,
                     full_autograd="batched", aggregate_autograd="batched")
    equivalent(expected, actual)
    assert actual.stats["candidate_events"] == len(actual.trace)
    assert actual.stats["visited_edges"] == len(actual.messages)
    roots = [(objective(expected), objective(actual)),
             (expected.trace[0]["content"], actual.trace[0]["content"]),
             (expected.continuation.pending[0].value, actual.continuation.pending[0].value)]
    if expected.outputs:
        roots.append((expected.outputs[0][-1], actual.outputs[0][-1]))
    for a, b in roots:
        for zero in (False, True):
            equivalent(gradients(a, variables, zero), gradients(b, variables, zero))


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("memory", ["ssm", "attention", "linear", "delta", "delta-rule-v1"])
def test_modules_cuts_and_streaming_transition(dtype, implementation, memory):
    graph, model, q, xs, variables = generated(dtype, 5, memory)
    expected = scalar(graph, model, q, xs, 8, sealed_until=8, mode="hst")
    actual = execute(implementation, graph, model, q, xs, 8, mode="hst")
    equivalent(expected, actual)
    events, outputs, messages = [], [], []
    snapshots = []
    for stop in (0, 2, 5, 8):
        inputs = [a for a in xs if q.cut <= a.time < stop]
        if stop == 8:
            part = streaming(graph, model, q, inputs, stop, sealed_until=stop, mode="hst")
        else:
            part = execute(implementation, graph, model, q, inputs, stop, mode="hst")
        q = part.continuation
        snapshots.append((q, {owner: state.value.detach().clone() for owner, state in q.states.items()}))
        events += part.trace; outputs += part.outputs; messages += part.messages
    chunked = canonicalize(graph, Result(q, events, outputs, messages, {}))
    equivalent(expected, chunked)
    equivalent(gradients(objective(expected), variables), gradients(objective(chunked), variables))
    for snapshot, values in snapshots:
        equivalent(values, {owner: state.value for owner, state in snapshot.states.items()})


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("feedback", [False, True])
def test_actual_time_batches_and_no_future_domain_expansion(dtype, implementation, feedback):
    edges = (Edge(0, 1, 2), Edge(1, 2, 2)) + ((Edge(2, 0, 20),) if feedback else ())
    graph = Graph(tuple(Node(n) for n in range(3)), edges, (Region(1),)*3, (0,), (2,))
    model = Model(graph, dtype=dtype)
    q = Continuation(graph.identity, 2)
    xs = [External(b, 0, t, t, torch.full((3,), .01*(t+b), dtype=dtype))
          for b in range(2) for t in range(8)]
    expected = scalar(graph, model, q, xs, 12, sealed_until=12)
    actual = execute(implementation, graph, model, q, xs, 12)
    equivalent(expected, actual)
    assert actual.stats["greedy_stages"] == 3
    assert actual.stats["full_blocks"] == 3
    assert actual.stats["max_full_batch"] == 16
    assert actual.stats["state_sequence_calls"] > 0
    assert actual.stats["max_state_sequence"] == 8


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_large_int64_clock_gap_is_not_a_tick_loop(dtype, implementation):
    start = 2**55+17
    graph = Graph((Node(0, identity=True), Node(1, identity=True)),
                  (Edge(0, 0, 1), Edge(1, 1, 1)), (Region(1),)*2, (0,), (0,))
    model = Model(graph, dtype=dtype)
    xs = [External(0, 0, 0, start, torch.ones(3, dtype=dtype))]
    expected = scalar(graph, model, Continuation(graph.identity, 1, cut=start), xs,
                      start+3, sealed_until=start+3)
    actual = execute(implementation, graph, model, Continuation(graph.identity, 1), xs, start+3)
    equivalent(expected, actual)
    assert [e["time"] for e in actual.trace] == [start, start+1, start+2]
    assert actual.stats["greedy_stages"] == 3
    assert actual.stats["max_live_fibers"] == 1


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_capacity_is_live_work_not_cumulative_and_failure_keeps_input(dtype, implementation):
    graph = Graph((Node(0, identity=True),), (Edge(0, 0, 1),), (Region(1),), (0,), (0,))
    model = Model(graph, dtype=dtype)
    q = Continuation(graph.identity, 1)
    xs = [External(0, 0, 0, 0, torch.ones(3, dtype=dtype))]
    actual = execute(implementation, graph, model, q, xs, 20, max_events=1)
    assert len(actual.trace) == 20
    assert len(actual.continuation.pending) == 1
    assert q.cut == 0 and not q.states and not q.pending and not q.ledger
    xs.append(External(0, 0, 1, 10, torch.ones(3, dtype=dtype)))
    with pytest.raises((ValueError, RuntimeError), match="live-fiber capacity"):
        execute(implementation, graph, model, q, xs, 20, max_events=1)
    assert q.cut == 0 and not q.states and not q.pending and not q.ledger


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_three_optimizer_updates(dtype, implementation, optimizer):
    records = []
    for candidate in (False, True):
        graph, model, q, xs, variables = generated(dtype, 2, "ssm")
        make = torch.optim.SGD if optimizer == "sgd" else torch.optim.AdamW
        options = dict(lr=.0001, weight_decay=.01)
        if optimizer == "sgd": options["momentum"] = .9
        else: options["eps"] = 1e-5
        opt = make(model.parameters(), **options)
        trajectory = []
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            result = execute(implementation, graph, model, q, xs, 8) if candidate else scalar(
                graph, model, q, xs, 8, sealed_until=8)
            loss = objective(result)/100
            loss.backward()
            grads = {k: None if p.grad is None else p.grad.detach().clone() for k,p in model.named_parameters()}
            opt.step()
            owners = {k:p.detach().clone() for k,p in model.named_parameters()}
            slots = {k:{name: value.detach().clone() if isinstance(value,torch.Tensor) else value
                        for name,value in opt.state[p].items()} for k,p in model.named_parameters()}
            trajectory.append((loss.detach(), grads, owners, slots))
        records.append(trajectory)
    equivalent(*records)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_changed_inputs_decide_actual_routes_and_zero_is_present(dtype, implementation):
    graph = Graph((Node(0), Node(0),
                   Node(1), Node(1)), (Edge(0, 2, 1), Edge(0, 2, 1), Edge(1, 3, 2)),
                  (Region(1, count_priority=False), Region(1)), (0, 1), (2, 3))
    model = Model(graph, dtype=dtype)
    with torch.no_grad():
        for weights in model.nodes:
            weights.read.fill_(1)
        for parameter in model.input_scale:
            parameter.fill_(1)
        for parameter in model.edge_scale:
            parameter.zero_()  # These present messages must still create events.
    routes = []
    for winner in (0, 1):
        xs = [External(0, port, 0, 0, torch.full((3,), .9 if port == winner else .1, dtype=dtype))
              for port in (0, 1)]
        q = Continuation(graph.identity, 1)
        expected = scalar(graph, model, q, xs, 3, sealed_until=3)
        actual = execute(implementation, graph, model, q, xs, 3)
        equivalent(expected, actual)
        targets = {a.node for a in actual.messages}
        assert targets == {winner+2}
        assert all(torch.count_nonzero(a.value) == 0 for a in actual.messages)
        assert (0, winner+2) in actual.continuation.states
        assert (0, 3-winner) not in actual.continuation.states
        if winner == 0:
            target = next(e for e in actual.trace if e["node"] == 2)
            assert {a.source for a in target["fiber"]} == {0, 1}
        routes.append(targets)
    assert routes[0] != routes[1]


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_clock_overflow_rejected_without_advancing_input(dtype, implementation):
    stop = 2**63-1
    graph = Graph((Node(0, identity=True),), (Edge(0, 0, 2),), (Region(1),), (0,), (0,))
    model = Model(graph, dtype=dtype)
    q = Continuation(graph.identity, 1, cut=stop-1)
    xs = [External(0, 0, 0, stop-1, torch.ones(3, dtype=dtype))]
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="overflow"):
        execute(implementation, graph, model, q, xs, stop)
    assert q.cut == stop-1 and not q.pending and not q.states and not q.ledger


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("policy", ["all", "selected", "clear", "old"])
def test_feedback_packed_fiber_attention_and_cache_roots(dtype, implementation, policy):
    from fiber_cases import fixture
    from fiber_pool_cases import profile
    graph, model, q, xs, leaves = fixture(dtype, policy, cyclic=True, profile=profile("all-softmax"))
    expected = scalar(graph, model, q, xs, 9, sealed_until=9, mode="hst")
    actual = execute(implementation, graph, model, q, xs, 9, mode="hst")
    equivalent(expected, actual)
    for a, b in [(objective(expected), objective(actual)),
                 (expected.continuation.states[0, 2].slots["key"],
                  actual.continuation.states[0, 2].slots["key"])]:
        for zero in (False, True):
            equivalent(gradients(a, leaves, zero), gradients(b, leaves, zero))
