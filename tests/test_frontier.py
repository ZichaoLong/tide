import random
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.records import Result
from cases import gradients


def case(dtype, seed):
    rng = random.Random(seed)
    g = Graph((Node(0), Node(1), Node(0, clear=seed % 2 == 0), Node(1)),
              (Edge(0, 1, 1), Edge(1, 2, 2), Edge(2, 3, 1), Edge(0, 3, rng.randrange(1, 5))),
              (Region(1, observe_all=seed % 3 != 0), Region(1, count_priority=seed % 2 == 0)), (0, 1), (0, 2, 3))
    m = Model(g, dtype=dtype, seed=seed)
    x = (torch.arange(48, dtype=dtype).reshape(16, 3) / 100 - 0.1).requires_grad_()
    xs = [External(b, p, i, t, x[b * 8 + p * 4 + i]) for b in range(2)
          for p in range(2) for i, t in enumerate((0, 1, 3, 6))]
    initial = torch.full((2, 4, 3), 0.17, dtype=dtype, requires_grad=True)
    q = Continuation(g.identity, 2, states={(b, v): State(initial[b, v]) for b in range(2) for v in range(4)})
    return g, m, q, xs, x, initial


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("implementation", ["python", "serial", "parallel"])
def test_frontier_trace_vjp(dtype, seed, mode, implementation):
    g, m, q, xs, x, initial = case(dtype, seed)
    expected = run(g, m, q, xs, 10, sealed_until=10, mode=mode)
    expected_grad = gradients(objective(expected), m, x, initial)
    g, m, q, xs, x, initial = case(dtype, seed)
    actual = frontier(g, m, q, xs, 10, sealed_until=10, mode=mode) if implementation == "python" else Native(
        g, m, mode=mode, workers=1 if implementation == "serial" else 3, algorithm="frontier").run(q, xs, 10, sealed_until=10)
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), m, x, initial))


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("prefill", [False, True])
def test_chain_forms_sequence_blocks(dtype, implementation, prefill):
    g = Graph(tuple(Node(i) for i in range(3)), (Edge(0, 1, 2), Edge(1, 2, 2)),
              (Region(1),) * 3, (0,), (2,))
    m = Model(g, dtype=dtype)
    q = Continuation(g.identity, 2)
    xs = [External(b, 0, t, t, torch.full((3,), 0.01 * (t + b), dtype=dtype)) for b in range(2) for t in range(8)]
    expected = run(g, m, q, xs, 12, sealed_until=12)
    actual = frontier(g, m, q, xs, 12, sealed_until=12, prefill=prefill) if implementation == "python" else Native(
        g, m, algorithm="frontier", workers=3, prefill=prefill).run(q, xs, 12, sealed_until=12)
    equivalent(expected, actual)
    assert actual.stats["frontier_stages"] == 3
    assert actual.stats["full_blocks"] == 3
    assert actual.stats["state_blocks"] == (6 if prefill else 0)
    assert actual.stats["state_steps"] == (0 if prefill else 48)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_frontier_chunk_continuation(dtype, implementation):
    g, m, q, xs, _, _ = case(dtype, 2)
    expected = run(g, m, q, xs, 10, sealed_until=10, mode="softp")
    engine = Native(g, m, algorithm="frontier", workers=3, mode="softp") if implementation == "native" else None
    events, outputs, messages = [], [], []
    for stop in (0, 2, 5, 10):
        inputs = [a for a in xs if q.cut <= a.time < stop]
        piece = engine.run(q, inputs, stop, sealed_until=stop) if engine else frontier(g, m, q, inputs, stop, sealed_until=stop, mode="softp")
        events += piece.trace; outputs += piece.outputs; messages += piece.messages; q = piece.continuation
    equivalent(expected, Result(q, events, outputs, messages, {}))


def test_frontier_limits_and_cycles(dtype):
    g, m, q, xs, _, _ = case(dtype, 0)
    with pytest.raises(ValueError, match="limit"):
        frontier(g, m, q, xs, 10, sealed_until=10, max_events=2)
    g = Graph((Node(0),), (Edge(0, 0, 1),), (Region(1),), (0,), ())
    with pytest.raises(ValueError, match="acyclic"):
        frontier(g, Model(g, dtype=dtype), Continuation(g.identity, 1), [], 1, sealed_until=1)
