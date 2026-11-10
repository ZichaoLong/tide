"""PDG live-fiber capacity counts current work, including future physical fibers."""
import pytest
import torch

from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run as reference


@pytest.mark.parametrize("feedback", [False, True], ids=["chain", "feedback"])
@pytest.mark.parametrize("compact", [False, True], ids=["ordinary", "compact"])
@pytest.mark.parametrize("trace", [False, True], ids=["boundary", "trace"])
def test_capacity_one_allows_many_sequential_events(feedback, compact, trace):
    n = 2 if feedback else 6
    edges = [Edge(i, i+1, 1) for i in range(n-1)]
    if feedback:
        edges.append(Edge(n-1, 0, 1))
    graph = Graph(tuple(Node(i) for i in range(n)), tuple(edges),
                  tuple(Region(1) for _ in range(n)), (0,), (n-1,))
    model = Model(graph, width=2, dtype=torch.float64, seed=7)
    initial = Continuation(graph.identity, 1)
    inputs = [External(0, 0, 0, 0, torch.tensor([.1, .2], dtype=torch.float64))]
    engine = Native(graph, model, algorithm="window", compact_events=compact,
                    max_events=1, trace=trace)
    with torch.no_grad():
        expected = reference(graph, model, initial, inputs, 12, sealed_until=12, trace=trace)
        actual = engine.run(initial, inputs, 12, sealed_until=12)
    equivalent(expected, actual, atol=1e-12, rtol=1e-12)
    assert actual.stats["max_live_fibers"] == 1
    assert actual.stats["candidate_events"] == (12 if feedback else n)
    assert len(actual.continuation.pending) == int(feedback)
    assert initial.cut == 0 and not initial.pending and not initial.states


@pytest.mark.parametrize("compact", [False, True], ids=["ordinary", "compact"])
def test_capacity_keeps_future_fibers_across_components_and_parallel_edges(compact):
    graph = Graph((Node(0), Node(1), Node(2)),
        (Edge(0, 1, 1), Edge(0, 1, 1), Edge(0, 2, 20)),
        (Region(1), Region(1), Region(1)), (0,), (1, 2))
    model = Model(graph, width=2, dtype=torch.float64, seed=7)
    initial = Continuation(graph.identity, 1)
    inputs = [External(0, 0, 0, 0, torch.tensor([.1, .2], dtype=torch.float64))]
    with torch.no_grad():
        # Two repeated physical edges form one fiber; the far-future arrival is
        # a second live fiber even while its component has not been visited.
        with pytest.raises(ValueError, match="capacity"):
            Native(graph, model, algorithm="window", compact_events=compact,
                   max_events=1).run(initial, inputs, 4, sealed_until=4)
        actual = Native(graph, model, algorithm="window", compact_events=compact,
                        max_events=2).run(initial, inputs, 4, sealed_until=4)
        expected = reference(graph, model, initial, inputs, 4, sealed_until=4)
    equivalent(expected, actual, atol=1e-12, rtol=1e-12)
    assert actual.stats["max_live_fibers"] == 2
    assert len(actual.messages) == 3 and len(actual.continuation.pending) == 1
    assert actual.continuation.pending[0].time == 20
