"""PDG §9.5 / timed-DAG §6: independent scalar scheduling, full returns and VJP."""
import pytest
import torch
from tidegraph import Graph, Node, Region, Edge, External, Continuation
from tidegraph.compare import equivalent, objective
from tidegraph.ops import Model
from tidegraph.reference import run as reference
from tidegraph.greedy import run as greedy
from tidegraph.native import Native
from test_greedy import generated, gradients


def execute(native, g, m, q, xs, stop, **kwargs):
    options = dict(full_autograd="batched", aggregate_autograd="batched", **kwargs)
    if native:
        return Native(g, m, algorithm="window", packed=True, **options).run(q, xs, stop, sealed_until=stop)
    return greedy(g, m, q, xs, stop, sealed_until=stop, component_window=True, **options)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_nonaligned_reconvergence_waits_for_all_predecessors(dtype, native):
    # A->X and B->C->D->X. Immediate greedy can visit X before D is complete.
    g = Graph(tuple(Node(i, full="swiglu") for i in range(5)),
              (Edge(0, 4, 1), Edge(1, 2, 1), Edge(2, 3, 1), Edge(3, 4, 1)),
              tuple(Region(1) for _ in range(5)), (0, 1), (4,))
    m = Model(g, width=3, dtype=dtype, seed=7)
    xs = [External(b, p, i, t, torch.tensor([.2+b*.1, -.3, .1+i*.05], dtype=dtype, requires_grad=True))
          for b in range(2) for p in range(2) for i, t in enumerate((0, 1, 4))]
    q = Continuation(g.identity, 2)
    expected = reference(g, m, q, xs, 8, sealed_until=8)
    actual = execute(native, g, m, q, xs, 8)
    equivalent(expected, actual)
    assert actual.stats["region_blocks"] == 5
    assert actual.stats["window_cyclic_components"] == 0
    assert actual.stats["window_causal_blocks"] == 0
    old = greedy(g, m, q, xs, 8, sealed_until=8)
    assert old.stats["region_blocks"] > actual.stats["region_blocks"]
    roots = dict(m.named_parameters()) | {f"input.{i}": x.value for i, x in enumerate(xs)}
    equivalent(gradients(objective(expected), roots), gradients(objective(actual), roots))
    first = execute(native, g, m, q, [x for x in xs if x.time < 2], 2)
    assert first.continuation.pending
    tail = execute(native, g, m, first.continuation, [x for x in xs if x.time >= 2], 8)
    equivalent(expected.continuation, tail.continuation)
    assert g.region_components is g.region_components


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("seed", [0, 3])
def test_feedback_and_shared_region_selection_keep_complete_records_and_roots(dtype, native, seed):
    g, m, q, xs, roots = generated(dtype, seed)
    expected = reference(g, m, q, xs, 8, sealed_until=8, mode="hst")
    actual = execute(native, g, m, q, xs, 8, mode="hst")
    equivalent(expected, actual)
    assert actual.stats["window_cyclic_components"] >= 1
    assert actual.stats["window_causal_blocks"] >= 1
    for a, b in ((objective(expected), objective(actual)),
                 (expected.trace[0]["content"], actual.trace[0]["content"])):
        for zero in (False, True): equivalent(gradients(a, roots, zero), gradients(b, roots, zero))


def test_region_scc_includes_self_edges_and_has_no_recursion_limit():
    g = Graph((Node(0), Node(0)), (Edge(0, 1, 1),), (Region(2),), (0,), (1,))
    assert g.region_components.cyclic == [True]  # Node DAG, cyclic region dependency.
    n = 1600
    chain = Graph(tuple(Node(i) for i in range(n)), tuple(Edge(i, i+1, 1) for i in range(n-1)),
                  tuple(Region(1) for _ in range(n)), (0,), (n-1,))
    assert len(chain.region_components.groups) == n
    assert all(chain.region_components.owner[i] < chain.region_components.owner[i+1] for i in range(n-1))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_window_keeps_future_pending_in_capacity(native):
    g = Graph((Node(0), Node(1)), (Edge(0, 1, 10),), (Region(1), Region(1)), (0,), (1,))
    m = Model(g, width=2, dtype=torch.float64)
    xs = [External(0, 0, i, i, torch.ones(2, dtype=torch.float64)) for i in range(3)]
    q = Continuation(g.identity, 1)
    with pytest.raises(ValueError, match="capacity"):
        execute(native, g, m, q, xs, 3, max_events=2)
    result = execute(native, g, m, q, xs, 3, max_events=3)
    assert len(result.continuation.pending) == 3
