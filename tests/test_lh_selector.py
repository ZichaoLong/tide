from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, History, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.readout import NormRead
from isolated_cases import vjp
from region_cases import fixture, linear_vjp
from read_cases import execute


IMPLEMENTATIONS = ["reference", "python-frontier", "python-causal", "native-serial", "native-packed", "native-frontier"]


def model(graph, dtype):
    m = Model(graph, width=2, dtype=dtype)
    with torch.no_grad():
        for p in m.input_scale:
            p.fill_(1)
    return m


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_fp64_norm_routing_is_not_a_cast_of_fp32_norm(dtype, implementation):
    g = Graph((Node(0, readout="norm-fp64-v1"), Node(0, readout="norm-fp64-v1")), (),
              (Region(1, read_mode="content", selector="lh-count-affect-v1"),), (0, 1), (0, 1))
    m = model(g, dtype)
    xs = [External(0, v, 0, 0, torch.tensor([1e6, v+1.], dtype=dtype, requires_grad=True)) for v in range(2)]
    result = execute(implementation, g, m, Continuation(g.identity, 1), xs, stop=1)
    assert [e["node"] for e in result.trace if e["active"]] == [1]
    assert all(e["descriptor"].dtype == torch.float64 and e["control"].dtype == dtype for e in result.trace)
    if dtype == torch.float32:
        assert torch.linalg.vector_norm(xs[0].value) == torch.linalg.vector_norm(xs[1].value)
    assert result.trace[1]["descriptor"] > result.trace[0]["descriptor"]
    leaves = {"x": xs[0].value, "unused": xs[1].value, "read": m.nodes[0].read}
    gradients = linear_vjp(result.trace[0]["descriptor"], leaves)
    expected = xs[0].value.double()/torch.linalg.vector_norm(xs[0].value.double())
    equivalent(gradients["x"], expected.to(dtype))
    assert gradients["unused"] is None and gradients["read"] is None


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_lh_prior_counts_affects_scores_and_id_order(dtype, implementation):
    g = Graph(tuple(Node(0, readout="norm-fp64-v1") for _ in range(5)), (),
              (Region(1, read_mode="content", selector="lh-count-affect-v1"),), (0, 1, 2, 3), (0, 1, 2, 3))
    m = model(g, dtype)
    initial = History(node_maps={"selected": {0: 1}, "affected": {0: 100, 1: 1, 2: 3, 3: 3}})
    q = Continuation(g.identity, 1, history={(0, 0): initial})
    position = [0]*4; xs = []
    for time, nodes in ((0, range(4)), (2, (1, 3)), (4, range(4))):
        for v in nodes:
            xs.append(External(0, v, position[v], time, torch.tensor([(9., 10., 1., 1.)[v], 0.], dtype=dtype)))
            position[v] += 1
    result = execute(implementation, g, m, q, xs)
    assert [e["node"] for e in result.trace if e["active"]] == [2, 3, 1]
    assert result.continuation.history[0, 0].node_maps == {
        "selected": {0: 1, 1: 1, 2: 1, 3: 1}, "affected": {0: 102, 1: 4, 2: 5, 3: 6}}
    assert result.continuation.history[0, 0].last_time == 4
    assert q.history[0, 0].node_maps["affected"][0] == 100


@pytest.mark.parametrize("read_mode", ["content", "old", "proposal"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("implementation", ["python-frontier", "native-packed", "native-frontier"])
def test_lh_norm_read_modes_control_training_and_continuation(dtype, read_mode, mode, implementation):
    g = Graph(tuple(Node(0, readout="norm-fp64-v1") for _ in range(3)), (),
              (Region(1, read_mode=read_mode, selector="lh-count-affect-v1"),), (0, 1), (0, 1))
    m = model(g, dtype); q = Continuation(g.identity, 2)
    x = torch.tensor([[[3., 4.], [0., 2.]], [[2., 1.], [1., 1.]]], dtype=dtype, requires_grad=True)
    xs = [External(b, v, i, i*3, x[b, i]*(v+1)) for b in range(2) for v in range(2) for i in range(2)]
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = execute("reference", g, m, q, xs, mode=mode)
    actual = execute(implementation, g, m, q, xs, mode=mode)
    equivalent(expected, actual)
    for root in (lambda r: objective(r), lambda r: r.outputs[0][-1], lambda r: r.trace[-1]["control"]):
        equivalent(vjp(root(expected), leaves), vjp(root(actual), leaves))
    prefix = execute(implementation, g, m, q, [e for e in xs if e.time < 2], stop=2, mode=mode)
    tail = execute(implementation, g, m, prefix.continuation, [e for e in xs if e.time >= 2], mode=mode)
    equivalent(actual.continuation, tail.continuation)
    with torch.no_grad():
        equivalent(actual, execute(implementation, g, m, q, xs, mode=mode))


@pytest.mark.parametrize("implementation", ["python-frontier", "native-packed", "native-frontier"])
def test_mixed_descriptor_dtypes_and_tensor_history(dtype, implementation):
    g, m, q, xs, leaves = fixture(dtype)
    g = replace(g, nodes=(replace(g.nodes[0], readout="norm-fp64-v1"), *g.nodes[1:])); q.identity = g.identity
    m.nodes[0].read_program = NormRead()
    expected = execute("reference", g, m, q, xs)
    actual = execute(implementation, g, m, q, xs)
    equivalent(expected, actual)
    assert all(t.dtype == dtype for h in actual.continuation.history.values() for t in h.tensors.values())
    for root in (lambda r: objective(r), lambda r: r.trace[0]["control"], lambda r: r.continuation.history[0, 0].tensors["memory"]):
        equivalent(vjp(root(expected), leaves), vjp(root(actual), leaves))


@pytest.mark.parametrize("implementation", ["reference", "native-packed", "native-frontier"])
def test_lh_affect_counter_overflow_includes_passive_candidates(dtype, implementation):
    g = Graph((Node(0), Node(0)), (), (Region(1, selector="lh-count-affect-v1"),), (0, 1), ())
    m = model(g, dtype)
    q = Continuation(g.identity, 1, history={(0, 0): History(node_maps={"selected": {1: 4}, "affected": {1: 2**63-1}})})
    xs = [External(0, v, 0, 0, torch.ones(2, dtype=dtype)) for v in range(2)]
    with pytest.raises(ValueError, match="counter overflow"):
        execute(implementation, g, m, q, xs, stop=1)


def test_fp64_read_policy_and_lh_history_are_explicit(dtype):
    g = Graph((Node(0, readout="norm-fp64-v1"),), (), (Region(1, selector="lh-count-affect-v1"),), (0,), ())
    m = model(g, dtype); m.nodes[0].read_program.precision = "payload"
    with pytest.raises(ValueError, match="precision"):
        Native(g, m)
    g = replace(g, regions=(replace(g.regions[0], count_priority=False),)); m = model(g, dtype)
    with pytest.raises(ValueError, match="count priority"):
        Native(g, m)
