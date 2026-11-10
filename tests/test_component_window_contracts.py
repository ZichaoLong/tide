"""General ports, same-rank regions and stateful selectors under SCC windows."""
import pytest
from tidegraph.compare import equivalent, objective
from tidegraph.reference import run as reference
from test_component_window import execute
from test_greedy import gradients
from test_settle_ports import fixture as ports
from region_cases import fixture as history


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_same_rank_general_ports_parallel_edges_and_every_cut(dtype, native):
    spec, model, q, xs, roots = ports(dtype)
    g = spec.graph
    for cut in (0, 2, 5, 15):
        inputs = [x for x in xs if x.time < cut]
        expected = reference(g, model, q, inputs, cut, sealed_until=cut, mode="hst")
        actual = execute(native, g, model, q, inputs, cut, mode="hst")
        equivalent(expected, actual)
        a = [v for *_, v in expected.outputs]+[v.value for v in expected.continuation.pending]
        b = [v for *_, v in actual.outputs]+[v.value for v in actual.continuation.pending]
        for left, right in zip(a, b): equivalent(gradients(left, roots), gradients(right, roots))
        tail = [x for x in xs if x.time >= cut]
        expected_tail = reference(g, model, expected.continuation, tail, 15, sealed_until=15, mode="hst")
        actual_tail = execute(native, g, model, actual.continuation, tail, 15, mode="hst")
        equivalent(expected_tail, actual_tail)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("policy", ["all", "selected", "clear", "blend"])
def test_history_selection_and_state_adoption_keep_causal_order(dtype, native, policy):
    g, m, q, xs, roots = history(dtype, policy=policy)
    expected = reference(g, m, q, xs, 6, sealed_until=6, mode="hst")
    actual = execute(native, g, m, q, xs, 6, mode="hst")
    equivalent(expected, actual)
    equivalent(gradients(objective(expected), roots), gradients(objective(actual), roots))
    for owner, h in expected.continuation.history.items():
        for field, value in h.tensors.items():
            equivalent(gradients(value, roots), gradients(actual.continuation.history[owner].tensors[field], roots))
