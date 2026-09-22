from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.reference import run
from coordinate_cases import IMPLEMENTATIONS, MALFORMED, fixture, runner, corrupt


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("field,value", [("time", .5), ("time", float("nan")), ("time", True),
                                        ("position", 0.), ("position", False), ("position", 2**63),
                                        ("batch", True), ("batch", 1.), ("port", False), ("port", 0.)])
def test_invalid_external_types_are_rejected_before_consumption_and_retry(dtype, implementation, field, value):
    g, m, q, xs = fixture(dtype); apply, cursor = runner(implementation, g, m, q)
    before = deepcopy((q, m.state_dict()))
    bad = [xs[0], replace(xs[1], **{field: value})]
    # A generator must not be consumed twice or partly applied before rejection.
    with pytest.raises(ValueError, match="int64"):
        apply(iter(bad), 3, 3)
    equivalent(before, (q, m.state_dict()))
    if cursor:
        assert not cursor.failed and cursor.cut == 0
        equivalent(q, cursor.snapshot())
    actual = apply(iter(xs), 3, 3)
    assert actual.outputs and actual.continuation.ledger
    equivalent(run(g, m, q, xs, 3, sealed_until=3), actual)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("field", ["stop", "seal"])
@pytest.mark.parametrize("value", [True, 3., 2**63, -(2**63)-1])
def test_window_types_are_checked_before_execution(dtype, implementation, field, value):
    g, m, q, xs = fixture(dtype); apply, cursor = runner(implementation, g, m, q)
    before = deepcopy((q, m.state_dict()))
    with pytest.raises(ValueError, match="int64"):
        apply(xs, value if field == "stop" else 3, value if field == "seal" else 3)
    equivalent(before, (q, m.state_dict()))
    if cursor:
        assert not cursor.failed and cursor.cut == 0
        equivalent(q, cursor.snapshot())
    equivalent(run(g, m, q, xs, 3, sealed_until=3), apply(xs, 3, 3))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("name", MALFORMED)
@pytest.mark.parametrize("convert", [bool, float], ids=["bool", "float"])
def test_imported_continuation_coordinates_cannot_be_coerced(dtype, implementation, name, convert):
    g, m, q, _ = fixture(dtype, continued=True)
    valid = deepcopy(q); corrupt(q, name, convert); before = deepcopy((q, m.state_dict()))
    with pytest.raises(ValueError, match="int64"):
        runner(implementation, g, m, q)[0]([], 3, 3)
    equivalent(before, (q, m.state_dict()))
    expected = run(g, m, valid, [], 3, sealed_until=3)
    equivalent(expected, runner(implementation, g, m, valid)[0]([], 3, 3))


@pytest.mark.parametrize("field", ["region", "budget", "source", "target", "input", "output"])
@pytest.mark.parametrize("convert", [bool, float, lambda _: 2**63], ids=["bool", "float", "overflow"])
def test_graph_indices_and_budgets_are_integer_typed(field, convert):
    g, _, _, _ = fixture(torch.float64)
    with pytest.raises(ValueError):
        if field == "region": replace(g, nodes=(replace(g.nodes[0], region=convert(0)), g.nodes[1]))
        elif field == "budget": replace(g, regions=(replace(g.regions[0], budget=convert(1)), g.regions[1]))
        elif field in {"source", "target"}: replace(g, edges=(replace(g.edges[0], **{field: convert(getattr(g.edges[0], field))}),))
        else: replace(g, **{field+"s": (convert(getattr(g, field+"s")[0]),)})
