"""Exploration may classify finite value errors, never erase other failures."""
from dataclasses import dataclass
import json
import pytest
import torch
from tidegraph.compare import (equivalent, collect_comparison, NumericMismatch,
                               DiscreteMismatch, StructureMismatch, NonFiniteMismatch)


@dataclass
class Observations:
    values: dict
    stats: dict


def test_numeric_report_retains_original_threshold_and_typed_strict_failure():
    left, right = torch.tensor([1.]), torch.tensor([1.01])
    with pytest.raises(NumericMismatch) as failure:
        equivalent(left, right)
    assert isinstance(failure.value, AssertionError)
    report = collect_comparison(left, right)
    assert report["finite"] == report["discrete"] == "passed"
    assert report["numeric"] == "failed" and not report["strict_passed"]
    assert report["differences"][0]["atol"] == 1e-6
    assert report["differences"][0]["rtol"] == 1e-5
    equivalent(left, right, atol=.02, rtol=0.)


def test_all_discrete_leaves_checked_after_numeric_error_and_bounded_diagnostics():
    left = Observations({"payload": torch.tensor(1.), "time": torch.tensor(2**63-1),
                         "gradient": None}, {"launches": 1})
    right = Observations({"payload": torch.tensor(2.), "time": torch.tensor(2**63-2),
                          "gradient": torch.tensor(0.)}, {"launches": float("nan")})
    report = collect_comparison(left, right, max_differences=1)
    assert report["numeric"] == "not_comparable" and report["discrete"] == "failed"
    assert report["mismatch_counts"] == dict(numeric=1, discrete=2)
    assert report["differences_truncated"] and len(report["differences"]) == 1


@pytest.mark.parametrize("left,right", [
    ({"root": None}, {"root": torch.tensor(0.)}),
    ({"route": 1}, {"other_route": 1}),
    ([1, 2], [1]),
    (torch.tensor([2**53]), torch.tensor([2**53+1])),
    (torch.tensor([True]), torch.tensor([False])),
])
def test_discrete_mismatch_never_numeric_exploration(left, right):
    with pytest.raises(DiscreteMismatch):
        equivalent(left, right, atol=100., rtol=100.)
    report = collect_comparison(left, right, atol=100., rtol=100.)
    assert report["discrete"] == "failed" and not report["strict_passed"]


@pytest.mark.parametrize("left,right", [
    (torch.tensor([1.]), torch.tensor(1.)),
    (torch.tensor([1.]), torch.tensor([1.], dtype=torch.float64)),
    (Observations({}, {}), {}),
    ({"root": []}, {"root": {}}),
])
def test_structure_errors_are_not_collected(left, right):
    with pytest.raises(StructureMismatch):
        collect_comparison(left, right)


@pytest.mark.parametrize("left,right", [
    (torch.tensor(float("nan")), torch.tensor(0.)),
    (torch.tensor(float("inf")), torch.tensor(float("inf"))),
    ({"unexpected": torch.tensor(float("inf"))}, {}),
    ({"scalar": float("nan")}, {"scalar": 0.}),
])
def test_nonfinite_even_on_unmatched_routes_is_fatal(left, right):
    with pytest.raises(NonFiniteMismatch):
        collect_comparison(left, right)


def test_complete_report_and_statistics_ignore_contract():
    a = Observations({"value": torch.tensor([1.]), "absent": None}, {"calls": 1})
    b = Observations({"value": torch.tensor([1.]), "absent": None}, {"calls": 999})
    equivalent(a, b)
    result = collect_comparison(a, b)
    assert result["strict_passed"] and not result["differences"]
    assert result["compared_leaves"] == dict(numeric=1, discrete=1)


def test_finite_operands_with_overflowing_error_have_json_safe_diagnostics():
    report = collect_comparison(torch.tensor([60000.], dtype=torch.float16),
                                torch.tensor([-60000.], dtype=torch.float16))
    assert report["numeric"] == "failed" and report["finite"] == "passed"
    assert report["differences"][0]["max_abs_error"] == "overflow"
    json.dumps(report, allow_nan=False)
