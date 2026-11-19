"""Finite candidate choice and resource-refusal rules, without device execution."""
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.memory_candidate_suite import (GIB, _inference_child, _train_children, budget_refusal,
                                               next_screen, project_budget, select_best)


def _point(batch, allocated, reserved, *, rate=10, formal=False, identity=None):
    return dict(identity=identity or f"b{batch}", batch=batch, within_budget=True, formal=formal,
                stable_decode_positions_per_second=rate, parameter_bytes=32*GIB,
                peak_bytes=dict(allocated=allocated*GIB, reserved=reserved*GIB))


def test_selection_preserves_common_formal_and_screen_identities():
    common = _point(8, 40, 44, rate=12, formal=True)
    screen = _point(16, 45, 49, rate=14, identity="screen16")
    confirmed = _point(16, 45, 49, rate=11.9, formal=True, identity="confirm16")
    points = [common, screen, confirmed, dict(_point(32, 60, 61, rate=50), within_budget=False)]
    assert select_best(points) is screen
    assert select_best(points, formal_only=True) is common
    assert points[1]["identity"] != points[2]["identity"]


def test_ties_choose_lower_batch_and_nonfinite_is_never_selected():
    small, large = _point(8, 40, 44), _point(16, 44, 48)
    assert select_best([large, small, _point(32, 45, 49, rate=float("nan"))]) is small
    assert select_best([]) is None


def test_budget_projection_uses_both_allocator_limits_and_measured_slopes():
    points = [_point(1, 34, 35), _point(8, 42, 45)]
    prediction = project_budget(points, 16)
    assert prediction["state"] == "projected_budget_refusal"
    assert prediction["predicted_peak_bytes"]["allocated"] > 48*GIB
    assert prediction["source_identities"] == ["b1", "b8"]
    # Largest fitting rung below twice the current batch is 12.
    selected, forecasts = next_screen(points)
    assert selected == 12
    assert [row["batch"] for row in forecasts] == [12, 16]
    reserved_limited = [_point(1, 33, 36), _point(8, 35, 53)]
    assert next_screen(reserved_limited)[0] is None


def test_flat_reserved_measurement_does_not_erase_earlier_growth():
    points = [_point(1, 33, 34), _point(8, 35, 42), _point(16, 37, 40)]
    forecast = project_budget(points, 24)
    assert forecast["slope_bytes_per_batch"]["reserved"] == pytest.approx(8*GIB/7)
    assert forecast["predicted_peak_bytes"]["reserved"] > 49*GIB


def test_confirmation_peak_and_absent_device_observation_remain_conservative():
    points = [_point(1, 33, 34), _point(8, 35, 37), _point(8, 39, 41, identity="confirm8")]
    forecast = project_budget(points, 12)
    assert forecast["predicted_peak_bytes"]["allocated"] >= 39*GIB
    cpu = dict(_point(8, 0, 0), peak_bytes=dict(allocated=None, reserved=None))
    assert project_budget([cpu], 12)["state"] == "unavailable"
    assert next_screen([cpu])[0] is None
    with pytest.raises(ValueError, match="larger batch"):
        project_budget(points, 8)


@pytest.mark.parametrize("result,expected", [
    ({"error": "MemoryError: npu:0 observed allocated peak exceeds 48 GiB sampled budget"}, True),
    ({"failure_kind": "device_budget_exceeded"}, True),
    ({"error": "DeviceBudgetExceeded('sampled peak')"}, True),
    ({"error": "RuntimeError: NPU out of memory"}, False),
    ({"error": "MemoryError: could not allocate memory"}, False),
    ({"error": "ArithmeticError: nonfinite inference diagnostic"}, False),
])
def test_sampled_budget_refusal_is_not_hardware_oom_or_semantic_failure(result, expected):
    assert budget_refusal(result) == expected


def test_commands_preserve_training_batch_and_distinct_normal_observation_counts(tmp_path):
    args = Namespace(config=Path("/frozen/candidate.json"), device="npu:0", threads=4)
    children = _train_children(args, tmp_path, 8)
    assert [row["phase"] for row in children] == ["train", "restore", "ledger"]
    assert all(row["command"][row["command"].index("--batch")+1] == "8" for row in children)
    assert "--continue-state" in children[-1]["command"]
    assert all(row["timeout_seconds"] == 1800 for row in children)
    formal = _inference_child(args, tmp_path, 8, formal=True, label="common")
    screen = _inference_child(args, tmp_path, 16, formal=False, label="screen")
    assert (formal["warmup"], formal["samples"], screen["warmup"], screen["samples"]) == (2, 5, 1, 2)
    assert all(row["command"][row["command"].index("--audit")+1] == "full" for row in (formal, screen))
    assert len({row["output"] for row in [*children, formal, screen]}) == 5


@pytest.mark.parametrize("outcome,expected_batches,expected_state", [
    ("budget_refused", [1, 8, 4], "completed_with_budget_stop"),
    ("failed", [1, 8], "failed"),
])
def test_only_sampled_b8_refusal_allows_one_formal_b4_fallback(monkeypatch, tmp_path, outcome, expected_batches, expected_state):
    from benchmarks import memory_candidate_suite as suite
    args = Namespace(config=Path("/frozen/candidate.json"), device="npu:0", threads=4)
    record = dict(children=[], measurements=[], decisions=[])
    def execute(child, record, persist):
        child["outcome"] = outcome if child["batch"] == 8 else "completed"
        record["children"].append(child)
        return {}
    monkeypatch.setattr(suite, "_execute", execute)
    monkeypatch.setattr(suite, "inference_point", lambda result, child: _point(
        child["batch"], 34+child["batch"], 35+child["batch"], rate=child["batch"],
        formal=child["formal"], identity=child["identity"]))
    assert suite.run_inference(args, tmp_path, record, lambda: None) == expected_state
    assert [row["batch"] for row in record["children"]] == expected_batches
    assert record["children"][1]["outcome"] == outcome
    if outcome == "budget_refused":
        assert record["best_formal_measured"] == "budget-fallback-b4"
        assert all(point["formal"] for point in record["measurements"])
