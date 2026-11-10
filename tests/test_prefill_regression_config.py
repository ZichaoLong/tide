"""Finite resident representatives reserve actual raw/journal partitions."""
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
from prefill_regression_config import config
from tidegraph.experiment.config import resolve


def test_resident_suite_bounds_cover_all_possible_events_and_retention():
    for case, family in (("query-fc", "kv"), ("layer-chain", "ssd")):
        value = config(case, family=family, device="npu:0", placement="resident")
        plan = resolve(value)
        limits = value["resources"]["resident_limits"]
        actual = plan["capacity_budget"]["resident_layout"]
        assert not actual["exceeds_tensor_bytes"]
        assert limits["retained_bytes"] >= 3*actual["preflight_total_bytes"]
        assert limits["total_bytes"] == sum(limits[n] for n in
            ("tensor_bytes", "retained_bytes", "backward_bytes", "optimizer_bytes"))
        assert limits["cache"] > 3*value["task"]["positions"]
        assert limits["stages"] > value["task"]["positions"]*plan["plan"]["graph"]["regions"]
        # Actual predecessor failures: layout receives only 1/16 of optimizer
        # quota; reverse reserves four complete views before the numeric tape.
        assert limits["optimizer_bytes"]//16 > 5*plan["inventory"]["totals"]["unique"]
        assert limits["backward_bytes"] > 4*actual["preflight_total_bytes"]
        value["resources"]["resident_limits"]["tensor_bytes"] = 1
        assert resolve(value)["capacity_budget"]["resident_layout"]["exceeds_tensor_bytes"]


def test_cpu_and_mixed_candidates_do_not_acquire_resident_reservations():
    assert "resident_limits" not in config("query-fc")["resources"]
    assert "resident_limits" not in config("query-fc", device="npu:0")["resources"]


def test_independent_samples_do_not_multiply_the_journal_stage_bound():
    small = resolve(config("query-fc", samples=2, device="npu:0", placement="resident"))
    large = resolve(config("query-fc", samples=16, device="npu:0", placement="resident"))
    a, b = [x["capacity_budget"]["resident_layout"] for x in (small, large)]
    assert a["limits"]["stages"] == b["limits"]["stages"]
    # Fixed L/graph: per-stage rows and the simultaneous queue scale with B,
    # not B squared. Keep constant spare rows; no pre-run selection is used.
    assert a["preflight_total_bytes"] < b["preflight_total_bytes"] <= 8*a["preflight_total_bytes"]
