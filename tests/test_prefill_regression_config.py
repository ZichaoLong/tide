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
        assert limits["stages"] > value["task"]["samples"]*value["task"]["positions"]*plan["plan"]["graph"]["regions"]
        value["resources"]["resident_limits"]["tensor_bytes"] = 1
        assert resolve(value)["capacity_budget"]["resident_layout"]["exceeds_tensor_bytes"]


def test_cpu_and_mixed_candidates_do_not_acquire_resident_reservations():
    assert "resident_limits" not in config("query-fc")["resources"]
    assert "resident_limits" not in config("query-fc", device="npu:0")["resources"]
