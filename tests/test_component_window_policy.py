"""Requested/effective schedule identities and historical opt-in reproduction."""
import pytest
from tidegraph.planning import ExecutionSpec


def test_prefill_policy_is_explicit_and_old_greedy_identity_is_reproducible():
    records = [ExecutionSpec(prefill_schedule=s).record() for s in ("auto", "window", "greedy")]
    assert len({r["execution_id"] for r in records}) == 3
    for r in records[:2]:
        assert r["schema"] == "tide-weighted-execution-v3"
        assert r["effective"]["scheduler"] == "component-window"
        assert r["effective"]["prefill_schedule"] == "window"
    assert "prefill_schedule" not in records[2]["requested"]
    for config in (dict(schedule="streaming"), dict(client="python-native", device="npu:0", placement="resident")):
        assert "prefill_schedule" not in ExecutionSpec(**config).record()["effective"]
        with pytest.raises(ValueError, match="host prefill"):
            ExecutionSpec(**config, prefill_schedule="window")
