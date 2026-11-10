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


@pytest.mark.parametrize("schedule", ["auto", "window", "greedy"])
@pytest.mark.parametrize("query", ["auto", "legacy"])
@pytest.mark.parametrize("returns", ["complete", "outputs"])
def test_execution_record_round_trip_preserves_omitted_historical_defaults(schedule, query, returns):
    record = ExecutionSpec(prefill_schedule=schedule, query_execution=query, return_mode=returns).record()
    assert ExecutionSpec.from_record(record).record() == record


def test_public_comparison_matrix_accepts_explicit_greedy_and_window():
    from tidegraph.experiment.config import resolve
    from test_experiment_config import example
    from copy import deepcopy
    first = example()
    first["execution"].update(return_mode="outputs", prefill_schedule="auto")
    first["measurement"] = dict(validation="none")
    second = deepcopy(first)
    second["execution"]["prefill_schedule"] = "greedy"
    a, b = resolve(first), resolve(second)
    assert a["model_id"] == b["model_id"] and a["workload_id"] == b["workload_id"]
    assert a["execution_id"] != b["execution_id"]
    assert resolve(b)["config_id"] == b["config_id"]
