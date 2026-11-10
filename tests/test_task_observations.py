"""Plugin observation requirements bind execution, reservation and exact resume."""
from copy import deepcopy
from pathlib import Path

import pytest

from tidegraph.compare import equivalent
from tidegraph.experiment.config import resolve
from tidegraph.experiment.eager import Eager
from tidegraph.experiment.record_policy import needs_trace
from test_experiment_task_module import config
from test_selected_boundary_capacity import request as capacity_request


def selected_request(config, observations="outputs", client="pytorch"):
    value = deepcopy(config["requested"])
    value["task"].update(samples=2, positions=3)
    value["task"]["plugin"]["observations"] = observations
    value["execution"].update(return_mode="outputs", client=client)
    value["measurement"]["validation"] = "none"
    if client == "python-native":
        import _tide_native
        value["site"] = dict(paths=dict(native_library=str(Path(_tide_native.__file__).parent)))
    return value


@pytest.mark.parametrize("client", ["pytorch", "python-native"])
def test_output_plugin_trains_against_independent_reference_and_resumes(config, client, tmp_path):
    selected = resolve(selected_request(config, client=client))
    probe = Eager(selected)
    try:
        assert not probe.trace_records
        result = probe.forward(0)
        assert not result.trace and not result.messages
        assert result.stats["joint_state_vjp_events"] > 0
        assert probe.manifest()["internal_records"] == "boundary-only; output-observing task"
    finally:
        probe.close()
    candidate, reference = Eager(selected), Eager(selected, reference=True)
    complete = Eager(resolve(selected_request(config, "complete", client)))
    restored = None
    try:
        assert reference.trace_records and complete.trace_records
        for index in range(2):
            actual, loss = candidate.step(index)
            for other in (reference, complete):
                expected, expected_loss = other.step(index)
                equivalent(actual, expected, atol=1e-9, rtol=1e-8)
                assert loss == pytest.approx(expected_loss, rel=1e-8, abs=1e-9)
                equivalent(candidate.decoded_state(), other.decoded_state(), atol=1e-9, rtol=1e-8)
                equivalent(candidate.training_state(), other.training_state(), atol=1e-9, rtol=1e-8)
        assert candidate.training_state()["gradients"]["task.weight"] is not None
        candidate.save(tmp_path)
        before = complete.decoded_state()
        with pytest.raises(ValueError, match="task source/configuration identity"):
            complete.restore(tmp_path)
        equivalent(before, complete.decoded_state(), atol=0, rtol=0)
        restored = Eager(selected, checkpoint=tmp_path)
        equivalent(candidate.decoded_state(), restored.decoded_state(), atol=0, rtol=0)
        equivalent(candidate.step(2), restored.step(2), atol=0, rtol=0)
        equivalent(candidate.decoded_state(), restored.decoded_state(), atol=0, rtol=0)
        equivalent(candidate.training_state(), restored.training_state(), atol=0, rtol=0)
        diagnostics, _ = candidate.step(3, return_mode="complete")
        assert diagnostics.trace and diagnostics.messages
    finally:
        candidate.close(); reference.close(); complete.close()
        if restored is not None:
            restored.close()


def test_declared_observations_change_task_identity_and_capacity_without_allocation():
    value = capacity_request()
    value["task"]["plugin"] = dict(module="example_downstream_task")
    before = resolve(value)
    value["task"]["plugin"]["observations"] = "outputs"
    after = resolve(value)
    assert before["task_id"] != after["task_id"] and before["config_id"] != after["config_id"]
    a = before["capacity_budget"]["per_device"]["npu:0"]["components"]
    b = after["capacity_budget"]["per_device"]["npu:0"]["components"]
    assert a["live_record_estimate"] == a["complete_record_estimate"]
    assert b["live_record_estimate"] < b["complete_record_estimate"]
    assert {k: v for k, v in a.items() if k != "live_record_estimate"} == {
        k: v for k, v in b.items() if k != "live_record_estimate"}
    assert needs_trace(before["execution"]["effective"], before["task"])
    assert not needs_trace(after["execution"]["effective"], after["task"])


def test_frozen_policies_and_explicit_complete_preserve_plugin_roots(config):
    resolved = resolve(selected_request(config))
    spec, task = resolved["execution"]["effective"], resolved["task"]
    assert not needs_trace(spec, task)
    assert needs_trace(spec, task, reference=True)
    assert needs_trace(spec | {"return_mode": "complete"}, task)
    assert needs_trace(spec | {"record_policy": "selected-boundary-v1; task plugins retain complete records"}, task)
    spec.pop("record_policy")
    assert needs_trace(spec, task)


@pytest.mark.parametrize("invalid", [None, False, [], {}, "trace", "metrics"])
def test_invalid_observation_declarations_fail_before_execution(config, invalid):
    with pytest.raises(ValueError, match="task.plugin.observations"):
        resolve(selected_request(config, invalid))
