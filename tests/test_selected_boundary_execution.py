"""Selected returns optimize hidden KV states without changing the task or recovery."""
from copy import deepcopy
from pathlib import Path
import pytest
from tidegraph.compare import equivalent
from tidegraph.experiment.config import resolve
from tidegraph.experiment.eager import Eager
from test_experiment_engine import config


@pytest.mark.parametrize("client", ["pytorch", "python-native"])
@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
def test_selected_boundary_preserves_task_training_raw_resume_and_work(client, family, tmp_path):
    request = config("training", client=client, dtype="float64", scoring_dtype="float64")
    if client == "python-native":
        import _tide_native
        request["site"] = dict(paths=dict(native_library=str(Path(_tide_native.__file__).parent)))
    request["model"]["recipe"].update(family=family)
    selected = deepcopy(request)
    selected["execution"]["return_mode"] = "outputs"
    selected["measurement"]["validation"] = "none"
    complete, boundary = Eager(resolve(request)), Eager(resolve(selected))
    try:
        for i in range(2):
            a, x = complete.step(i)
            b, y = boundary.step(i)
            equivalent(a.outputs, b["outputs"])
            assert x == pytest.approx(y, rel=1e-9, abs=1e-12)
            equivalent(complete.decoded_state(), boundary.decoded_state())
            equivalent(complete.training_state(), boundary.training_state())
            for field in ("candidate_updates", "full_calls", "messages", "outputs", "pending"):
                assert complete.last_work[field] == boundary.last_work[field]
            assert boundary.last_work["candidate_updates"] > 0
            assert boundary.last_work["effective_window_execution"]["schedule"] == "settle-columns-v1"
            assert boundary.last_work["actual_stats"]["state_column_calls"] > 0
        boundary.save(tmp_path)
        restored = Eager(resolve(selected), checkpoint=tmp_path)
        try:
            equivalent(boundary.decoded_state(), restored.decoded_state(), atol=0, rtol=0)
            equivalent(boundary.step(2), restored.step(2), atol=0, rtol=0)
            equivalent(boundary.decoded_state(), restored.decoded_state(), atol=0, rtol=0)
            equivalent(boundary.training_state(), restored.training_state(), atol=0, rtol=0)
        finally:
            restored.close()
        # An explicit diagnostic request still supplies all intermediate roots.
        result, _ = boundary.step(3, return_mode="complete")
        assert result.trace and result.messages
        assert boundary.manifest()["effective_input_interface"] == "records"
    finally:
        complete.close(); boundary.close()


def test_untraced_work_counts_do_not_confuse_full_batches_with_selected_events():
    from types import SimpleNamespace
    from tidegraph.experiment.eager import work
    result = SimpleNamespace(trace=[], messages=[], outputs=[],
        continuation=SimpleNamespace(pending=[]),
        stats=dict(candidate_events=23, selected_events=17, full_calls=4, visited_edges=51))
    row = work(result)
    assert (row["candidate_updates"], row["full_calls"], row["messages"]) == (23, 17, 51)


def test_pre_policy_frozen_configuration_keeps_its_internal_roots():
    request = config("training", client="pytorch", dtype="float64", scoring_dtype="float64")
    request["execution"]["return_mode"] = "outputs"
    request["measurement"]["validation"] = "none"
    resolved = resolve(request)
    resolved["execution"]["effective"].pop("record_policy")
    owner = Eager(resolved)
    try:
        assert owner.trace_records
        result = owner.forward(0)
        assert result.trace and result.messages
        assert result.stats.get("joint_state_vjp_events", 0) == 0
    finally:
        owner.close()
