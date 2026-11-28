"""Migration preserves the original learning segment and exact predecessor state."""
import hashlib
import json

import pytest

from pilotlab.gate3 import read_evidence
from pilotlab.runtime_migration3 import (
    IMAGE, IMAGE_DIGEST, SOURCE, TARGET, build_runtime_plan, inherited_evaluation,
    read_plan, valid_allocation, verify_result,
)


def write(path, data):
    path.write_text(json.dumps(data, indent=2))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(tmp_path):
    identity = {"python_version": SOURCE["python"], "runtime_packages": {k: v for k, v in SOURCE.items() if k != "python"}}
    task = dict(id="one-a2", phase="a2", model_id="one", source_identity=identity, device="cpu",
                resume_checkpoint="/local/parent.pt", resume_checkpoint_sha256="0" * 64, batch_size=2,
                sequence_length=8, evaluation_start_tokens=48, target_tokens=128, eval_tokens=8, eval_interval_tokens=64)
    fp = write(tmp_path / "task", task)
    cp = dict(steps=4, tokens=64, path="/old/end.pt", sha256="a" * 64, state_sha256="b" * 64, size_bytes=20)
    initial = dict(status="completed", tokens=8, requested_tokens=8, nll=4.2, split="selection", seconds=1.2)
    result = {**{k: task[k] for k in ("id", "phase", "model_id", "source_identity")}, "task_sha256": fp,
              "schema": "pilot-r3-worker-result-v1", "status": "budget_exhausted", "stop_reason": "signal",
              "health": {"blocking_reasons": ["evaluation_incomplete"]}, "causality": {"within_tolerance": True},
              "evaluation": {"tokens": 8}, "validation_initial": initial, "checkpoint": cp, "device": "cpu",
              "resume": {"sha256": task["resume_checkpoint_sha256"], "fallback_used": False,
                         "runtime_override": False, "from_tokens": 48, "checkpoint_load": {"status": "exact"}},
              "training": {"from_tokens": 48, "from_steps": 3, "tokens": 64, "steps": 4,
                           "segment_tokens": 16, "segment_steps": 1}}
    old_sha = write(tmp_path / "before", result)
    plan = build_runtime_plan(tmp_path / "task", prior_results=[{"path": tmp_path / "before", "sha256": old_sha}],
                              reserved_evaluation_tokens=72)
    plan_sha = write(tmp_path / "plan", plan)
    allocation = dict(name="allocated-job", region="cn-southwest-2", image=IMAGE, image_digest=IMAGE_DIGEST)
    runtime = dict(schema="tide-runtime-attempt/v1", runtime_plan_sha256=plan_sha, observed=TARGET, declared=SOURCE,
                   allocation=allocation, run="allocated-job", prior_result_hashes=[old_sha])
    current = {**result, "execution_runtime": runtime, "status": "completed", "stop_reason": None,
               "evaluation": dict(tokens=8, prior_attempt_tokens=8, cumulative_task_observed_tokens=16),
               "validation_initial": inherited_evaluation(plan, task),
               "resume": {**result["resume"], "source_task_id": task["id"], "runtime_override": True,
                          "from_tokens": 64, "from_steps": 4, "sha256": cp["sha256"], "state_sha256": cp["state_sha256"]},
               "training": dict(tokens=128, steps=8, from_tokens=64, from_steps=4, segment_steps=4, segment_tokens=64)}
    return task, plan, plan_sha, result, current


def test_migrated_attempt_reopens_with_its_original_stage_parent(tmp_path):
    task, plan, checksum, before, current = inputs(tmp_path)
    assert plan["evaluation"]["future_upper_bound"] == 40 and plan["evaluation"]["additional_reservation"] == 8
    verify_result(task, current, plan, [before], checksum)
    write(tmp_path / "cloud", current)
    entry = {"task_path": str(tmp_path / "task"), "result_path": str(tmp_path / "cloud"),
             "runtime_plan": {"path": str(tmp_path / "plan"), "sha256": checksum}}
    _, result, evidence = read_evidence(entry)
    assert result.origin["training"]["from_tokens"] == 48 and result.origin["resume"]["path"] == task["resume_checkpoint"]
    assert result["training"]["from_tokens"] == 64 and evidence["runtime_plan"] == entry["runtime_plan"]
    with pytest.raises(ValueError, match="must be read together"):
        read_evidence({k: v for k, v in entry.items() if k != "runtime_plan"})


def test_migration_rejects_changed_bytes_state_image_and_repeated_evaluation(tmp_path):
    task, plan, checksum, before, actual = inputs(tmp_path)
    for changed in ({"resume": {**actual["resume"], "state_sha256": "changed"}},
                    {"execution_runtime": {**actual["execution_runtime"], "observed": SOURCE}},
                    {"validation_initial": before["validation_initial"]},
                    {"evaluation": dict(tokens=16, prior_attempt_tokens=8, cumulative_task_observed_tokens=24)}):
        with pytest.raises(ValueError):
            verify_result(task, {**actual, **changed}, plan, [before], checksum)
    with pytest.raises(ValueError):
        valid_allocation(actual["execution_runtime"]["allocation"], "another-job")
    old_ref = plan["prior_results"][0]
    old_ref["json"] += "\n"
    with pytest.raises(ValueError, match="predecessor"):
        read_plan(tmp_path / "plan", write(tmp_path / "plan", plan), task, actual["task_sha256"])
