"""Bound ancestry and finite controller relay; no real campaign mutations."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pilotlab import prepare_relay3 as relay, resume3
from pilotlab.io import sha256
from pilotlab.scheduler3 import Blocked


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return relay.reference(path)


@pytest.fixture
def lineage(tmp_path):
    identity = {"source_files": {}, "data_manifest_sha256": "data", "runtime_packages": {}, "python_version": "same"}
    task = {"id": "model-a1", "model_id": "model", "phase": "a1", "target_tokens": 64,
            "source_identity": identity, "eval_tokens": 16, "device": "cpu",
            "evaluation_budget_key": "model", "evaluation_token_limit": 1024}
    original = write(tmp_path/"r04/task.json", task)
    result = write(tmp_path/"r04/attempt-0/result.json", {"schema": "pilot-r3-worker-result-v1",
        "id": task["id"], "model_id": "model", "phase": "a1", "status": "completed",
        "task_sha256": original["sha256"], "source_identity": identity,
        "training": {"tokens": 64}, "health": {"eligible": True}})
    review = write(tmp_path/"r04/review.json", {"status": "supplemental_review_completed"})
    retirement = write(tmp_path/"r04/retired.json", {"schema": "pilot-r3-retirement-v1", "files": []})
    old_import = write(tmp_path/"r05/import.json", {"schema": "pilot-r3-resume-import-v1",
        "reconciliation": review, "worker_source_identity": identity, "prior_retirement": retirement,
        "tasks": {task["id"]: {"task": original, "result": result, "worker_return_code": 0,
            "disposition": "completed", "attempt_path": str(tmp_path/"r04/attempt-0")}}})
    copied = write(tmp_path/"r05/round/tasks/model-a1/task.json", task)
    assert copied["sha256"] == original["sha256"]
    record = {"task_path": copied["path"], "status": "completed", "stage": "a1",
              "attempts": [], "prior_attempts": [{"path": str(tmp_path/"r04/attempt-0")}],
              "imported_completion": True, "result_path": result["path"], "result_sha256": result["sha256"],
              "evaluation_reserved_tokens": 64, "evaluation_budget_key": "model"}
    return task, record, old_import, retirement


def test_per_attempt_binding_and_recursive_retirements(tmp_path, lineage):
    task, record, parent, retirement = lineage
    history, evidence = relay.attempt_evidence({"config": {"resume_import": parent}}, task["id"], record)
    row = relay.classify(task, record, evidence, task["source_identity"])
    row["inherited_from"] = {"import": parent, "task_id": task["id"]}
    review = write(tmp_path/"r06/review.json", {"status": "passed"})
    manifest = {"schema": "pilot-r3-resume-import-v1", "reconciliation": review,
        "worker_source_identity": task["source_identity"], "prior_imports": [parent], "tasks": {task["id"]: row}}
    config = {"resume_import": write(tmp_path/"r06/import.json", manifest)}
    entry, original, result = resume3.imported_attempt(config, task["id"], history[0]["path"])
    assert original == task and result["status"] == "completed" and entry["worker_return_code"] == 0
    assert resume3.retirement_references(manifest) == [retirement]
    with pytest.raises(Blocked, match="missing_or_ambiguous"):
        resume3.imported_attempt(config, task["id"], tmp_path/"unrelated")
    Path(parent["path"]).write_text('{}')
    with pytest.raises(Blocked, match="reference_changed"):
        resume3.imported_entry(config, task["id"])


def test_capacity_retry_is_explicit_and_unknown_math_failures_block(tmp_path):
    task = {"id": "capacity", "model_id": "m", "phase": "preflight", "purpose": "capacity_calibration",
            "source_identity": {"frozen": True}, "target_tokens": 64, "eval_tokens": 16}
    task_ref = write(tmp_path/"task.json", task)
    record = {"task_path": task_ref["path"], "status": "failed", "evaluation_reserved_tokens": 64}
    evidence = [{"task": task_ref, "result": None, "attempt_path": str(tmp_path/"attempt-0"),
                 "disposition": "missing_result_after_start", "worker_return_code": -9}]
    row = relay.classify(task, record, evidence, task["source_identity"])
    assert row["disposition"] == "retry" and row["evaluation_reserved_tokens"] == 64
    assert row["additional_evaluation_reserved_tokens"] == 64 and "no recovery claim" in row["retry_scope"]
    value = {"id": "capacity", "source_identity": task["source_identity"], "task_sha256": task_ref["sha256"],
             "status": "budget_exhausted", "stop_reason": "signal", "checkpoint": {"saved": True}}
    evidence[0]["result"] = write(tmp_path/"attempt-0/result.json", value)
    assert relay.classify(task, record, evidence, task["source_identity"])["disposition"] == "retry"
    value["health"] = {"blocking_reasons": ["nonfinite gradient"]}
    evidence[0]["result"] = write(tmp_path/"attempt-0/result.json", value)
    with pytest.raises(ValueError, match="scientific failure"):
        relay.classify(task, record, evidence, task["source_identity"])
    value = {"id": "capacity", "source_identity": task["source_identity"], "task_sha256": task_ref["sha256"],
             "status": "failed", "failure_class": "numerical_error", "stop_reason": "nonfinite"}
    evidence[0]["result"] = write(tmp_path/"attempt-0/result.json", value)
    with pytest.raises(ValueError, match="unclassified"):
        relay.classify(task, record, evidence, task["source_identity"])
    value.update(status="recipe_rejected", failure_class="declared_optimizer_instability")
    evidence[0]["result"] = write(tmp_path/"attempt-0/result.json", value)
    rejected = relay.classify(task, record, evidence, task["source_identity"])
    assert rejected["disposition"] == "recipe_rejected"
    assert rejected["additional_evaluation_reserved_tokens"] == 0


def test_pending_task_bytes_must_match_terminal_audit(tmp_path):
    task_path = tmp_path/"tasks/pending/task.json"
    original = write(task_path, {"id": "pending"})
    review = {"artifacts": {"files": [dict(original, kind="task", task_id="pending", status="passed")]}}
    record = {"status": "pending", "attempts": []}
    normalized = relay.normalized_record(tmp_path, "pending", record, review)
    assert normalized["task_path"] == str(task_path) and "task_path" not in record
    task_path.write_text('{"id":"pending","changed":true}')
    with pytest.raises(ValueError, match="immutable task"):
        relay.normalized_record(tmp_path, "pending", record, review)


def test_terminal_review_refuses_live_or_stale_state(tmp_path, monkeypatch):
    old_path = tmp_path/"prepared.json"
    old = {"output": str(tmp_path/"round")}; write(old_path, old)
    state = {"status": "interrupted", "stage_b_gate": "closed", "all_seen_processes": {}}
    state_ref = write(tmp_path/"round/scheduler.json", state)
    review = {"schema": "pilot-r3-terminal-audit-v1", "status": "passed", "findings": [],
        "prepared": relay.reference(old_path), "round": old["output"], "execution": {"scheduler": state_ref},
        "lifecycle": {"status": "terminal", "processes": {"released": True}, "cgroup": {"released": True},
            "queue": {"released": True}, "systemd": {"properties": {"MainPID": "0"}}}}
    monkeypatch.setattr(relay, "live", lambda _: False)
    relay.validate_terminal(old_path, old, state, review)
    with pytest.raises(ValueError, match="stage A"):
        relay.validate_terminal(old_path, old, dict(state, status="running"), review)
    monkeypatch.setattr(relay, "live", lambda _: True)
    with pytest.raises(ValueError, match="live"):
        relay.validate_terminal(old_path, old, state, review)
    review["execution"]["scheduler"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="content-bound"):
        relay.validate_terminal(old_path, old, state, review)


@pytest.mark.parametrize("damage", [None,"scientific_failure","not_shutdown","repeated","bad_checkpoint","bad_heartbeat"])
def test_forced_replay_is_one_bound_shutdown_with_preserved_incomplete_evidence(tmp_path, damage):
    directory=tmp_path/"attempt";directory.mkdir()
    task={"id":"a1", "phase":"a1", "target_tokens":128}
    state={"status":"a_blocked_needs_repair","blocking_reason":"selected confirmation"}
    record={}
    group={"status":"failed","reason":"signal_15","owner":{"pid":99999999,"start_ticks":1}}
    worker={"return_code":-9,"cleanup_verified":True,"descendant_cleanup_verified":True,
        "remaining_descendants":[],"pid":99999998,"process":{"pid":99999998,"start_ticks":1}}
    write(directory/"result.json",{"status":"running","health":{"blocking_reasons":["not_evaluated"]}})
    (directory/"steps.jsonl").write_text(json.dumps({"step":1,"tokens":32,"loss":5.,"objective":5.,"grad_norm":.5,"lr":.001})+'\n')
    (directory/"health.jsonl").write_text(json.dumps({"observed_finite":damage!="scientific_failure"})+'\n')
    write(directory/"heartbeat.json",{"task_id":"a1","pid":worker["pid"],"tokens":32,"steps":1})
    cp=directory/"checkpoints/one.pt";cp.parent.mkdir();cp.write_bytes(b"retained incomplete state")
    write(directory/"checkpoint.json",{"schema":"pilot-r3-checkpoint-generations-v1",
        "current":{**relay.reference(cp),"tokens":32,"steps":1}})
    if damage=="not_shutdown":group["reason"]="worker_failed"
    elif damage=="repeated":record["prior_attempts"]=[{"path":"another-attempt"}]
    elif damage=="bad_checkpoint":cp.write_bytes(b"changed")
    elif damage=="bad_heartbeat":write(directory/"heartbeat.json",{"task_id":"other"})
    before={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    if damage:
        with pytest.raises(ValueError):relay.hard_stop_retry_proof(state,task,record,directory,group,worker)
    else:
        proof=relay.hard_stop_retry_proof(state,task,record,directory,group,worker)
        assert proof["observed_training_tokens"]==32 and proof["prior_training_upper_bound_tokens"]==128
        assert "no recovery claim" in proof["reason"]
    assert before=={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}


def test_forced_a3_replay_binds_same_recipe_parent_and_exact_exposure(tmp_path):
    parent = {"id":"m-a2-seed", "phase":"a2", "recipe_id":"r1", "seed":123,
              "source_identity":{"frozen":True}, "target_tokens":128}
    original = write(tmp_path/"a2-task.json", parent)
    cp = write(tmp_path/"a2-checkpoint", {"immutable":"state"})
    final = {"id":parent["id"], "status":"completed", "source_identity":parent["source_identity"],
             "task_sha256":original["sha256"], "training":{"tokens":128}, "health":{"eligible":True},
             "checkpoint":dict(cp, tokens=128, steps=4)}
    result = write(tmp_path/"a2-result.json", final)
    record = {"status":"completed", "attempts":[{"return_code":0}], "task_path":original["path"],
              "result_path":result["path"], "result_sha256":result["sha256"]}
    state = {"blocking_reason":"user_signal", "tasks":{parent["id"]:record}}
    task = dict(parent, id="m-a3-seed", phase="a3", evaluation_start_tokens=128, target_tokens=256,
                resume_checkpoint=cp["path"], resume_checkpoint_sha256=cp["sha256"])
    assert relay.hard_stop_a3_parent(state, task)["checkpoint"] == final["checkpoint"]
    for changes in ({"recipe_id":"another"}, {"evaluation_start_tokens":160}, {"resume_checkpoint_sha256":"changed"}):
        with pytest.raises(ValueError, match="identity, exposure or checkpoint"):
            relay.hard_stop_a3_parent(state, dict(task, **changes))
    Path(cp["path"]).write_text("changed")
    with pytest.raises(ValueError, match="identity, exposure or checkpoint"):
        relay.hard_stop_a3_parent(state, task)


def test_complete_relay_preserves_original_sources_reservations_and_task_bytes(tmp_path, monkeypatch, lineage):
    task, record, parent, retirement = lineage
    old_root, new_root = tmp_path/"r05", tmp_path/"r06"
    ledger = write(tmp_path/"budget.json", {"segments": {str(old_root/"round"): {"device_seconds": 123.}}})
    identity = task["source_identity"]
    common = {"python": "/existing/python", "data": "/existing/data", "budget_ledger": ledger["path"], "source_identity": identity}
    old = dict(common, source=str(old_root/"source"), foundation=str(old_root/"foundation"), output=str(old_root/"round"))
    old_path = old_root/"campaign-prepared.json"; write(old_path, old)
    adapter = new_root/"source/pilotlab/resume_adapter3.py"; adapter.parent.mkdir(parents=True); adapter.write_text("adapter")
    new = dict(common, source=str(new_root/"source"), foundation=str(new_root/"foundation"), output=str(new_root/"round"),
               run_id="relay", environment={"PYTHONPATH": "new-controller"})
    new_path = new_root/"prepared.json"; write(new_path, new)
    config = {**common, "source": old["source"], "foundation": old["foundation"],
        "worker_source": "/original/r04/source", "worker_foundation": "/original/r04/foundation",
        "resume_import": parent, "deadline_unix": 9000, "storage_reserve_bytes": 16,
        "max_rss_bytes": 1000, "max_cpu_workers": 40, "prior_evaluation_reservations": {"earlier": 12}}
    state = {"config": config, "status": "interrupted", "stage_b_gate": "closed", "tasks": {task["id"]: record},
             "all_seen_processes": {}}
    pending = dict(task, id="capacity-pending", phase="preflight", purpose="capacity_calibration")
    pending_ref = write(old_root/"round/tasks/capacity-pending/task.json", pending)
    state["tasks"][pending["id"]] = {"status": "pending", "stage": "capacity-large", "attempts": []}
    state_ref = write(old_root/"round/scheduler.json", state)
    plan_ref = write(old_root/"campaign-plan.json", {"config": config, "source_identity": identity,
        "subjects": [{"id": "fixed"}], "protocol": {}, "worker_prepared": {"path": "r04", "sha256": "unchanged"}})
    write(old_root/"campaign-launch-draft.json", {"plan_sha256": plan_ref["sha256"], "prepared_sha256": sha256(old_path)})
    review = {"schema": "pilot-r3-terminal-audit-v1", "status": "passed", "findings": [],
        "prepared": relay.reference(old_path), "round": old["output"], "execution": {"scheduler": state_ref},
        "artifacts": {"files": [dict(relay.reference(record["task_path"]), kind="task", task_id=task["id"], status="passed"),
                                dict(pending_ref, kind="task", task_id=pending["id"], status="passed")]},
        "lifecycle": {"status": "terminal", "processes": {"released": True}, "cgroup": {"released": True},
            "queue": {"released": True}, "systemd": {"properties": {"MainPID": "0"}}}}
    review_ref = write(tmp_path/"terminal.json", review)
    monkeypatch.setattr(relay, "source_inventory", lambda _: ({}, {}))
    monkeypatch.setattr(relay, "live", lambda _: False)
    monkeypatch.setattr(relay, "verify_resume_payloads", lambda *_: None)
    monkeypatch.setattr(relay, "budget_remaining", lambda *_: {"started_unix": 1,"deadline_unix": 9000,"remaining_wall_seconds": 100})
    monkeypatch.setattr(relay, "storage_observation", lambda *_args, **_kwargs: {"checked": True})
    before = {p: sha256(p) for p in (old_path, Path(state_ref["path"]), Path(ledger["path"]))}
    draft = relay.prepare(new_path, old_path, review_ref["path"])
    migrated = json.loads((new_root/"round/scheduler.json").read_text())
    manifest = json.loads((new_root/"resume-import.json").read_text())
    assert migrated["config"]["dynamic_shared_admission"] is True
    assert migrated["config"]["dynamic_probe_owners"] == 1
    assert migrated["config"]["dynamic_owner_max_total_workers"] == 256
    assert migrated["config"]["dynamic_owner_max_commands"] == 257
    assert migrated["config"]["worker_source"] == config["worker_source"]
    assert migrated["config"]["worker_foundation"] == config["worker_foundation"]
    assert migrated["config"]["max_cpu_workers"] == 40
    adopted = migrated["tasks"][task["id"]]
    assert adopted["prior_attempts"] == record["prior_attempts"] and adopted["attempts"] == []
    assert adopted["evaluation_reserved_tokens"] == 64 and adopted["device_seconds"] == 0
    assert sha256(adopted["task_path"]) == sha256(record["task_path"])
    assert manifest["prior_retirements"] == [retirement]
    assert manifest["tasks"][task["id"]]["inherited_from"]["import"] == parent
    assert "task_path" not in state["tasks"][pending["id"]]
    assert manifest["tasks"][pending["id"]]["disposition"] == "not_started"
    assert manifest["tasks"][pending["id"]]["attempt_path"] is None
    assert sha256(migrated["tasks"][pending["id"]]["task_path"]) == pending_ref["sha256"]
    assert draft["status"] == "not_submitted" and draft["deadline_unix"] == 9000
    assert before == {p: sha256(p) for p in before}
    with pytest.raises(FileExistsError):
        relay.prepare(new_path, old_path, review_ref["path"])
