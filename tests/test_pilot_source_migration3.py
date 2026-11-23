from copy import deepcopy
import json

import pytest

from pilotlab.prepare_resume3 import reference
from pilotlab.source_migration3 import RECEIPT, digest, migration_inputs, prior_evaluation_tokens, revised_model


def test_planned_storage_change_is_bound_to_the_review_and_original_resource_limits(tmp_path, monkeypatch):
    from pilotlab import prepare_campaign3
    from pilotlab.cutover_review3 import PLANNED_CONTROLLER_CONTINUATION, reviewed_storage_settings
    from pilotlab.stage_a_continued3 import verify_scope

    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return reference(path)

    # Source inventory has its own hash checks. This fixture exercises the
    # reviewed resource comparison and its binding to a particular campaign.
    monkeypatch.setattr(prepare_campaign3, "source_inventory", lambda prepared: None)
    prior = dict(python="python", data="data", budget_ledger="budget", deadline_unix=1000,
        max_workers=None, max_npu_devices=8, target_npu_devices=8, max_cpu_workers=40,
        max_cpu_task_threads=160, max_rss_bytes=10**12, host_cpu_affinity=[0, 1],
        shared_max_hbm_bytes=48*2**30, exclusive_max_hbm_bytes=52*2**30, dynamic_shared_admission=True,
        storage_reserve_bytes=24*2**30, max_inflight_checkpoint_bytes=24*2**30, use_live_checkpoint_headroom=True,
        source="/worker", foundation="/base", resume_adapter={"sha256": "adapter"})
    old = {"subjects": [{"id": "one", "model": {}}], "stage_a_revision": {"policy": "A only"},
           "source_identity": {"source_files": {"/base/src/tide/a.py": "frozen"}}, "config": prior}
    old_ref = save("old.json", old)
    qualification = save("acceptance.json", {"authority": {"files_sha256": {"a.py": "frozen"}}})
    review = {"schema": PLANNED_CONTROLLER_CONTINUATION, "storage_policy": "hard_quota",
              "foundation": {"published_head": "head"}, "inputs": {"campaign_plan": old_ref, "foundation_acceptance": qualification},
              "stage_a_scope": {"prospective_recipe_check": {"canonical_prospective_subjects_sha256": digest(old["subjects"])}}}
    batch = {"schema": "pilot-r3-stage-a-migration-plan-v1", "readiness": save("review.json", review),
             "old_plan": old_ref, "qualification": qualification, "foundation_commit": "head",
             "new_source_digest": digest(old["source_identity"])}
    plan = {**old, "config": {**prior, **reviewed_storage_settings(review), "worker_source": prior["source"],
                              "worker_foundation": prior["foundation"]}, "controller_source_identity": {},
            "stage_a_migration": save("batch.json", batch)}
    assert verify_scope(plan) == old["stage_a_revision"]
    for change in ({"max_npu_devices": 9}, {"deadline_unix": 2000}, {"storage_reserve_bytes": 1},
                   {"max_inflight_checkpoint_bytes": 2**30}, {"use_live_checkpoint_headroom": True}):
        with pytest.raises(ValueError):
            verify_scope({**plan, "config": {**plan["config"], **change}})
    del review["storage_policy"]
    batch["readiness"] = save("original-review.json", review)
    plan["stage_a_migration"] = save("original-batch.json", batch)
    with pytest.raises(ValueError):
        verify_scope(plan)
    assert verify_scope({**plan, "config": {**plan["config"], **prior}}) == old["stage_a_revision"]


def test_cumulative_evaluation_counts_every_original_attempt_once():
    results = {"first":{"id":"t","evaluation":{"tokens":16}},
               "second":{"id":"t","evaluation":{"tokens":4},"execution_adapter":{"original_result":{"path":"first"}}}}
    receipt = {"mode":"signal","task_id":"t","result":{"path":"second"}}
    assert prior_evaluation_tokens(receipt, lambda ref: results[ref["path"]]) == 20
    assert prior_evaluation_tokens({**receipt,"mode":"stage_parent"}) == 0


def test_migrated_parent_requires_exact_rung_and_qualified_recipe(tmp_path):
    def save(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return reference(path)

    identity = {"source_files":{"old":"1"}, "data_manifest_sha256":"d", "python_version":"3", "runtime_packages":{}}
    old = {"id":"t-a1-1-r0","phase":"a1","model_id":"t","seed":1,"lr":.001,
           "model":{"family":"tide","config":{"allocation":"shannon"}},"source_identity":identity,
           "target_tokens":256,"evaluation_start_tokens":0}
    old_ref = save("old.json",old)
    cp = {"path":"original.pt","sha256":"a","steps":2,"tokens":256,"state_sha256":"b"}
    result = {**{k:old[k] for k in ("id","model_id","phase","source_identity")},
              "schema":"pilot-r3-worker-result-v1","task_sha256":old_ref["sha256"],
              "status":"completed","health":{"eligible":True},"checkpoint":cp}
    task = {**old,"id":"t-a2-1-r0","phase":"a2","target_tokens":1024,"evaluation_start_tokens":256,
            "resume_checkpoint":cp["path"],"resume_checkpoint_sha256":cp["sha256"],
            "source_identity":{**identity,"source_files":{"new":"2"}},"model":revised_model(old["model"])}
    receipt = {"schema":RECEIPT,"mode":"stage_parent","task_id":task["id"],"new_source_digest":digest(task["source_identity"]),
               "original_task":old_ref,"result":save("result.json",result),"checkpoint":cp,
               "initial_evaluation":{"mode":"rerun","exposure":256}}
    assert migration_inputs(task,receipt) == (old,result)
    for change in ({"lr":.002},{"seed":3},{"model":old["model"]},{"resume_checkpoint_sha256":"other"},
                   {"phase":"a3"},{"evaluation_start_tokens":512}):
        with pytest.raises(ValueError):
            migration_inputs({**task,**change},receipt)
    failed = {**receipt,"mode":"signal"}
    with pytest.raises(ValueError):
        migration_inputs(task,failed)
    other = deepcopy(receipt)
    other["checkpoint"]["tokens"] += 1
    with pytest.raises(ValueError):
        migration_inputs(task,other)


def test_initial_selection_survives_second_source_migration(tmp_path):
    from pilotlab.resume_adapter3 import initial_origin

    def save(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return reference(path)

    old = {"id":"model-a1","phase":"a1","model_id":"model","source_identity":{"revision":1},
           "eval_tokens":8,"evaluation_start_tokens":0}
    old_ref = save("old-task.json",old)
    observation = {"status":"completed","split":"selection","tokens":8,"requested_tokens":8,"nll":2.5}
    old_result = {"schema":"pilot-r3-worker-result-v1","id":old["id"],"phase":"a1","model_id":"model",
                  "source_identity":old["source_identity"],"task_sha256":old_ref["sha256"],
                  "validation_initial":observation,"training":{"from_tokens":0}}
    first = save("old-result.json",old_result)
    origin = initial_origin(first,old,old_ref["sha256"])
    receipt = save("migration.json",{"mode":"signal","original_task":old_ref,"result":first,
                                    "initial_evaluation":origin})
    new = {**old,"stage_a_migration":receipt,"source_identity":{"revision":2}}
    new_ref = save("new-task.json",new)
    inherited = {**observation,"tokens":0,
                 "inherited_from":{**first,"field":"validation_initial","task_id":old["id"],"exposure":0}}
    result = {**old_result,"source_identity":new["source_identity"],"task_sha256":new_ref["sha256"],
              "training":{"from_tokens":32},"validation_initial":inherited,
              "execution_adapter":{"schema":"pilot-r3-source-migration-adapter-v1","new_task":new_ref,
                                   "original_task":old_ref,"original_result":first,"receipt":receipt}}
    assert initial_origin(save("new-result.json",result),new,new_ref["sha256"]) == origin
    result["validation_initial"]["nll"] += .5
    with pytest.raises(ValueError,match="original initial observation"):
        initial_origin(save("changed-result.json",result),new,new_ref["sha256"])


def test_unclosed_segment_preserves_original_evaluation_count():
    earlier = {"mode":"signal","result":{"path":"initial"}}
    interrupted = {"mode":"checkpoint_only","task_id":"t","result":{"path":"placeholder"},
                   "original_task":{"path":"interrupted-task"}}
    results = {"initial":{"id":"t","evaluation":{"tokens":16}},
               "interrupted-task":{"stage_a_migration":{"path":"earlier"}},
               "earlier":earlier,"checkpoint-only":interrupted,
               "recovered":{"id":"t","evaluation":{"tokens":4},
                            "execution_adapter":{"migration_mode":"checkpoint_only","receipt":{"path":"checkpoint-only"}}}}
    read = lambda ref: results[ref["path"]]
    assert prior_evaluation_tokens(interrupted,read) == 16
    assert prior_evaluation_tokens({"mode":"signal","task_id":"t","result":{"path":"recovered"}},read) == 20


@pytest.mark.parametrize("damage", [None, "job_lock", "device_lock", "live_process", "undercharge", "other_audit_finding"])
def test_planned_stop_with_missing_owner_receipt_needs_independent_release(tmp_path, damage):
    from pilotlab.cutover_review3 import reconcile_stop
    entry = {"allocation_unix":10., "cleanup_verified_unix":21., "device_seconds":11.,
             "accounting_estimated":True, "owner_return_code":-9}
    if damage == "undercharge":
        entry["device_seconds"] = 10.
    path = tmp_path/"scheduler.json"
    path.write_text(json.dumps({"blocking_reason":"user_signal", "remaining_descendants":[],
                                "shared_groups":{"g":entry}, "tasks":{}}))
    ref = reference(path)
    queue = {"job_id":"job", "shared_group":"g", "recorded_status":"running",
             "job_lock":{"status":"held" if damage == "job_lock" else "unlocked"},
             "device_locks":[{"status":"held" if damage == "device_lock" else "unlocked"}]}
    group = {"status":"running", "job_id":"job", "physical_id":4, "allocation_acquired_unix":10.,
             "allocation_released_unix":None}
    audit = {"execution":{"scheduler":ref}, "shared_groups":{"groups":{"g":{"group":group,
                "binding_verified":True, "owner_alive":False}}},
             "lifecycle":{"status":"terminal", "processes":{"released":damage != "live_process"},
                "cgroup":{"released":True}, "queue":{"attempts":[queue]}},
             "findings":[{"group_id":"g", "reason":"shared_group_not_terminal"}]}
    if damage == "other_audit_finding":
        audit["findings"].append({"kind":"checkpoint", "reason":"checkpoint_digest_changed"})
    if damage:
        with pytest.raises(ValueError):
            reconcile_stop(audit, ref)
    else:
        assert reconcile_stop(audit, ref)["stale_owner_receipts"]["g"]["device_seconds"] == 11.


@pytest.mark.parametrize("finding", [
    {"task_id": "interrupted", "reason": "shared_worker_result_binding_mismatch", "kind": "result"},
    {"task_id": "different", "reason": "checkpoint_digest_changed", "kind": "checkpoint"},
])
def test_hard_stop_rejects_its_own_or_unrelated_unresolved_artifact(tmp_path, finding):
    from pilotlab.source_migration3 import verify_hard_stop
    path = tmp_path / "scheduler.json"
    path.write_text(json.dumps({"tasks": {}}))
    audit = {"status": "incomplete", "findings": [finding], "execution": {"scheduler": reference(path)},
             "lifecycle": {k: {"released": True} for k in ("processes", "cgroup", "queue")}}
    with pytest.raises(ValueError):
        verify_hard_stop({"id": "interrupted"}, {}, {}, {}, audit, {})


@pytest.mark.parametrize("tokens", [256, 384])
def test_partial_stage_start_evaluation_can_only_be_repeated_at_same_exposure(tmp_path, tokens):
    from pilotlab.followup_migration3 import resume_interruption
    task = {"id":"trial", "model_id":"model", "phase":"a2", "source_identity":{"revision":1},
            "evaluation_start_tokens":256, "eval_tokens":64}
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    task_ref = reference(path)
    result = {**{k:task[k] for k in ("id", "model_id", "phase", "source_identity")},
              "schema":"pilot-r3-worker-result-v1", "task_sha256":task_ref["sha256"], "status":"budget_exhausted",
              "stop_reason":"signal", "causality":{"within_tolerance":True}, "health":{"blocking_reasons":[]},
              "training":{"tokens":tokens, "from_tokens":256}, "checkpoint":{"tokens":tokens},
              "validation_initial":{"status":"budget_exhausted", "tokens":32, "split":"selection"}}
    (tmp_path/"result.json").write_text(json.dumps(result))
    args = (task_ref, tmp_path, {"reason":"signal"}, {}, {}, "fault")
    if tokens != 256:
        with pytest.raises(ValueError, match="initial selection missing"):
            resume_interruption(*args)
    else:
        _, receipt, _ = resume_interruption(*args)
        assert receipt["initial_evaluation"] == {"mode":"rerun", "exposure":256}


def test_pool_policy_is_an_explicit_directional_recipe_migration():
    from pilotlab.source_migration3 import POLICY, verify_recipe
    spec = {"family":"tide","config":{"allocation":"pool"}}
    ident = {"runtime_packages":{}, "data_manifest_sha256":"x", "python_version":"3"}
    old = {"model":spec, "source_identity":ident}
    current = {**old, "model":revised_model(spec, pool_floor=True)}
    assert revised_model(spec) == spec and current["model"]["config"]["pool_support_policy"] == POLICY
    verify_recipe(old, old)
    verify_recipe(old, current)
    with pytest.raises(ValueError):
        verify_recipe(current, old)


def test_waiter_only_reenters_queue_with_positive_proof_of_no_allocation_or_work(tmp_path):
    from pilotlab.cutover_review3 import unallocated_worker
    attempt = tmp_path/"attempt"
    attempt.mkdir()
    task = tmp_path/"task.json"
    task.write_text("{}")
    queue_file = tmp_path/"queue.json"
    queue_file.write_text(json.dumps({"job_id":"job","physical_ids":[],"status":"failed","return_code":1}))
    queue = {"released":True, "queue_state":reference(queue_file)}
    record = {"attempts":[{"path":str(attempt)}],"shared_group":"g","task_path":str(task)}
    group = {"job_id":"job","status":"failed","reason":"signal_15","remaining_descendants":[],
             "queue_released_unix":100,"allocation_acquired_unix":None,"physical_id":None,
             "physical_device_seconds":0,"workers":{"t":{"status":"not_started","output":str(attempt)}}}
    owner = {"binding_verified":True,"owner_alive":False,"group":group}
    assert unallocated_worker("t", record, owner, queue)["task"] == reference(task)
    group["allocation_acquired_unix"] = 1
    with pytest.raises(ValueError):
        unallocated_worker("t", record, owner, queue)
    group["allocation_acquired_unix"] = None
    (attempt/"result.json").write_text("{}")
    with pytest.raises(ValueError):
        unallocated_worker("t", record, owner, queue)


@pytest.mark.parametrize("damage", [None,"strict","gap","two_extra","tokens","clock","checkpoint","nonfinite"])
def test_one_durable_step_after_heartbeat_is_only_admissible_for_a_rung_restart(damage):
    from pilotlab.source_migration3 import verify_stop_journal
    steps=[dict(step=i,tokens=256*i,time_unix=float(i),loss=8.,objective=8.,grad_norm=1.,lr=.001)
           for i in (1,2,3)]
    hb=dict(steps=2,tokens=512,status="training",updated_unix=2.1)
    cp=dict(steps=1,tokens=256)
    task=dict(batch_size=2,sequence_length=128,target_tokens=262144)
    if damage=="gap": steps[0]["step"]=0
    if damage=="two_extra": steps.append({**steps[-1],"step":4,"tokens":1024})
    if damage=="tokens": steps[-1]["tokens"]-=1
    if damage=="clock": steps[-1]["time_unix"]=1.9
    if damage=="checkpoint": cp=dict(steps=3,tokens=768)
    if damage=="nonfinite": steps[-1]["loss"]=float("nan")
    def verify():
        return verify_stop_journal(steps,hb,cp,task,{"finished_unix":4.},allow_restart_tail=damage!="strict")
    if damage:
        with pytest.raises(ValueError,match="finite training journal"):
            verify()
    else:
        assert verify()=={"step":3,"tokens":768,"time_unix":3.}
        assert verify_stop_journal(steps[:2],hb,cp,task,{"finished_unix":4.}) is None


def test_checkpoint_only_counts_stage_start_replay_once_across_repeated_resume():
    replay = {"path":"replay"}
    receipt = {"task_id":"a2","mode":"checkpoint_only","initial_replay":replay}
    files = {"replay":{"id":"a2","evaluation":{"tokens":128},"execution_adapter":{"migration_mode":"stage_parent"}},
             "receipt":receipt,
             "next":{"id":"a2","evaluation":{"tokens":64},
                     "execution_adapter":{"migration_mode":"checkpoint_only","receipt":{"path":"receipt"}}}}
    def load(ref):
        return files[ref["path"]]
    assert prior_evaluation_tokens(receipt, load)==128
    assert prior_evaluation_tokens({"task_id":"a2","mode":"signal","result":{"path":"next"}}, load)==192
