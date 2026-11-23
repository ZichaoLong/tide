"""Freeze an A-only continuation against the published and qualified instance base."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from . import audit3, prepare3
from .adapter_recovery3 import VERIFIER_CONTINUATION
from .cutover_review3 import CONTROLLER_CONTINUATIONS, HBM_CONTINUATION, OWNER_EXIT_CONTINUATION, STALE_GROUP_REASON
from .prepare_campaign3 import CORE_FILES, budget_remaining, source_inventory, storage_observation
from .prepare_relay3 import write_campaign_draft
from .prepare_resume3 import reference
from .report3 import build_report
from .resume_adapter3 import initial_origin
from .scheduler3 import evaluate_outcome, evaluation_bound, freeze_json, read, worker_limit
from .source_migration3 import ACTIVITY_CONTINUATION, POOL_ACCEPTANCE, RECEIPT, digest, migration_inputs, prior_evaluation_tokens, read_bound, revised_model
from .stage_a_continued3 import verify_scope


def prior_tasks(root, state, review, refs, identity, destination):
    report = build_report(root)
    verified = {row["id"] for row in report["tasks"] if row["execution_complete"]}
    supplement = {row["task_id"]:row for row in review["shared_group_review"]["outcomes"]}
    verified.update(tid for tid,row in supplement.items() if row["outcome"] == "old_worker_completion_verified")
    classified = {row["id"]:row for row in read_bound(refs["same_contract_attempt_classification"])["items"]}
    checkpoints = {row["id"]:row for row in read_bound(refs["checkpoint_decode_observations"])["items"]}
    failure = review["worker_and_checkpoint_review"]["shannon_failure"]
    completed, outstanding, extra = {}, {}, {}
    prior = Counter(state["config"].get("prior_evaluation_reservations", {}))
    for tid, record in state["tasks"].items():
        task_ref = reference(record.get("task_path") or root/"tasks"/tid/"task.json")
        old = read_bound(task_ref)
        if old["id"] != tid:
            raise ValueError("scheduler task id differs from its original request")
        key = record.get("evaluation_budget_key")
        if key:
            prior[key] += record.get("evaluation_reserved_tokens", 0)
        if record["status"] == "completed":
            result_ref = {"path":record["result_path"], "sha256":record["result_sha256"]}
            result = read_bound(result_ref)
            if (tid not in verified or result["task_sha256"] != task_ref["sha256"]
                    or evaluate_outcome(old,result,0)[0] != "completed" or not result["health"]["eligible"]):
                raise ValueError("old completion lacks bound worker evidence: "+tid)
            completed[tid] = {"task":task_ref, "result":result_ref}
            continue
        item = outstanding[tid] = {"task":task_ref}
        cls = classified[tid]
        case = supplement.get(tid, {})
        cp_check = checkpoints.get(tid)
        allowance = cls.get("additional_evaluation_reserved_tokens")
        if tid == failure["task_id"] or cp_check and cp_check["worker_status"] == "budget_exhausted":
            result_ref = reference(Path(record["attempts"][-1]["path"])/"result.json")
            result = read_bound(result_ref)
            cp = result["checkpoint"]
            if (not cp_check["original_contract_checked"] or cp != cp_check["checkpoint"]
                    or reference(cp["path"])["sha256"] != cp["sha256"]):
                raise ValueError("original checkpoint integrity changed: "+tid)
            initial = initial_origin(result_ref,old,task_ref["sha256"])
            if initial is None:
                initial = {"mode":"rerun","exposure":old.get("evaluation_start_tokens",0)}
            mode = "signal"
            if tid == failure["task_id"]:
                mode = "shannon_repair"
                if result_ref["sha256"] != failure["old_result"]["sha256"] or cp != failure["official_checkpoint"]:
                    raise ValueError("Shannon witness checkpoint or result changed")
                # The failed attempt already spent its initial evaluation and
                # causality probe. Its final evaluation was never executed.
                allowance = max(0, result["evaluation"]["tokens"] + 32 + old["eval_tokens"] + 32
                                - record["evaluation_reserved_tokens"])
            elif allowance is None:
                if case.get("outcome") != "signal_result_with_checkpoint_but_missing_owner_receipt" or case["result"]["sha256"] != result_ref["sha256"]:
                    raise ValueError("signal result lacks its supplementary review")
                allowance = case["same_contract_incremental_evaluation_tokens"]
            receipt = {"schema":RECEIPT,"mode":mode,"task_id":tid,"new_source_digest":digest(identity),
                "original_task":task_ref,"result":result_ref,"checkpoint":cp,"initial_evaluation":initial,
                "evaluation_additional_reservation_tokens":allowance}
            if mode == "shannon_repair":
                receipt["qualification"] = refs["foundation_acceptance"]
            receipt["prior_observed_evaluation_tokens"] = prior_evaluation_tokens(receipt)
            migration_inputs({**old,"model":revised_model(old["model"]),"source_identity":identity}, receipt)
            item["receipt"] = reference(freeze_json(destination/"source-migrations"/(tid+".json"),receipt))
            item["disposition"] = mode
        elif cls.get("disposition") in {"not_started","retry"}:
            item["disposition"] = cls["disposition"]
        elif case.get("outcome") == "pending_without_observed_execution_or_result":
            item["disposition"] = "not_started"
            allowance = max(0, evaluation_bound(old)-record.get("evaluation_reserved_tokens",0))
        else:
            raise ValueError("unclassified original task: "+tid)
        if old.get("evaluation_budget_key"):
            if type(allowance) is not int or not 0 <= allowance <= evaluation_bound(old):
                raise ValueError("invalid additional evaluation reservation: "+tid)
            extra[tid] = allowance
    if set(completed) != verified or set(outstanding) != set(classified):
        raise ValueError("source revision does not account for every registered task")
    projected = prior.copy()
    for tid,reserve in extra.items():
        projected[read_bound(outstanding[tid]["task"])["evaluation_budget_key"]] += reserve
    if any(value > 2**20 for value in projected.values()):
        raise ValueError("source revision exceeds a subject's original evaluation budget")
    return completed, outstanding, dict(prior), extra


def prepare(repo, foundation, output, readiness):
    repo, foundation, output = (Path(p).resolve() for p in (repo, foundation, output))
    review_ref = reference(readiness)
    review = read_bound(review_ref)
    refs = {k:{"path":str((repo/v["path"]).resolve()),"sha256":v["sha256"]} for k,v in review["inputs"].items()}
    old, state = read_bound(refs["campaign_plan"]), read_bound(refs["scheduler"])
    old_round = Path(refs["scheduler"]["path"]).parent
    config = copy.deepcopy(old["config"])
    schema = review.get("schema")
    scheduling = schema in CONTROLLER_CONTINUATIONS
    verifier_repair = schema == VERIFIER_CONTINUATION
    activity_repair = schema == ACTIVITY_CONTINUATION
    if scheduling or verifier_repair or activity_repair:
        previous = read_bound(old["stage_a_migration"])
        commit = previous["foundation_commit"]
        if verifier_repair:
            tid = review["priority_task_id"]
            gid = state["tasks"][tid]["shared_group"]
            reasons = {f"Blocked: dynamic_owner_failed:{gid}:worker_process_failed:{tid}:1"}
        elif activity_repair:
            from .round3 import mean_activity, TUNING_SEEDS
            pairs = []
            for tid in review["confirmation_tasks"]:
                row = state["tasks"][tid]
                task_ref = reference(row["task_path"])
                task = read_bound(task_ref)
                from .resume_adapter3 import _result
                result = _result({"path":row["result_path"], "sha256":row["result_sha256"]}, task, task_ref["sha256"])
                if row["status"] != "completed" or task["phase"] != "a3" or task["id"] != tid:
                    raise ValueError("activity repair requires the completed A3 seed group")
                pairs.append((task,result))
            tasks, results = zip(*pairs)
            if ([t["seed"] for t in tasks] != TUNING_SEEDS or len({(t["model_id"],t["recipe_id"]) for t in tasks}) != 1
                    or review["priority_task_id"] not in review["confirmation_tasks"]
                    or mean_activity(results, tasks) != review["activity"] or not review["activity"].get("inherited_samples")):
                raise ValueError("activity repair disagrees with the original seed measurements")
            reasons = {"Blocked: missing_actual_activity_measurement"}
        elif schema == OWNER_EXIT_CONTINUATION:
            reasons = {f"Blocked: {state['tasks'][tid]['stage']}: {STALE_GROUP_REASON}"
                       for tid in review["completed_after_owner_exit"]}
        else:
            reasons = {"user_signal"}
        if (state.get("blocking_reason") not in reasons or refs["foundation_acceptance"] != previous["qualification"]
                or state["tasks"][review["priority_task_id"]]["model_id"] not in {m["id"] for m in old["subjects"]}):
            raise ValueError("controller continuation requires its reviewed stop and original qualification")
    else:
        commit = subprocess.check_output(["git","-C",str(foundation),"rev-parse","HEAD"],text=True).strip()
    if commit != review["foundation"]["published_head"] or state["status"] != "a_blocked_needs_repair" or state["stage_b_gate"] != "closed":
        raise ValueError("foundation release or prior A state changed")
    service = audit3.systemd_observation(review["current_stage_a"]["unit"])
    group = audit3.cgroup_observation(service)
    processes = audit3.process_observation(state,service,group,*audit3.process_snapshot())
    if group["released"] is not True or processes["released"] is not True or service["properties"]["MainPID"] != "0":
        raise ValueError("previous Stage A still owns processes")
    remaining = budget_remaining(config["budget_ledger"],time.time())
    if config["deadline_unix"] != remaining["deadline_unix"] or not set(config["host_cpu_affinity"]) <= os.sched_getaffinity(0):
        raise ValueError("original deadline or CPU allocation unavailable")
    sources = [*(repo/"pilotlab").glob("*.py"), *(foundation/"src/tide").rglob("*.py")]
    pool_floor = read_bound(refs["foundation_acceptance"])["schema"] == POOL_ACCEPTANCE
    write_reserve = config["max_inflight_checkpoint_bytes"]
    if pool_floor and config.get("dynamic_shared_admission"):
        config["use_live_checkpoint_headroom"] = True
        from .scheduler_shared3 import reservations
        write_reserve = max(reservations(read(r.get("task_path") or old_round/"tasks"/tid/"task.json"))["checkpoint_reservation_bytes"]
                            for tid,r in state["tasks"].items())
        if write_reserve > config["max_inflight_checkpoint_bytes"]:
            raise ValueError("a registered checkpoint reservation exceeds the campaign write limit")
    space = storage_observation(output, reserve=config["storage_reserve_bytes"]+write_reserve,
                                snapshot_bytes=sum(p.stat().st_size for p in sources))
    prepare3.freeze(repo,foundation,output,python=config["python"],data=config["data"],budget=config["budget_ledger"],
        max_workers=worker_limit(config),storage_reserve_bytes=config["storage_reserve_bytes"],max_rss_bytes=config["max_rss_bytes"])
    new = read(output/"prepared.json")
    identity = new["source_identity"]
    local, base = source_inventory(new)
    old_local, _ = source_inventory({"source":config.get("worker_source",config["source"]),
        "foundation":config.get("worker_foundation",config["foundation"]),"source_identity":old["source_identity"]})
    if (any(local[k] != old_local[k] for k in CORE_FILES)
            or base != read_bound(refs["foundation_acceptance"])["authority"]["files_sha256"]
            or any(identity[k]!=old["source_identity"][k] for k in ("runtime_packages","data_manifest_sha256","python_version"))):
        raise ValueError("frozen training core, foundation, data or runtime differs from qualification")
    if scheduling:
        identity = old["source_identity"]
    if old.get("stage_a_migration"):
        from .followup_migration3 import prior_tasks as reconcile_followup
        done, running, prior, extra = reconcile_followup(old_round,state,review,refs,identity,output)
        priority = review["priority_task_id"] if scheduling or verifier_repair or activity_repair else review["worker_and_checkpoint_review"]["fault_task_id"]
    else:
        done, running, prior, extra = prior_tasks(old_round,state,review,refs,identity,output)
        priority = review["worker_and_checkpoint_review"]["shannon_failure"]["task_id"]
    batch = {"schema":"pilot-r3-stage-a-migration-plan-v1","readiness":review_ref,"old_plan":refs["campaign_plan"],
        "qualification":refs["foundation_acceptance"],"old_scheduler":refs["scheduler"],"old_selected":refs["selected"],
        "old_resources":reference(old_round/"model-resources.json"), "foundation_commit":commit,
        "new_source_digest":digest(identity), "completed":done,"outstanding":running,
        "exposed_subject_ids":review["stage_a_scope"]["exposed_subject_ids"]}
    batch["priority_model" if scheduling or verifier_repair or activity_repair else "fault_model"] = state["tasks"][priority]["model_id"]
    if old.get("stage_a_migration"):
        batch["continuing_model_ids"] = sorted({state["tasks"][tid]["model_id"] for tid in running})
    batch_ref = reference(freeze_json(output/"stage-a-migration.json",batch))
    for key in ("worker_source","worker_foundation","resume_import"):
        config.pop(key,None)
    config.update(source=new["source"],foundation=new["foundation"],run_id=new["run_id"],
        resume_adapter=reference(Path(new["source"])/"pilotlab/resume_adapter3.py"),
        prior_evaluation_reservations=prior,migration_evaluation_reservations=extra)
    if scheduling:
        config.update(worker_source=old["config"].get("worker_source", old["config"]["source"]),
            worker_foundation=old["config"].get("worker_foundation", old["config"]["foundation"]),
            resume_adapter=old["config"]["resume_adapter"])
        if schema == HBM_CONTINUATION:
            config["exclusive_max_hbm_bytes"] = 52 * 2**30
    plan = {**old,"config":config,"source_identity":identity,"stage_a_migration":batch_ref,
        "subjects":[{**row,"model":revised_model(row["model"], pool_floor=pool_floor)} for row in old["subjects"]],
        "pipeline_model_limit":(old["pipeline_model_limit"] if old.get("stage_a_migration") else
            old["pipeline_model_limit"]+len(set(batch["exposed_subject_ids"])-set(read_bound(refs["selected"])))),
        "storage_observation":space,"budget_remaining_at_freeze":remaining}
    for key in ("worker_prepared","resume_import","controller_source_identity"):
        plan.pop(key,None)
    if scheduling:
        plan["controller_source_identity"] = new["source_identity"]
        if "stage_a_cpu_placement" in review:
            plan["stage_a_cpu_placement"] = review["stage_a_cpu_placement"]
    verify_scope(plan)
    plan_ref = reference(freeze_json(output/"campaign-plan.json",plan))
    freeze_json(output/"campaign-prepared.json",{**new,"campaign_plan":plan_ref,"stage_a_migration":batch_ref,
                **({"worker_source_identity":identity} if scheduling else {}),
                "status":"frozen_controller_continuation_not_submitted" if scheduling else "frozen_source_migration_not_submitted"})
    freeze_json(Path(new["output"])/"selected.json",read_bound(refs["selected"]))
    freeze_json(Path(new["output"])/"model-resources.json",read_bound(batch["old_resources"]))
    summary = dict(Counter(v["disposition"] for v in running.values()), completed=len(done),
                   additional_evaluation_reserved_tokens=sum(extra.values()))
    return write_campaign_draft(output,new,plan,"pilotlab.stage_a_revision3",remaining,
        {"stage_a_migration":batch_ref,"foundation_commit":commit,"summary":summary,
         "resume_policy":("A only; revised controller, original frozen workers, checkpoints, budget and deadline"
                          if scheduling else "A only; published source and explicit checkpoint migration; continue original budget and deadline")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","foundation","output","readiness"):
        parser.add_argument("--"+name,required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)),indent=2))
