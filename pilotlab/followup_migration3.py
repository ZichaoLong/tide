"""Reconcile the immediate predecessor and its inherited results before a further source migration."""
from collections import Counter, defaultdict
from pathlib import Path

from .adapter_recovery3 import VERIFIER_CONTINUATION, recover_completion
from .continuation3 import continuation_origin
from .cutover_review3 import (bound_terminal_audit, CONTROLLER_CONTINUATIONS, HBM_CONTINUATION,
                             OWNER_EXIT_CONTINUATION, PLANNED_CONTROLLER_CONTINUATION, RESOURCE_MEMORY_CONTINUATION, STALE_GROUP_REASON)
from .prepare_relay3 import verify_resume_payloads
from .prepare_resume3 import reference
from .report3 import build_report
from .resume_adapter3 import _result, initial_origin
from .scheduler3 import evaluate_outcome, evaluation_bound, freeze_json
from .source_migration3 import ACTIVITY_CONTINUATION, POOL_ACCEPTANCE, RECEIPT, REPAIR_SCHEMAS, digest, migration_inputs, prior_evaluation_tokens, read_bound, revised_model, verify_hard_stop


def remaining_evaluation(task, receipt):
    """One probe, any outstanding periodic selection and the final selection."""
    if not receipt:
        return evaluation_bound(task)
    tokens = receipt["checkpoint"]["tokens"]
    interval = task.get("eval_interval_tokens", 1048576)
    middle = max(0, (task["target_tokens"]-1)//interval - tokens//interval) if interval > 0 else 0
    return 32 + task["eval_tokens"] * (1 + middle + (receipt["initial_evaluation"]["mode"] == "rerun"))


def resume_interruption(task_ref, attempt, record, refs, audit_findings, fault, *, verified_signal=None,
                        recover_prefix=False, initial_replay=None):
    task = read_bound(task_ref)
    tid = task["id"]
    result_ref = reference(attempt/"result.json")
    result = _result(result_ref, task, task_ref["sha256"])
    reason = record.get("reason")
    initial = initial_origin(result_ref, task, task_ref["sha256"])
    cp = result.get("checkpoint")
    extra_fields = {}
    if tid == fault:
        schema = read_bound(refs["foundation_acceptance"])["schema"]
        mode = {value:key for key,value in REPAIR_SCHEMAS.items()}[schema]
        extra_fields["qualification"] = refs["foundation_acceptance"]
    elif reason in ("signal", "shared_group_not_completed:signal_15") or verified_signal == result_ref:
        mode = "signal"
        permitted = {"declared_training_exposure_not_reached", "evaluation_incomplete",
                     "checkpoint_load_not_verified", "causality_not_verified"}
        if (result.get("status") != "budget_exhausted" or result.get("stop_reason") != "signal"
                or result.get("failure_class") or result.get("causality",{}).get("within_tolerance") is not True
                or set(result["health"]["blocking_reasons"]) - permitted):
            raise ValueError("worker interruption has additional failures: "+tid)
    elif result["status"] == "running" and (reason == "worker_exit_-9" or
            tid in audit_findings and audit_findings[tid]["sha256"] == result_ref["sha256"]):
        # The worker's initial result.json is a placeholder. Its independently
        # generated checkpoint index and step journal may still be intact.
        index = reference(attempt/"checkpoint.json")
        cp = read_bound(index)["current"]
        tail = None
        if tid not in audit_findings:
            hard_stop = {"heartbeat": reference(attempt/"heartbeat.json"), "step_journal": reference(attempt/"steps.jsonl")}
            if recover_prefix:
                hard_stop["resume_prefix"] = True
            tail = verify_hard_stop(task, task_ref, result_ref, cp, bound_terminal_audit(refs["terminal_audit"]),
                                    {**hard_stop, "checkpoint_index": index}, allow_restart_tail=True)
            if tail and recover_prefix:
                hard_stop["journal_step_after_last_heartbeat"] = tail
            extra_fields["hard_stop"] = hard_stop
        if tail and not recover_prefix:
            return "restart_stage", None, {"placeholder":result_ref, "retained_checkpoint":cp, **extra_fields,
                "journal_step_after_last_heartbeat":tail,
                "reason":"hard stop between step and heartbeat publications; restart at the frozen rung origin"}
        previous = read_bound(task["stage_a_migration"]) if task.get("stage_a_migration") else None
        if initial_replay:
            if not recover_prefix or not previous or previous["mode"] != "stage_parent":
                raise ValueError("unexpected stage-start replay for "+tid)
            initial = initial_origin(initial_replay, task, task_ref["sha256"])
            extra_fields["initial_replay"] = initial_replay
        elif previous and previous["mode"] == "signal" and previous["initial_evaluation"]["mode"] == "inherit":
            initial = previous["initial_evaluation"]
        else:
            return "restart_stage", None, {"placeholder":result_ref, "retained_checkpoint":cp, **extra_fields,
                                          "reason":"original stage-start selection has no bound final worker receipt"}
        mode = "checkpoint_only"
        extra_fields.update(terminal_audit=refs["terminal_audit"], checkpoint_index=index)
        heartbeat = record["heartbeat"]
        if not cp["tokens"] <= heartbeat["tokens"] < task["target_tokens"]:
            raise ValueError("unclosed segment crosses an evaluation boundary: "+tid)
    else:
        raise ValueError("unexpected failed predecessor: "+tid)
    if mode == "signal" and initial is None and cp and cp["tokens"] == task.get("evaluation_start_tokens", 0):
        # The interrupted initial evaluation can be repeated at the identical
        # training exposure. Its partial tokens still count toward the budget.
        initial = {"mode": "rerun", "exposure": cp["tokens"]}
    if not cp or not initial:
        raise ValueError("checkpoint or measured initial selection missing: "+tid)
    return mode, {"schema":RECEIPT, "mode":mode, "task_id":tid, "original_task":task_ref,
                  "result":result_ref, "checkpoint":cp, "initial_evaluation":initial, **extra_fields}, {}


def prior_tasks(root, state, review, refs, identity, destination):
    old = read_bound(refs["campaign_plan"])
    inherited = read_bound(old["stage_a_migration"])
    audit = bound_terminal_audit(refs["terminal_audit"])
    findings = {f["task_id"]:f for f in audit["findings"] if f.get("reason")=="shared_worker_result_binding_mismatch"}
    schema = review.get("schema")
    scheduling = schema in CONTROLLER_CONTINUATIONS
    reconciled = audit.get("reconciled_stop", {})
    pool_floor = read_bound(refs["foundation_acceptance"])["schema"] == POOL_ACCEPTANCE
    if schema in {HBM_CONTINUATION, PLANNED_CONTROLLER_CONTINUATION, RESOURCE_MEMORY_CONTINUATION}:
        from .cutover_review3 import reconcile_stop
        reconciled = reconcile_stop(audit, refs["scheduler"], resource_memory=schema == RESOURCE_MEMORY_CONTINUATION)
        if reconciled != review["stop_reconciliation"]:
            raise ValueError("planned cutover review differs from the underlying terminal records")
    elif (audit["execution"]["scheduler"]["sha256"] != refs["scheduler"]["sha256"]
            or len(findings)!=len(audit["findings"])
            or not all(audit["lifecycle"][k]["released"] is True for k in ("cgroup","processes","queue"))):
        raise ValueError("previous ownership or terminal evidence is unresolved")
    if not scheduling and review.get("unpublished_signal_results"):
        from .cutover_review3 import unpublished_signal
        signals = {tid: unpublished_signal(audit, state, findings[tid])
                   for tid in review["unpublished_signal_results"]}
        if signals != review["unpublished_signal_results"]:
            raise ValueError("worker signal results differ from terminal evidence")
        reconciled["signal_results"] = signals
    inspected = build_report(root)
    verified = {row["id"] for row in inspected["tasks"] if row["execution_complete"] and row["evidence_integrity_verified"]}
    terminated = {}
    if review.get("completed_before_signal"):
        from .cutover_review3 import completed_before_signal
        terminated = completed_before_signal(audit, refs["scheduler"])
        if terminated != review["completed_before_signal"]:
            raise ValueError("completed worker shutdown evidence changed")
    if schema == OWNER_EXIT_CONTINUATION:
        from .cutover_review3 import completed_on_exit
        recovered = completed_on_exit(audit, refs["scheduler"], reason=STALE_GROUP_REASON)
        if not recovered or recovered != review["completed_after_owner_exit"]:
            raise ValueError("stale owner observation lacks its verified terminal result")
        terminated.update(recovered)
    verified.update(reconciled.get("completed_worker_receipts", {}))
    recorded_complete = {tid for tid,row in state["tasks"].items() if row["status"]=="completed"}
    if verified != recorded_complete:
        raise ValueError("previous completed-task inventory changed")
    verified.update(terminated)
    recovered = {}
    if schema == VERIFIER_CONTINUATION:
        tid = review["priority_task_id"]
        result_ref = review["recovered_result"]
        if (tid in verified or read_bound(result_ref) != recover_completion(refs["terminal_audit"], refs["scheduler"], tid)):
            raise ValueError("post-run completion differs from the audited worker and repaired verifier")
        recovered[tid] = result_ref
    if set(inherited["completed"]) & set(state["tasks"]) or not set(inherited["outstanding"]) <= set(state["tasks"]):
        raise ValueError("previous run double-counted or omitted an inherited task")
    done = dict(inherited["completed"])
    outstanding, extra, decodes = {}, {}, defaultdict(list)
    prior = Counter(state["config"].get("prior_evaluation_reservations",{}))
    activity_repair = schema == ACTIVITY_CONTINUATION
    fault = None if scheduling or recovered or activity_repair else review["worker_and_checkpoint_review"]["fault_task_id"]
    for tid, record in state["tasks"].items():
        task_ref = reference(record.get("task_path") or root/"tasks"/tid/"task.json")
        task = read_bound(task_ref)
        if task["id"]!=tid:
            raise ValueError("scheduler/task identity mismatch")
        key = task.get("evaluation_budget_key")
        charged = record.get("evaluation_reserved_tokens",0)
        if key:
            prior[key] += charged
        if tid in recovered:
            done[tid] = {"task": task_ref, "result": recovered[tid]}
            continue
        if tid in verified:
            result_ref = {"path":record["result_path"],"sha256":record["result_sha256"]}
            result = _result(result_ref,task,task_ref["sha256"])
            if evaluate_outcome(task,result,0)[0]!="completed" or not result["health"]["eligible"]:
                raise ValueError("uncertified worker result: "+tid)
            done[tid] = {"task":task_ref, "result":result_ref}
            continue
        prior_receipt = read_bound(task["stage_a_migration"]) if task.get("stage_a_migration") else None
        if (record["status"]=="pending" or record.get("reason")=="stopped_before_worker_start"
                or tid in reconciled.get("unstarted_before_allocation", {})):
            if record.get("heartbeat") or any((Path(a["path"])/"result.json").exists() for a in record.get("attempts",[])):
                raise ValueError("purported unstarted worker has training evidence: "+tid)
            receipt = dict(prior_receipt) if prior_receipt else None
            mode = "carry_checkpoint" if receipt else "not_started"
            annotation = {}
            original_extra = state["config"]["migration_evaluation_reservations"].get(tid,evaluation_bound(task))
            increment = max(0, original_extra-charged)
        elif tid in reconciled.get("initial_check_replays", {}):
            # Retain the entire prior reservation, including incomplete checks,
            # and reserve the replay against the same subject's remaining budget.
            receipt = dict(prior_receipt) if prior_receipt else None
            mode = "carry_checkpoint" if receipt else "retry_initial_checks"
            annotation = {"initial_check_replay": reconciled["initial_check_replays"][tid]}
            increment = remaining_evaluation(task, receipt)
        else:
            attempt = Path(record["attempts"][-1]["path"])
            replay = review.get("stage_start_replays", {}).get(tid) if activity_repair else None
            mode,receipt,annotation = resume_interruption(task_ref,attempt,record,refs,findings,fault,
                verified_signal=reconciled.get("signal_results", {}).get(tid),
                recover_prefix=activity_repair, initial_replay=replay)
            if replay:
                if not key or mode != "checkpoint_only" or receipt.get("initial_replay") != replay:
                    raise ValueError("stage-start replay is not bound to its interrupted task")
                prior[key] += read_bound(replay)["evaluation"]["tokens"] + 32
            # Previously reserved work covers the last attempt's original bound.
            # Only extra evaluations/probes are charged. For missing terminal
            # counts, retain the prior reservation and reserve the replacement.
            if mode in ("restart_stage","checkpoint_only"):
                increment = remaining_evaluation(task,receipt)
            else:
                observed = read_bound(receipt["result"])["evaluation"]["tokens"]
                increment = max(0,32 + observed + remaining_evaluation(task,receipt) - remaining_evaluation(task,prior_receipt))
        item = outstanding[tid] = {"task":task_ref,"disposition":mode,**annotation}
        if key:
            if type(increment) is not int or not 0 <= increment <= evaluation_bound(task):
                raise ValueError("invalid incremental evaluation allowance: "+tid)
            extra[tid] = increment
        if receipt:
            receipt.update(new_source_digest=digest(identity), evaluation_additional_reservation_tokens=increment)
            receipt["prior_observed_evaluation_tokens"] = prior_evaluation_tokens(receipt)
            origin_task, previous = migration_inputs(
                {**task,"model":revised_model(task["model"], pool_floor=pool_floor),"source_identity":identity},receipt)
            if receipt["mode"] == "signal":
                continuation_origin(origin_task, previous)
            item["receipt"] = reference(freeze_json(destination/"source-migrations"/(tid+".json"),receipt))
            original_ref, cp = receipt["original_task"], receipt["checkpoint"]
        elif mode=="restart_stage":
            original_ref, cp = task_ref, item["retained_checkpoint"]
        else:
            continue
        original = read_bound(original_ref)
        files = [Path(path) for path in original["source_identity"]["source_files"]]
        source = next(p.parents[1] for p in files if p.name=="worker3.py")
        base = next(p.parents[2] for p in files if p.name=="kv.py" and p.parent.name=="tide")
        decodes[(str(source),str(base))].append({"disposition":"resume","task":original_ref,"checkpoint":cp})
    for (source,base),entries in decodes.items():
        verify_resume_payloads({**state["config"],"worker_source":source,"worker_foundation":base},entries)
    projected = prior.copy()
    for tid,reserve in extra.items():
        task = read_bound(outstanding[tid]["task"])
        projected[task["evaluation_budget_key"]] += reserve
        if projected[task["evaluation_budget_key"]] > task["evaluation_token_limit"]:
            raise ValueError("projected stage-A evaluation exposure exceeds the subject limit: "+tid)
    return done,outstanding,dict(prior),extra
