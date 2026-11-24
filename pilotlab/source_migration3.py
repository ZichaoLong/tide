"""Provenance of stage-A states crossing qualified foundation revisions."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .resume_adapter3 import checked, initial_origin, read_json, _result


RECEIPT = "pilot-r3-stage-a-source-migration-v1"
ADAPTER = "pilot-r3-source-migration-adapter-v1"
POLICY = "positive-floor-v1"
POOL_ACCEPTANCE = "pilot-r3-pool-floor-acceptance-v1"
REGIONAL_PRIOR_ACCEPTANCE = "pilot-r3-regional-key-prior-acceptance-v1"
QUAD_TOTAL_ACCEPTANCE = "pilot-r3-quad-total-acceptance-v1"
ACTIVITY_CONTINUATION = "pilot-r3-activity-evidence-continuation-v1"
REPAIR_SCHEMAS = {
    "quad_total_repair": QUAD_TOTAL_ACCEPTANCE,
    "regional_key_prior_repair": REGIONAL_PRIOR_ACCEPTANCE,
    "pool_floor_repair": POOL_ACCEPTANCE,
    "kv_no_write_repair": "pilot-r3-kv-no-write-acceptance-v1",
    "quad_reception_repair": "pilot-r3-quad-reception-acceptance-v1",
    "timed_ingress_repair": "pilot-r3-timed-ingress-acceptance-v1",
    "timed_route_repair": "pilot-r3-timed-route-acceptance-v1",
}
RESOURCE_FIELDS = {"reserved_hbm_bytes", "reserved_rss_bytes", "checkpoint_reservation_bytes"}
SEGMENT_FIELDS = {"phase", "id", "target_tokens", "evaluation_start_tokens",
                  "resume_checkpoint", "resume_checkpoint_sha256"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def revised_model(spec, *, pool_floor=False):
    value = copy.deepcopy(spec)
    if (value.get("family") == "tide" and value["config"]["allocation"] == "shannon"
            and value["config"].get("shannon_support_policy", "strict") == "strict"):
        value["config"]["shannon_support_policy"] = POLICY
    if pool_floor and value.get("family") == "tide" and value["config"]["allocation"] == "pool":
        value["config"]["pool_support_policy"] = POLICY
    return value


def read_bound(ref):
    return read_json(checked(ref, "migration input"))


def _activity_sources(task):
    files = task["source_identity"]["source_files"]
    required = {"worker3.py", "models3.py", "models.py"}
    observed = {}
    for path, hashed in files.items():
        source = Path(path)
        if source.parent.name == "pilotlab" and source.name in required:
            if source.name in observed:
                raise ValueError("duplicate activity measurement source")
            checked({"path":path, "sha256":hashed}, "activity measurement source")
            observed[source.name] = hashed
    if observed.keys() != required:
        raise ValueError("missing activity measurement source")
    return observed


def inherited_activity(task, result):
    """Locate the final measured training sample across exact, zero-update A3 resumes."""
    from .continuation3 import continuation_origin

    if task.get("phase") != "a3" or task.get("id") != result.get("id") or result.get("work") != []:
        raise ValueError("missing final measured activity")
    continuation_origin(task, result)
    sources = _activity_sources(task)
    links = []
    while not result.get("work"):
        receipt_ref = task.get("stage_a_migration")
        mark = result.get("execution_adapter") or {}
        if (len(links) >= 64 or not receipt_ref or receipt_ref in links
                or mark.get("schema") != ADAPTER or mark.get("receipt") != receipt_ref):
            raise ValueError("missing bound zero-step A3 migration")
        receipt = read_bound(receipt_ref)
        cp, end = receipt["checkpoint"], result.get("checkpoint") or {}
        training = result["training"]
        if (receipt["mode"] != "signal" or cp["tokens"] != task["target_tokens"]
                or any(training.get("segment_"+k) != 0 or training.get(k) != cp[k]
                       or training.get("from_"+k) != cp[k] or end.get(k) != cp[k] for k in ("steps","tokens"))
                or end.get("state_sha256") != cp["state_sha256"] or not cp["state_sha256"]):
            raise ValueError("activity inheritance requires the same final state without training")
        checked(end, "zero-step final checkpoint")
        old = read_bound(receipt["original_task"])
        previous = _result(receipt["result"], old, receipt["original_task"]["sha256"])
        if old["model"] != task["model"] or previous.get("parameters") != result.get("parameters"):
            raise ValueError("activity parameter identity changed")
        if _activity_sources(old) != sources or not isinstance(previous.get("work"), list):
            raise ValueError("activity measurement source or record changed")
        links.append(receipt_ref)
        task, result = old, previous
    row = result["work"][-1]
    activity = row["work"]["activity"]
    if (row["step"] != cp["steps"] or row["tokens"] != cp["tokens"]
            or result["training"]["segment_steps"] <= 0 or activity.get("qualified") is not True
            or activity.get("version") != "unique-external-token-parameters-v1"
            or activity.get("whole_model_parameters") != result["parameters"]):
        raise ValueError("predecessor lacks the prescribed qualified final activity sample")
    origin = {"seed":task["seed"], "task_id":task["id"], "task":receipt["original_task"],
              "result":receipt["result"], "checkpoint":cp, "step":row["step"], "tokens":row["tokens"],
              "measurement_version":activity["version"], "measured_external_tokens":activity["measured_external_tokens"],
              "source_digest":digest(task["source_identity"]), "migrations":links}
    return {**activity, "measurement_origin":origin}


def prior_evaluation_tokens(receipt, read_reference=read_bound):
    """Count actual evaluations in the original task's bound attempt chain."""
    if receipt["mode"] == "stage_parent":
        return 0
    def predecessor(link):
        if link.get("mode") == "checkpoint_only":
            if link.get("initial_replay"):
                return link["initial_replay"]
            task = read_reference(link["original_task"])
            return read_reference(task["stage_a_migration"])["result"]
        return link["result"]

    ref, seen, total = predecessor(receipt), set(), 0
    while ref:
        if ref["path"] in seen or len(seen) >= 64:
            raise ValueError("cyclic or excessively long migration evaluation history")
        seen.add(ref["path"])
        result = read_reference(ref)
        count = result.get("evaluation", {}).get("tokens")
        if result.get("id") != receipt["task_id"] or type(count) is not int or count < 0:
            raise ValueError("invalid prior migration evaluation observation")
        total += count
        adapter = result.get("execution_adapter")
        if not adapter or adapter.get("migration_mode") == "stage_parent":
            break
        ref = (predecessor(read_reference(adapter["receipt"])) if adapter.get("migration_mode") == "checkpoint_only"
               else adapter["original_result"])
    return total


def verify_recipe(old, task, *, parent=False):
    ignored = RESOURCE_FIELDS | {"model", "source_identity", "stage_a_migration"}
    if parent:
        ignored |= SEGMENT_FIELDS
    pool_floor = task["model"].get("config", {}).get("pool_support_policy") == POLICY
    if (revised_model(old["model"], pool_floor=pool_floor) != task["model"]
            or {k:v for k,v in old.items() if k not in ignored} !=
               {k:v for k,v in task.items() if k not in ignored}):
        raise ValueError("migration changed a field outside the qualified source and policy")
    if any(old["source_identity"][k] != task["source_identity"][k]
           for k in ("runtime_packages", "data_manifest_sha256", "python_version")):
        raise ValueError("migration changed runtime or data")


def verify_hard_stop(old, task_ref, result_ref, cp, audit, evidence, *, allow_restart_tail=False):
    """Authenticate a terminal -9 with a worker placeholder correctly bound by the scheduler."""
    if not all(audit["lifecycle"][k]["released"] is True for k in ("processes", "cgroup", "queue")):
        raise ValueError("hard-stop ownership or audit unresolved")
    state = read_bound(audit["execution"]["scheduler"])
    if audit["findings"]:
        from .cutover_review3 import unpublished_signal
        if audit["status"] != "incomplete" or any(f.get("task_id") == old["id"] for f in audit["findings"]):
            raise ValueError("hard-stop worker has unresolved evidence")
        for finding in audit["findings"]:
            unpublished_signal(audit, state, finding)
    elif audit["status"] != "passed":
        raise ValueError("hard-stop audit incomplete")
    row = state["tasks"][old["id"]]
    attempt = row["attempts"][-1]
    directory = Path(attempt["path"])
    heartbeat = read_bound(evidence["heartbeat"])
    index = evidence["checkpoint_index"]
    if (row["status"] != "failed" or row["reason"] != "worker_exit_-9"
            or attempt["reason"] != row["reason"] or attempt["return_code"] != -9
            or not attempt.get("worker_cleanup_verified_unix")
            or row["task_path"] != task_ref["path"] or row["result_path"] != result_ref["path"]
            or row["result_sha256"] != result_ref["sha256"]
            or Path(result_ref["path"]) != directory / "result.json"
            or Path(index["path"]) != directory / "checkpoint.json" or read_bound(index)["current"] != cp
            or Path(cp["path"]).parent != directory / "checkpoints"
            or Path(evidence["heartbeat"]["path"]) != directory / "heartbeat.json"
            or Path(evidence["step_journal"]["path"]) != directory / "steps.jsonl"
            or heartbeat != row["heartbeat"] or heartbeat["pid"] != attempt["worker_process"]["pid"]
            or heartbeat["task_id"] != old["id"] or heartbeat["phase"] != old["phase"]
            or heartbeat["status"] not in ("training", "saving_checkpoint") or heartbeat["tokens"] >= old["target_tokens"]):
        raise ValueError("hard-stop attempt, checkpoint or heartbeat binding changed")
    checked({"path": str(directory / "task.json"), "sha256": task_ref["sha256"]}, "hard-stop worker task")
    with checked(evidence["step_journal"], "hard-stop step journal").open() as stream:
        steps = [json.loads(line) for line in stream]
    tail = verify_stop_journal(steps, heartbeat, cp, old, attempt, allow_restart_tail=allow_restart_tail)
    if evidence.get("resume_prefix"):
        prior_ref = old["stage_a_migration"]
        start = read_bound(prior_ref)["checkpoint"]
        command = attempt["command"]
        expected = {"--task":task_ref["path"], "--output":str(directory), "--device":old["device"],
                    "--receipt":prior_ref["path"], "--receipt-sha256":prior_ref["sha256"],
                    "--resume-checkpoint":start["path"]}
        quantum = old["batch_size"] * old["sequence_length"]
        if (any(command.count(k) != 1 or command[command.index(k)+1] != v for k,v in expected.items())
                or any(row["step"] != start["steps"]+i or row["tokens"] != start["tokens"]+i*quantum
                       for i,row in enumerate(steps,1))
                or not start["steps"] < cp["steps"] <= heartbeat["steps"]
                or (directory/"validation.jsonl").exists()
                or (old.get("eval_interval_tokens", 0) > 0 and
                    start["tokens"]//old["eval_interval_tokens"] != steps[-1]["tokens"]//old["eval_interval_tokens"])):
            raise ValueError("hard-stop journal does not bind the interrupted prefix and its evaluation exposure")
    return tail


def verify_stop_journal(steps, heartbeat, cp, task, attempt, *, allow_restart_tail=False):
    # The worker fsyncs each step before replacing heartbeat.json. On a hard
    # kill exactly one complete update can precede its heartbeat publication.
    position = (heartbeat["steps"], heartbeat["tokens"])
    tail = None
    covered = bool(steps) and (steps[-1]["step"], steps[-1]["tokens"]) == position
    if not covered and allow_restart_tail and len(steps) >= 2:
        before, last = steps[-2:]
        covered = ((before["step"], before["tokens"]) == position and heartbeat["status"] == "training"
                   and last["step"] == before["step"]+1 and last["tokens"] < task["target_tokens"]
                   and last["tokens"]-before["tokens"] == task["batch_size"]*task["sequence_length"]
                   and heartbeat["updated_unix"] <= last["time_unix"] <= attempt["finished_unix"])
        if covered:
            tail = {k:last[k] for k in ("step","tokens","time_unix")}
    if (not covered or cp["steps"] > position[0] or cp["tokens"] > position[1]
            or not any(x["step"] == cp["steps"] and x["tokens"] == cp["tokens"] for x in steps)
            or any(b["step"] != a["step"]+1 or b["tokens"] <= a["tokens"] for a,b in zip(steps,steps[1:]))
            or any(not math.isfinite(x[k]) for x in steps for k in ("loss","objective","grad_norm","lr"))):
        raise ValueError("hard-stop checkpoint lacks a finite training journal through the last heartbeat")
    return tail


def migration_inputs(task, receipt):
    old = read_bound(receipt["original_task"])
    previous = _result(receipt["result"], old, receipt["original_task"]["sha256"])
    cp = receipt["checkpoint"]
    if (receipt.get("schema") != RECEIPT or receipt.get("task_id") != task["id"]
            or receipt.get("new_source_digest") != digest(task["source_identity"])
            or (receipt["mode"] != "checkpoint_only" and previous.get("checkpoint") != cp)
            or task["phase"] not in ("a1","a2","a3","init","preflight")
            or not task.get("source_identity") or task.get("allow_confirmation")):
        raise ValueError("migration source, task or checkpoint binding changed")
    parent = receipt["mode"] == "stage_parent"
    verify_recipe(old, task, parent=parent)
    if parent:
        child_phase = {"a1":"a2", "a2":"a3"}.get(old["phase"])
        if (task["phase"] != child_phase or task["id"] != old["id"].replace("-"+old["phase"]+"-", "-"+child_phase+"-")
                or cp["tokens"] != old["target_tokens"] or cp["tokens"] != task["evaluation_start_tokens"]
                or (task.get("resume_checkpoint"),task.get("resume_checkpoint_sha256")) != (cp["path"],cp["sha256"])
                or previous.get("status") != "completed" or previous.get("health",{}).get("eligible") is not True):
            raise ValueError("migration parent is not the completed preceding rung")
        initial = {"mode":"rerun", "exposure":cp["tokens"]}
    elif receipt["mode"] == "checkpoint_only":
        # A hard-stop can leave only the worker's initial placeholder on disk.
        # Only reuse an incomplete segment with an independently bound initial
        # measurement and no possible additional evaluation before termination.
        from .cutover_review3 import bound_terminal_audit
        audit = bound_terminal_audit(receipt["terminal_audit"])
        prior = read_bound(old["stage_a_migration"])
        migration_inputs(old, prior)
        index = receipt["checkpoint_index"]
        hard_stop = receipt.get("hard_stop") or {}
        prefix = hard_stop.get("resume_prefix") is True
        if hard_stop:
            tail = verify_hard_stop(old, receipt["original_task"], receipt["result"], cp, audit,
                                    {**hard_stop, "checkpoint_index": index}, allow_restart_tail=prefix)
            if tail != hard_stop.get("journal_step_after_last_heartbeat"):
                raise ValueError("hard-stop journal tail changed")
        elif not any(f.get("task_id") == task["id"] and f.get("reason") == "shared_worker_result_binding_mismatch"
                     and f.get("sha256") == receipt["result"]["sha256"] for f in audit["findings"]):
            raise ValueError("unclosed worker lacks a bound terminal observation")
        if (previous.get("status") != "running" or previous.get("checkpoint") is not None
                or old["phase"] not in ("a1", "a2", "a3") or (old["phase"] != "a1" and not receipt.get("hard_stop"))
                or not prior["checkpoint"]["tokens"] < cp["tokens"] < old["target_tokens"]
                or not prefix and 0 < old.get("eval_interval_tokens", 0) < old["target_tokens"]
                or Path(index["path"]) != Path(receipt["result"]["path"]).parent/"checkpoint.json"
                or read_bound(index)["current"] != cp
                or not all(audit["lifecycle"][k]["released"] is True for k in ("processes","cgroup","queue"))):
            raise ValueError("unclosed worker checkpoint or stage-start evaluation lacks evidence")
        if "initial_replay" in receipt:
            if not prefix or prior["mode"] != "stage_parent" or prior["initial_evaluation"]["mode"] != "rerun":
                raise ValueError("unexpected stage-start evaluation replay")
            replay_ref = receipt["initial_replay"]
            replay = _result(replay_ref, old, receipt["original_task"]["sha256"])
            origin = migration_origin(old, replay)
            if (origin is not replay or replay.get("status") != "budget_exhausted" or replay.get("stop_reason") != "signal"
                    or replay["training"]["segment_steps"] != 0 or replay["training"]["segment_tokens"] != 0
                    or replay["evaluation"]["tokens"] != old["eval_tokens"]
                    or replay["causality"]["within_tolerance"] is not True):
                raise ValueError("stage-start replay does not bind the original state and full evaluation")
            initial = initial_origin(replay_ref, old, receipt["original_task"]["sha256"])
        else:
            if prior["mode"] != "signal" or prior["initial_evaluation"]["mode"] != "inherit":
                raise ValueError("unclosed worker has no bound stage-start evaluation")
            initial = prior["initial_evaluation"]
    else:
        if receipt["mode"] == "backward_range_repair":
            proof = read_bound(receipt["qualification"])
            fault = previous["training"]["steps"] + 1
            if (proof.get("schema") != "pilot-r3-backward-range-qualification-v1"
                    or proof.get("original_task") != receipt["original_task"]
                    or proof.get("old_result") != receipt["result"] or proof.get("official_checkpoint") != cp
                    or proof.get("failure_input") != previous.get("failure_capture")
                    or proof.get("new_source_digest") != receipt["new_source_digest"]
                    or previous.get("status") != "failed" or previous.get("stop_reason") != "ArithmeticError"
                    or old["device"] != "cpu" or proof.get("device") != old["device"]
                    or not cp["steps"] < fault or proof.get("fault_step") != fault
                    or proof.get("fault_owners_exact") is not True or proof.get("fault_batch_exact") is not True
                    or proof.get("fault_divisor") not in (8,16) or proof.get("saved", {}).get("steps") != fault
                    or proof.get("restored_step") != fault + 1 or proof.get("successor_state_exact") is not True
                    or proof.get("status") != "passed"):
                raise ValueError("backward range repair lacks a qualified checkpoint, optimizer update or successor")
        elif receipt["mode"] == "shannon_repair":
            acceptance = read_bound(receipt["qualification"])
            qualified = acceptance["qualification"]["npu"]["direct"]
            if (acceptance["original_task"]["sha256"] != receipt["original_task"]["sha256"]
                    or qualified["official_checkpoint"]["sha256"] != cp["sha256"]
                    or previous.get("status") != "failed" or previous.get("error") != qualified["strict_error"]
                    or previous.get("failure_capture",{}).get("sha256") != acceptance["failure_input"]["sha256"]
                    or task["model"]["config"].get("shannon_support_policy") != POLICY):
                raise ValueError("failed state is outside the qualified Shannon repair")
        elif receipt["mode"] in REPAIR_SCHEMAS:
            acceptance = read_bound(receipt["qualification"])
            npu = acceptance["qualification"]["npu"]
            schema = REPAIR_SCHEMAS[receipt["mode"]]
            backend = "cpu" if schema == POOL_ACCEPTANCE else "npu"
            if schema == QUAD_TOTAL_ACCEPTANCE:
                backend = old["device"].split(":")[0]
            trajectory = acceptance["qualification"][backend]
            if schema == POOL_ACCEPTANCE and (old["device"] != "cpu" or task["model"]["config"].get("pool_support_policy") != POLICY):
                raise ValueError("pool repair requires its qualified CPU trajectory and explicit policy")
            direct = trajectory["direct"]
            if (acceptance["schema"] != schema
                    or acceptance["original_task"]["sha256"] != receipt["original_task"]["sha256"]
                    or read_bound(acceptance["original_task"]) != old or acceptance["old_result"] != receipt["result"]
                    or direct["official_checkpoint"] != cp or not direct["original_state_loaded_exact"]
                    or not direct["fault_batch_matched"] or not trajectory["restored"]["passed"]
                    or previous.get("status") != "failed" or previous.get("error") != direct["original_error"]
                    or previous.get("failure_capture") != acceptance["failure_input"]):
                raise ValueError("failed task is outside the verified foundation repair")
            if receipt["mode"] != "kv_no_write_repair":
                sources = task["source_identity"]["source_files"]
                bases = {Path(p).parent for p in sources if Path(p).parts[-3:] == ("src", "tide", "attention.py")}
                if len(bases) != 1:
                    raise ValueError("reception qualification needs one explicit foundation")
                base = bases.pop()
                files = {str(Path(p).relative_to(base)): h for p,h in sources.items() if Path(p).is_relative_to(base)}
                authority = acceptance["authority"]
                restored = trajectory["restored"]
                if schema == QUAD_TOTAL_ACCEPTANCE:
                    for backend in ("cpu", "npu"):
                        replay = acceptance["qualification"][backend]
                        run, reload = replay["direct"], replay["restored"]
                        if (run["official_checkpoint"] != cp or not run["fault_batch_matched"]
                                or not run["passed"] or not reload["passed"] or not reload["discrete_exact"]
                                or not reload["loaded_state_exact"] or reload["checkpoint_sha256"] != run["new_checkpoint"]["sha256"]
                                or reload["step"] != run["new_checkpoint"]["steps"] + 1
                                or run["files_digest"] != authority["files_digest"] or reload["files_digest"] != authority["files_digest"]):
                            raise ValueError("quad recovery is outside the qualified backend trajectory")
                if (not authority.get("commit") or files != authority["files_sha256"]
                        or len({digest(files), direct["files_digest"], restored["files_digest"], authority["files_digest"],
                                npu["capture"]["files_digest"], acceptance["qualification"]["cpu"]["capture"]["files_digest"]}) != 1
                        or not direct["passed"] or not npu["capture"]["passed"] or not acceptance["qualification"]["cpu"]["capture"]["passed"]
                        or not restored["loaded_state_exact"] or not restored["discrete_exact"]
                        or restored["checkpoint_sha256"] != direct["new_checkpoint"]["sha256"]
                        or not cp["steps"] < direct["fault_step"] < direct["new_checkpoint"]["steps"]
                        or restored["step"] != direct["new_checkpoint"]["steps"]+1):
                    raise ValueError("reception repair qualification does not cover the frozen source and successor")
        elif (receipt["mode"] != "signal" or previous.get("status") != "budget_exhausted"
              or previous.get("stop_reason") != "signal" or previous.get("failure_class")
              or previous["training"]["tokens"] != cp["tokens"] or previous["training"]["steps"] != cp["steps"]):
            raise ValueError("migration predecessor is not a bound signal interruption")
        initial = initial_origin(receipt["result"], old, receipt["original_task"]["sha256"])
        if initial is None and cp["tokens"] == task.get("evaluation_start_tokens",0):
            initial = {"mode":"rerun", "exposure":cp["tokens"]}
    if initial is None or receipt["initial_evaluation"] != initial:
        raise ValueError("stage-start evaluation lacks its original exposure")
    return old, previous


def migration_origin(task, result):
    """Verify the exact source transition before following the old attempt chain."""
    from .continuation3 import continuation_origin, restored_private_state
    mark = result["execution_adapter"]
    receipt_ref = task.get("stage_a_migration")
    if (mark.get("schema") != ADAPTER or not receipt_ref or mark.get("receipt") != receipt_ref
            or read_bound(mark["new_task"]) != task or result.get("task_sha256") != mark["new_task"]["sha256"]
            or result.get("source_identity") != task["source_identity"] or result.get("id") != task["id"]):
        raise ValueError("result does not bind its migrated task")
    for ref in (mark["adapter"], mark["worker"]):
        if task["source_identity"]["source_files"].get(ref["path"]) != ref["sha256"]:
            raise ValueError("migration executed outside the frozen source")
        checked(ref, "migration implementation")
    receipt = read_bound(receipt_ref)
    old, previous = migration_inputs(task, receipt)
    cp = receipt["checkpoint"]
    resume, training = result["resume"], result["training"]
    checked(cp, "source migration checkpoint")
    if (mark["checkpoint"] != cp or mark["original_result"] != receipt["result"]
            or mark["original_task"] != receipt["original_task"]
            or resume.get("path") != cp["path"] or resume.get("sha256") != cp["sha256"]
            or resume.get("fallback_used") is not False or resume.get("source_task_id") != old["id"]
            or not restored_private_state(resume, cp)
            or resume.get("checkpoint_load",{}).get("status") != "exact"
            or any(training["from_"+key] != cp[key] or resume["from_"+key] != cp[key]
                   or training["segment_"+key] != training[key]-cp[key] for key in ("tokens","steps"))):
        raise ValueError("migration did not restore the declared state and exposure")
    if receipt["mode"] == "stage_parent":
        return result
    if receipt["mode"] == "checkpoint_only":
        if "initial_replay" in receipt:
            return continuation_origin(old, read_bound(receipt["initial_replay"]))
        earlier = read_bound(old["stage_a_migration"])
        return continuation_origin(read_bound(earlier["original_task"]), read_bound(earlier["result"]))
    return continuation_origin(old, previous)
