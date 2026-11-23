"""Reconcile missing owner receipts after a planned stop, preserving the raw audit."""
import signal
import re
from pathlib import Path

from .io import sha256
from .prepare_resume3 import reference
from .resume_adapter3 import _result
from .scheduler3 import evaluate_outcome
from .source_migration3 import read_bound


HBM_CONTINUATION = "pilot-r3-hbm-admission-continuation-v1"
OWNER_EXIT_CONTINUATION = "pilot-r3-owner-exit-continuation-v1"
PLANNED_CONTROLLER_CONTINUATION = "pilot-r3-planned-controller-continuation-v1"
RESOURCE_MEMORY_CONTINUATION = "pilot-r3-direct-cpu-memory-continuation-v1"
CONTROLLER_CONTINUATIONS = {HBM_CONTINUATION, OWNER_EXIT_CONTINUATION, PLANNED_CONTROLLER_CONTINUATION,
                            RESOURCE_MEMORY_CONTINUATION}
STALE_GROUP_REASON = "shared_group_not_completed:running"


def reviewed_storage_settings(review):
    """Storage fields authorized by a planned controller continuation."""
    if "storage_policy" not in review:
        return {}
    if review.get("schema") != PLANNED_CONTROLLER_CONTINUATION or review["storage_policy"] != "hard_quota":
        raise ValueError("unrecognized controller continuation storage policy")
    return {"storage_reserve_bytes": 0, "max_inflight_checkpoint_bytes": None, "use_live_checkpoint_headroom": False}


def completed_before_signal(audit, scheduler_ref):
    """Verify a published final result when SIGTERM arrived during worker exit."""
    return completed_on_exit(audit, scheduler_ref, reason="target_reached")


def completed_on_exit(audit, scheduler_ref, *, reason):
    """Recover a bound final result from a clean owner exit or a signal during worker exit."""
    from .gate3 import eligible

    if reason not in {"target_reached", STALE_GROUP_REASON}:
        raise ValueError("unrecognized terminal reconciliation")
    state = read_bound(scheduler_ref)
    if (audit.get("status") != "passed" or audit.get("findings")
            or audit["execution"]["scheduler"]["sha256"] != scheduler_ref["sha256"]
            or state.get("remaining_descendants") != [] or not all(audit["lifecycle"][k]["released"]
                    for k in ("processes", "cgroup", "queue"))):
        raise ValueError("worker completion needs its terminal ownership and file audit")
    completed = {}
    for tid, row in state["tasks"].items():
        if row["status"] != "failed" or row.get("reason") != reason:
            continue
        attempt = row["attempts"][-1]
        directory = Path(attempt["path"])
        task_ref = reference(row["task_path"])
        result_ref = reference(directory/"result.json")
        task = read_bound(task_ref)
        result = _result(result_ref, task, task_ref["sha256"])
        heartbeat = read_bound(reference(directory/"heartbeat.json"))
        cp = read_bound(reference(directory/"checkpoint.json"))["current"]
        owner = audit["shared_groups"]["groups"][row["shared_group"]]
        group = owner["group"]
        worker = group["workers"][tid]
        recovery = result["recovery"]["checkpoint_load"]
        if reason == STALE_GROUP_REASON:
            entry = owner["entry"]
            queue = read_bound(reference(Path(entry["path"]) / "queue.json"))
            exit_verified = (
                row.get("observation_error") == reason and attempt["reason"] == reason
                and entry["owner_return_code"] == 0 and group["status"] == "completed" and not group.get("reason")
                and group["remaining_descendants"] == [] and group["allocation_released_unix"]
                and group["queue_released_unix"] and queue["job_id"] == group["job_id"]
                and queue["status"] == "completed" and queue["return_code"] == 0
                and worker["status"] == "completed" and worker["return_code"] == 0
                and attempt["return_code"] in (None, 0) and group["finished_unix"] <= attempt["finished_unix"]
            )
        else:
            exit_verified = (group["reason"] == "signal_15" and worker["return_code"] == -signal.SIGTERM
                             and attempt["return_code"] == worker["return_code"])
        if (not owner["binding_verified"] or owner["owner_alive"] or not exit_verified
                or worker["output"] != str(directory) or not worker["descendant_cleanup_verified"]
                or worker["remaining_descendants"] != [] or worker["pid"] != heartbeat["pid"]
                or worker["pid"] != attempt["worker_process"]["pid"]
                or worker["start_ticks"] != attempt["worker_process"]["start_ticks"]
                or heartbeat != row["heartbeat"] or heartbeat["task_id"] != tid
                or result["status"] != "completed" or heartbeat["status"] != result["status"]
                or worker["result_status"] != result["status"] or result["stop_reason"] != "target_reached"
                or heartbeat["stop_reason"] != result["stop_reason"] or result.get("failure_class")
                or not result["finished_unix"] <= heartbeat["updated_unix"] <= worker["finished_unix"]
                or {worker["result_sha256"], row["result_sha256"]} != {result_ref["sha256"]}
                or {worker["result_path"], row["result_path"]} != {result_ref["path"]}
                or cp != result["checkpoint"] or sha256(cp["path"]) != cp["sha256"]
                or task["target_tokens"] != cp["tokens"] or recovery["status"] != "exact"
                or cp["state_sha256"] != recovery["state_sha256"]
                or any(record[k] != cp[k] for record in (heartbeat, result["training"], recovery)
                       for k in ("tokens", "steps"))
                or eligible(task, result, require_resume=task["phase"] in {"a2","a3"})
                or result.get("causality",{}).get("within_tolerance") is not True):
            raise ValueError("termination lacks an independently verified final worker result: " + tid)
        completed[tid] = result_ref
    return completed


def unpublished_signal(audit, state, finding):
    """Match an unpublished worker signal to its terminal files, owner and exact process."""
    tid = finding.get("task_id")
    if finding.get("reason") != "shared_worker_result_binding_mismatch" or finding.get("kind") != "result":
        raise ValueError("unreconciled terminal audit finding: " + str(finding))
    rec = state["tasks"][tid]
    owner = audit["shared_groups"]["groups"][rec["shared_group"]]
    worker = owner["group"]["workers"][tid]
    last = rec["attempts"][-1]
    path = Path(last["path"])
    task_ref = reference(rec["task_path"])
    task = read_bound(task_ref)
    result_ref = {"path":str(path/"result.json"), "sha256":finding["sha256"]}
    result = _result(result_ref, task, task_ref["sha256"])
    heartbeat = read_bound(reference(path/"heartbeat.json"))
    index = read_bound(reference(path/"checkpoint.json"))
    cp = result["checkpoint"]
    if (not owner["binding_verified"] or owner["owner_alive"]
            or not rec["reason"].startswith("cleanup_evidence_error:")
            or worker.get("status") != "running" or worker.get("result_sha256") or worker.get("return_code") is not None
            or Path(worker["output"]) != path or finding["path"] != result_ref["path"]
            or worker["pid"] != heartbeat["pid"] or worker["start_ticks"] != last["worker_process"]["start_ticks"]
            or worker["pid"] != last["worker_process"]["pid"] or task["id"] != tid or heartbeat["task_id"] != tid
            or result["status"] != heartbeat["status"] or result["status"] != "budget_exhausted"
            or result["stop_reason"] != heartbeat["stop_reason"] or result["stop_reason"] != "signal"
            or result.get("failure_class") or cp != index["current"] or sha256(cp["path"]) != cp["sha256"]
            or any(cp[k] != result["training"][k] or cp[k] != heartbeat[k] for k in ("steps","tokens"))):
        raise ValueError("unpublished owner receipt lacks its observed worker, terminal heartbeat or checkpoint: " + tid)
    return result_ref


def unallocated_worker(tid, record, owner, queue):
    """Identify a signalled queue waiter which never obtained a card or started a worker."""
    group = owner["group"]
    worker = group["workers"][tid]
    attempt = record["attempts"][-1]
    directory = Path(attempt["path"])
    receipt = read_bound(queue["queue_state"])
    if (not owner["binding_verified"] or owner["owner_alive"] or not queue["released"]
            or receipt["job_id"] != group["job_id"] or receipt["physical_ids"] != []
            or receipt["status"] != "failed" or receipt["return_code"] != 1
            or group["status"] != "failed" or group["reason"] != "signal_15"
            or group["remaining_descendants"] != [] or not group["queue_released_unix"]
            or group["allocation_acquired_unix"] is not None or group.get("physical_id") is not None
            or group["physical_device_seconds"] != 0 or group.get("queue_release_error")
            or worker["status"] != "not_started" or any(worker.get(k) is not None for k in
                ("pid", "start_ticks", "started_unix", "return_code", "result_path"))
            or worker["output"] != str(directory) or record.get("heartbeat") or record.get("worker_process")
            or record.get("result_path") not in (None, str(directory / "result.json"))
            or record.get("result_sha256") is not None
            or any(a.get("worker_process") or any(Path(a["path"]).iterdir()) for a in record["attempts"])):
        raise ValueError("stopped queue waiter has allocation or execution evidence: " + tid)
    return {"group":record["shared_group"], "queue":queue["queue_state"], "task":reference(record["task_path"])}


def initial_check_replays(audit, state, *, memory_failure):
    """Identify interrupted initial checks before any published optimizer update."""
    reason = state.get("blocking_reason", "")
    if (not state["config"].get("resource_dispatch") or audit["physical_accounting"]["status"] != "passed"
            or memory_failure and (state.get("status") != "stopped_budget" or not re.fullmatch(
                r"Blocked: (init|a[123]): shared_group_not_completed:GroupBlocked: group_rss_budget_exceeded", reason))
            or not memory_failure and reason != "user_signal"):
        raise ValueError("unexpected resource continuation origin")
    replays, exceeded = {}, False
    for tid, record in state["tasks"].items():
        if not record.get("attempts"):
            continue
        if record["status"] == "completed":
            continue
        owner = audit["shared_groups"]["groups"][record["shared_group"]]
        group = owner["group"]
        manifest = read_bound({"path": owner["entry"]["manifest_path"], "sha256": owner["entry"]["manifest_sha256"]})
        if memory_failure and group.get("reason") == "GroupBlocked: group_rss_budget_exceeded":
            if (manifest["backend"] != "cpu" or group["peak_rss_bytes"] <= manifest["max_rss_bytes"]
                    or manifest["max_rss_bytes"] >= state["config"]["max_rss_bytes"]):
                raise ValueError("unconfirmed per-group CPU memory fault")
            exceeded = True
        path = Path(record["attempts"][-1]["path"])
        if not (path / "result.json").exists():
            continue
        journal = path / "steps.jsonl"
        if journal.exists() and journal.stat().st_size:
            continue
        task_ref = reference(record["task_path"])
        task = read_bound(task_ref)
        result_ref = reference(path / "result.json")
        result = _result(result_ref, task, task_ref["sha256"])
        hb_ref = reference(path / "heartbeat.json")
        hb = read_bound(hb_ref)
        worker = group["workers"][tid]
        if (not owner["binding_verified"] or owner["owner_alive"] or worker["output"] != str(path)
                or task["id"] != tid or hb["task_id"] != tid or hb["pid"] != worker["pid"]
                or result.get("failure_class") or group.get("reason") not in
                ("signal_15", "GroupBlocked: group_rss_budget_exceeded")):
            raise ValueError("initial-check replay lacks its recorded worker: " + tid)
        prior = read_bound(task["stage_a_migration"])["checkpoint"] if task.get("stage_a_migration") else None
        if result["status"] == "running":
            if hb["status"] != "initial_checks" or hb["steps"] or result.get("checkpoint") or (path / "checkpoint.json").exists():
                raise ValueError("unclosed initial check has published progress: " + tid)
        elif result.get("stop_reason") == "signal" and result["status"] == "budget_exhausted":
            cp = result["checkpoint"]
            if (read_bound(reference(path / "checkpoint.json"))["current"] != cp or sha256(cp["path"]) != cp["sha256"]
                    or any(cp[k] != (prior[k] if prior else 0) or cp[k] != hb[k] for k in ("steps", "tokens"))):
                raise ValueError("interrupted initial check advanced past its origin: " + tid)
        else:
            raise ValueError("initial check contains a terminal training outcome: " + tid)
        replays[tid] = {"task": task_ref, "result": result_ref, "heartbeat": hb_ref,
                        "original_resume": task.get("stage_a_migration"), "group": record["shared_group"],
                        "retained_evaluation_reservation": record.get("evaluation_reserved_tokens", 0)}
    if memory_failure and not exceeded:
        raise ValueError("CPU group memory exhaustion lacks its measured peak")
    return replays


def reconcile_stop(audit, scheduler_ref, *, fault_task=None, resource_memory=False):
    state = read_bound(scheduler_ref)
    lifecycle = audit["lifecycle"]
    expected_reason = "user_signal"
    direct_stop = resource_memory or (state.get("config", {}).get("resource_dispatch") and
                                       state.get("blocking_reason") == "user_signal" and fault_task is None)
    replays = initial_check_replays(audit, state, memory_failure=resource_memory) if direct_stop else {}
    if resource_memory:
        expected_reason = state["blocking_reason"]
    if fault_task is not None:
        record = state["tasks"][fault_task]
        expected_reason = f'Blocked: {record["stage"]}: shared_group_not_completed:worker_process_failed:{fault_task}:1'
        if record["status"] != "failed" or read_bound(reference(record["result_path"])).get("status") != "failed":
            raise ValueError("shutdown fault lacks a failed worker result")
    if (audit["execution"]["scheduler"]["sha256"] != scheduler_ref["sha256"]
            or state.get("blocking_reason") != expected_reason or state.get("remaining_descendants") != []
            or lifecycle["status"] != "terminal" or lifecycle["processes"]["released"] is not True
            or lifecycle["cgroup"]["released"] is not True):
        raise ValueError("planned cutover has unresolved process ownership")
    groups = audit["shared_groups"]["groups"]
    stale, signals, completed = {}, {}, {}
    for queue in lifecycle["queue"]["attempts"]:
        if queue.get("released") is True:
            continue
        gid = queue["shared_group"]
        observed = groups[gid]
        group, entry = observed["group"], state["shared_groups"][gid]
        start = entry["allocation_unix"]
        through = entry["cleanup_verified_unix"]
        device_locks = queue["device_locks"]
        if (not observed["binding_verified"] or observed["owner_alive"] or entry["owner_return_code"] != -9
                or group["status"] != "running" or group["allocation_released_unix"] is not None
                or queue["job_id"] != group["job_id"] or queue["recorded_status"] != "running"
                or queue["job_lock"]["status"] not in ("absent", "unlocked")
                or not device_locks or any(lock["status"] != "unlocked" for lock in device_locks)
                or start != group["allocation_acquired_unix"] or through < start
                or not entry["accounting_estimated"] or entry["device_seconds"] + 1e-6 < through - start):
            raise ValueError("missing lease receipt lacks independent release and a full conservative debit: " + gid)
        stale[gid] = {"physical_id":group["physical_id"], "job_lock":queue["job_lock"],
                      "device_locks":device_locks, "charged_through_unix":through, "device_seconds":entry["device_seconds"]}
    excluded = {}
    for finding in audit["findings"]:
        gid, tid = finding.get("group_id"), finding.get("task_id")
        if tid in replays and finding.get("reason") == "shared_worker_result_binding_mismatch":
            if (finding.get("kind") != "result" or {k:finding[k] for k in ("path","sha256")} != replays[tid]["result"]):
                raise ValueError("initial-check audit finding differs from verified result")
            continue
        if ((gid in stale and finding.get("reason") in ("group allocation lacks terminal interval","shared_group_not_terminal"))
                or stale and finding == {"scope":"queue", "status":"unknown", "reason":None}):
            continue
        if fault_task and finding.get("kind") == "checkpoint" and finding.get("reason") == "checkpoint_file_has_no_recorded_generation_digest":
            # Interrupted atomic publication may leave complete or partial data
            # beyond the last published index. Do not adopt or delete that file.
            path = Path(finding["path"])
            attempt = path.parent.parent
            tid = attempt.parent.name
            rec = state["tasks"][tid]
            index = read_bound(reference(attempt / "checkpoint.json"))
            generation = re.fullmatch(r"step-(\d+)-[a-f0-9]+[.]pt", path.name)
            if (rec.get("reason") != "worker_exit_-9" or not generation or path.parent != attempt / "checkpoints"
                    or Path(rec["attempts"][-1]["path"]) != attempt or rec["attempts"][-1]["return_code"] != -9
                    or not rec["attempts"][-1].get("worker_cleanup_verified_unix")
                    or not index["current"]["steps"] < int(generation[1]) <= rec["heartbeat"]["steps"]
                    or any(g.get("path") == str(path) for g in (index.get("previous") or {}, index["current"]))):
                raise ValueError("unindexed file is not an interrupted post-index generation")
            excluded[tid] = {"unpublished":reference(path), "adoptable_index":reference(attempt / "checkpoint.json")}
            continue
        signals[tid] = unpublished_signal(audit, state, finding)
    for tid, rec in state["tasks"].items():
        if rec["status"] != "completed" or rec.get("shared_group") not in stale:
            continue
        owner = groups[rec["shared_group"]]
        worker = owner["group"]["workers"][tid]
        task_ref = reference(rec["task_path"])
        result_ref = {"path":rec["result_path"], "sha256":rec["result_sha256"]}
        task = read_bound(task_ref)
        result = _result(result_ref, task, task_ref["sha256"])
        if (worker.get("status") != "completed" or worker.get("return_code") != 0
                or worker.get("result_sha256") != result_ref["sha256"] or worker.get("result_path") != result_ref["path"]
                or worker.get("descendant_cleanup_verified") is not True or worker.get("remaining_descendants") != []
                or evaluate_outcome(task, result, 0)[0] != "completed" or not result["health"]["eligible"]):
            raise ValueError("completed worker on subsequently stopped owner lacks its individual receipt: " + tid)
        completed[tid] = result_ref
    resolution = {"stale_owner_receipts":stale, "signal_results":signals, "completed_worker_receipts":completed}
    if direct_stop:
        resolution["initial_check_replays"] = replays
    queues = {q["shared_group"]:q for q in lifecycle["queue"]["attempts"] if q.get("shared_group")}
    unstarted = {}
    for tid, record in state["tasks"].items():
        if record.get("reason") in ("cleanup_evidence_error:Blocked:shared_group_lease_release_missing",
                                    "shared_group_not_completed:signal_15"):
            gid = record["shared_group"]
            group = groups[gid]["group"]
            if group["workers"][tid]["status"] == "not_started" and (
                    record["reason"] != "shared_group_not_completed:signal_15" or group.get("allocation_acquired_unix") is None):
                unstarted[tid] = unallocated_worker(tid, record, groups[gid], queues[gid])
    if unstarted:
        resolution["unstarted_before_allocation"] = unstarted
    if fault_task:
        resolution["excluded_unpublished_generations"] = excluded
    return resolution


def bound_terminal_audit(ref):
    """Return the migration view after independently checking every raw shutdown finding."""
    receipt = read_bound(ref)
    if receipt.get("schema") != "pilot-r3-failed-shutdown-review-v1":
        return receipt
    raw = read_bound(receipt["original_audit"])
    resolved = reconcile_stop(raw, receipt["scheduler"], fault_task=receipt["fault_task"])
    if resolved != receipt["resolution"] or any(not g["binding_verified"] for g in raw["shared_groups"]["groups"].values()):
        raise ValueError("failed-shutdown resolution differs from its bound primary records")
    queue = {**raw["lifecycle"]["queue"], "released":True, "release_basis":"verified process exit and lease locks"}
    return {**raw, "status":"passed", "findings":[], "reconciled_stop":resolved,
            "lifecycle":{**raw["lifecycle"], "queue":queue}}
