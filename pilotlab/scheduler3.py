"""Finite round-three process supervision; no Torch or vendor import.

Scientific decisions belong to round3/gate3. This supervisor owns immutable
requests, process identities, cooperative queue admission and resource ledgers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .io import atomic_json, sha256
from .lifecycle3 import cancelled_before_allocation
from .scheduler import (QUEUE_HELPER, child_subreaper, current_quota, family_members,
                        immutable_json, proc_identity, process_table, storage_headroom)

GIB = 1024**3
TERMINAL = {"completed", "failed", "recipe_rejected", "stopped_budget", "interrupted"}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def freeze_json(path, value):
    path = Path(path)
    # The durable contract is JSON. Python tuples (e.g. multiscale group
    # widths) and their loaded lists describe the same immutable request.
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if read(path) != value:
            raise ValueError(f"immutable request differs: {path}")
    else:
        immutable_json(path, value)
    return path


def live(identity):
    if not identity:
        return False
    actual = proc_identity(identity["pid"])
    return bool(actual and actual["start_ticks"] == identity["start_ticks"]
                and actual["state"] != "Z")


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def queue_interval(log, now=None):
    """Use allocation/exit events, including jobs shorter than the poll period."""
    start = end = None
    try:
        with Path(log).open() as stream:
            for line in stream:
                if not line.startswith('{"event": "ascend_queue"'):
                    # emit(sort_keys=True) puts event after elapsed_seconds.
                    if '"event": "ascend_queue"' not in line:
                        continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if event.get("status") == "running":
                    start = timestamp(event["timestamp"])
                elif start is not None and event.get("status") in {"completed", "failed", "cancelled"}:
                    end = timestamp(event["timestamp"])
    except FileNotFoundError:
        pass
    if start is not None and (not math.isfinite(start) or end is not None and (not math.isfinite(end) or end < start)):
        raise ValueError("invalid queue device-time interval")
    seconds = max(0.0, (end if end is not None else now or time.time())-start) if start else 0.0
    return start, end, seconds


def evaluate_outcome(task, result, return_code, queue=None):
    """Exit zero alone never establishes the declared exposure or health."""
    if not isinstance(result, dict):
        return "failed", "missing_worker_result"
    if not task.get("source_identity") or result.get("source_identity") != task.get("source_identity"):
        return "failed", "worker_source_identity_mismatch"
    status = result.get("status")
    if status in {"interrupted", "stopped_budget", "budget_exhausted", "incomplete"}:
        return ("stopped_budget" if status != "interrupted" else status), result.get("stop_reason", status)
    if status == "recipe_rejected" and result.get("failure_class") == "declared_optimizer_instability":
        return "recipe_rejected", result.get("stop_reason", status)
    if return_code not in (0, None):
        return "failed", result.get("stop_reason") or f"worker_exit_{return_code}"
    if queue is not None and (queue.get("status") != "completed" or queue.get("return_code") != 0):
        return "failed", "queue_not_completed"
    if status != "completed":
        return "failed", result.get("stop_reason") or result.get("error") or "worker_not_completed"
    if result.get("training", {}).get("tokens") != task["target_tokens"]:
        return "failed", "declared_token_target_not_reached"
    if not task.get("source_identity") or result.get("source_identity") != task.get("source_identity"):
        return "failed", "worker_source_identity_mismatch"
    return "completed", None


def evaluation_bound(task):
    """Reserve all declared evaluations, including conservative retry overhead."""
    start = task.get("evaluation_start_tokens", 0)
    target, interval = task["target_tokens"], task.get("eval_interval_tokens", 1048576)
    if not isinstance(start, int) or start < 0 or start > target:
        raise ValueError("invalid evaluation starting exposure")
    middle = max(0, (target-1)//interval-start//interval) if interval > 0 else 0
    return (2+middle)*task.get("eval_tokens", 16384) + (
        task.get("confirmation_tokens", 131072) if task.get("allow_confirmation") else 0) + 32


class Blocked(RuntimeError):
    pass


def worker_limit(config, default=16):
    configured = config.get("max_workers", default)
    if configured is not None:
        return configured
    if not (config.get("shared_execution") and config.get("dynamic_shared_admission")):
        raise ValueError("resource-based concurrency requires dynamic shared admission")
    # This upper bound follows from the two pools. It also bounds storage
    # forecasts without introducing a third, independent worker limit.
    cpu_workers = config.get("max_cpu_workers", 40)
    devices = config.get("max_npu_devices", 8)
    workers_per_device = min(16, config.get("npu_workers_per_device", 16))
    return cpu_workers + devices * workers_per_device


class Supervisor:
    def __init__(self, config, output):
        self.config, self.output = config, Path(output).resolve()
        if "host_cpu_affinity" in config:
            requested = set(config["host_cpu_affinity"])
            if not requested or len(requested) > config["max_cpu_task_threads"]:
                raise ValueError("host CPU affinity exceeds the CPU budget or is empty")
            os.sched_setaffinity(0, requested & os.sched_getaffinity(0))
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output/"controller.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child_subreaper(True)
        self.processes = {}
        self.handles = {}
        self.stop_reason = None
        self.state_path = self.output/"scheduler.json"
        self.state = read(self.state_path)
        if self.state:
            if self.state["config"] != config:
                raise ValueError("controller configuration changed on restart")
        else:
            self.state = {"schema": "pilot-r3-scheduler-v1", "config": config,
                          "started_unix": time.time(), "status": "preparing",
                          "stage": "preparing", "stage_b_gate": "closed", "tasks": {}, "events": [],
                          "all_seen_processes": {}, "peak_actual_npu_concurrency": 0}
        self.state["controller_identity"] = proc_identity(os.getpid(), strict=True)
        self.last_resources = 0
        self.resource_cache = {}
        self._save()

    def _save(self):
        self.state["updated_unix"] = time.time()
        self.state["counts"] = dict(Counter(x["status"] for x in self.state["tasks"].values()))
        self.state["device_seconds"] = self.device_seconds()
        atomic_json(self.state_path, self.state)
        ledger_path = Path(self.config["budget_ledger"])
        with Path(str(ledger_path)+".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = read(ledger_path)
            if ledger is None:
                raise ValueError("budget ledger must be created once before first submission")
            ledger.setdefault("segments", {})[str(self.output)] = {
                "device_seconds": self.state["device_seconds"],
                "preparation": bool(self.config.get("preparation")),
                "preparation_device_seconds": self.preparation_device_seconds(),
                "preparation_reserved_tokens": sum(v.get("preparation_reserved_tokens", 0) for v in self.state["tasks"].values()),
                "updated_unix": self.state["updated_unix"], "status": self.state["status"]}
            ledger["device_seconds"] = sum(v["device_seconds"] for v in ledger["segments"].values())
            atomic_json(ledger_path, ledger)

    def device_seconds(self):
        return sum(x.get("device_seconds", 0) for x in self.state["tasks"].values())

    def preparation_device_seconds(self):
        return sum(v.get("device_seconds", 0) for v in self.state["tasks"].values()
                   if self.config.get("preparation") or v.get("preparation"))

    def actual_npu_concurrency(self, records):
        return sum(r["status"] == "running" and r.get("queue", {}).get("status") == "running"
                   for r in records)

    def supervision_tick(self):
        """Observe resource owners even when their individual tasks have finished."""

    def stage_drained(self):
        """Finish any persistent owners before returning a completed stage."""
        return True

    def start_ready_tasks(self, tasks, slots):
        for task in tasks:
            if slots <= 0:
                break
            record = self.state["tasks"][task["id"]]
            if record["status"] == "pending":
                self._start(task, record)
                slots -= 1

    def verify_completed(self, task, record):
        if record.get("imported_completion"):
            from .resume3 import verify_imported_completion
            verify_imported_completion(self.config, task, record)
            return
        result_path = Path(record["result_path"])
        if not result_path.is_file() or sha256(result_path) != record.get("result_sha256"):
            raise Blocked("completed_result_missing_or_changed")
        last_attempt = record["attempts"][-1]
        q = read(Path(last_attempt["path"])/"queue.json", {}) if task.get("device", "npu:0").startswith("npu") else None
        result = read(result_path)
        status, reason = evaluate_outcome(task, result, last_attempt.get("return_code"), q)
        if result.get("task_sha256") != sha256(record["task_path"]):
            status, reason = "failed", "worker_task_identity_mismatch"
        if status != "completed":
            raise Blocked(f"completed_result_invalid:{reason}")

    def request_stop(self, signum=None, frame=None):
        self.stop_reason = self.stop_reason or ("user_signal" if signum else "controller_stop")

    def resource_reason(self):
        now = time.time()
        ledger = read(self.config["budget_ledger"])
        current_seconds = max(self.state.get("device_seconds", 0), self.device_seconds())
        other_segments = [v for key, v in ledger.get("segments", {}).items() if key != str(self.output)]
        device_seconds = max(ledger.get("device_seconds", 0), current_seconds +
                             sum(v.get("device_seconds", 0) for v in other_segments))
        if now >= ledger["deadline_unix"]:
            return "global_deadline"
        if device_seconds >= ledger["max_device_seconds"]:
            return "global_device_hour_limit"
        preparation_seconds = sum(v.get("preparation_device_seconds", v.get("device_seconds", 0) if v.get("preparation") else 0)
                                  for v in other_segments)
        current_preparation = current_seconds if self.config.get("preparation") else self.preparation_device_seconds()
        preparation_active = self.config.get("preparation") or any(
            v.get("preparation") and v["status"] in {"pending", "running"} for v in self.state["tasks"].values())
        if preparation_active and preparation_seconds+current_preparation >= 48*3600:
            return "preparation_device_hour_limit"
        if now-self.last_resources >= 15:
            q = current_quota()
            free = shutil.disk_usage(self.output).free
            table = process_table(strict=True)
            seeds = {os.getpid()}
            seeds.update(v["process"]["pid"] for v in self.state["tasks"].values()
                         if v.get("process") and live(v["process"]))
            family = family_members(seeds, table)
            rss = sum(table[p]["rss_bytes"] for p in family)
            for pid in family:
                item = table[pid]
                self.state["all_seen_processes"][f'{pid}:{item["start_ticks"]}'] = item
            self.resource_cache = {"observed_unix": now, "quota": asdict(q),
                "filesystem_free_bytes": free, "rss_bytes": rss,
                "quota_limit_basis": "hard", "available_storage_bytes": storage_headroom(q, free)}
            self.state["resources"] = self.resource_cache
            self.last_resources = now
        reserve = self.config.get("storage_reserve_bytes", 24*GIB)
        if self.resource_cache.get("available_storage_bytes", 0) < reserve:
            return "storage_reserve_exhausted"
        if self.resource_cache.get("rss_bytes", 0) > self.config.get("max_rss_bytes", 512*GIB):
            return "process_tree_rss_limit"
        return None

    def reserve_task(self, task, record):
        if (record.get("attempts") and record["attempts"][-1].get("status") == "not_started"
                and record["attempts"][-1].get("reason") == "hbm_admission_deferred"):
            # The verified pre-launch deferral consumed no evaluation or
            # preparation tokens. The original reservation is still charged.
            return
        evaluation_key = task.get("evaluation_budget_key")
        if evaluation_key:
            reserve = evaluation_bound(task)
            if record.get("imported_from") and not record.get("attempts"):
                from .resume3 import resume_reservation
                reserve = resume_reservation(self.config, task, record)
            elif not record.get("attempts") and task["id"] in self.config.get("migration_evaluation_reservations", {}):
                extra = self.config["migration_evaluation_reservations"][task["id"]]
                if type(extra) is not int or not 0 <= extra <= reserve:
                    raise Blocked("invalid_source_migration_evaluation_reservation")
                reserve = extra
            used = sum(v.get("evaluation_reserved_tokens", 0) for v in self.state["tasks"].values()
                       if v.get("evaluation_budget_key") == evaluation_key)
            prior = self.config.get("prior_evaluation_reservations", {}).get(evaluation_key, 0)
            if type(prior) is not int or prior < 0:
                raise Blocked(f"invalid_prior_evaluation_reservation:{evaluation_key}")
            used += prior
            if used+reserve > task["evaluation_token_limit"]:
                raise Blocked(f"evaluation_token_limit:{evaluation_key}")
            record["evaluation_reserved_tokens"] = record.get("evaluation_reserved_tokens", 0)+reserve
            record["evaluation_budget_key"] = evaluation_key
        if self.config.get("preparation") or task.get("phase") == "preflight":
            reserve_tokens = int(task.get("budget_tokens", task["target_tokens"] + 2*task.get("eval_tokens", 16384)+32))
            ledger = read(self.config["budget_ledger"])
            claims = ledger.get("preparation_token_reservations", {})
            spent = sum(v["reserved_tokens"] for v in claims.values())
            spent += sum(v.get("preparation_reserved_tokens", 0) for key, v in ledger.get("segments", {}).items()
                         if key != str(self.output) and key not in claims)
            local_spent = sum(v.get("preparation_reserved_tokens", 0) for v in self.state["tasks"].values())
            if str(self.output) not in claims:
                spent += local_spent+reserve_tokens
            elif local_spent+reserve_tokens > claims[str(self.output)]["reserved_tokens"]:
                raise Blocked("preparation_segment_token_limit")
            if spent > 64*2**20:
                raise Blocked("preparation_token_limit")
            record["preparation_reserved_tokens"] = record.get("preparation_reserved_tokens", 0)+reserve_tokens

    def retry_transient(self, task, record, result):
        """At most one same-request retry, backed by intact same-source state."""
        if (record.get("status") != "failed" or record.get("observation_error") or record.get("resume_receipt")
                or not isinstance(result, dict) or not task.get("source_identity")
                or result.get("source_identity") != task["source_identity"]
                or result.get("failure_class") != "infrastructure_transient"
                or result.get("task_sha256") != sha256(record["task_path"])
                or len(record["attempts"]) >= 2):
            return False
        checkpoint = result.get("checkpoint") or {}
        if not checkpoint.get("path") or not checkpoint.get("sha256"):
            return False
        cp = Path(checkpoint["path"])
        if not cp.is_file() or sha256(cp) != checkpoint["sha256"]:
            return False
        record.update(status="pending", retry_checkpoint=str(cp),
                      retry_checkpoint_sha256=checkpoint["sha256"])
        return True

    def _start(self, task, record):
        self.reserve_task(task, record)
        index = len(record["attempts"])
        attempt = self.output/"tasks"/task["id"]/f"attempt-{index}"
        attempt.mkdir(parents=True, exist_ok=False)
        task_path = self.output/"tasks"/task["id"]/"task.json"
        freeze_json(task_path, task)
        worker = [self.config["python"], "-u", "-m", self.config.get("worker_module", "pilotlab.worker3"),
                  "--task", str(task_path), "--output", str(attempt), "--data", self.config["data"],
                  "--device", task.get("device", "npu:0"), "--deadline-unix", str(self.config["deadline_unix"])]
        if record.get("retry_checkpoint"):
            worker += ["--resume-checkpoint", record["retry_checkpoint"]]
        if task.get("device", "npu:0").startswith("npu"):
            job_id = f'{self.config["run_id"]}-{task["id"]}-{index}'
            if len(job_id) > 128:
                job_id = job_id[:95]+"-"+hashlib.sha256(job_id.encode()).hexdigest()[:32]
            command = [self.config["python"], str(QUEUE_HELPER), "--job-id", job_id,
                "--project", "tide-learning-lab-r3", "--priority", "0", "--npu-count", "1",
                "--max-wait-seconds", str(max(1, int(self.config["deadline_unix"]-time.time()))),
                "--state-file", str(attempt/"queue.json"), "--", *worker]
        else:
            command = worker
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                       "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(task.get("cpu_threads", 4)),
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": str(task.get("cpu_threads", 4))}
        environment["PYTHONPATH"] = os.pathsep.join([self.config["source"], self.config["foundation"]+"/src"])
        log = (attempt/"stdout.log").open("a", buffering=1)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env=environment, cwd=self.config["source"], start_new_session=True)
        # Register ownership before /proc or any durable write can fail.
        self.processes[task["id"]] = process
        self.handles[task["id"]] = log
        record.update(status="running", task_path=str(task_path), provisional_pid=process.pid)
        identity = proc_identity(process.pid, strict=True)
        info = {"path": str(attempt), "process": identity, "command": command,
                "started_unix": time.time(), "status": "running", "device_seconds": 0.0}
        record["attempts"].append(info)
        record.update(status="running", process=identity, task_path=str(task_path))
        self._save()

    def owned_members(self):
        """Track surviving descendants even if their original root has exited."""
        table = process_table(strict=True)
        seeds = {os.getpid()}
        for identity in self.state["all_seen_processes"].values():
            actual = table.get(identity["pid"])
            if actual and actual["start_ticks"] == identity["start_ticks"]:
                seeds.add(identity["pid"])
        members = family_members(seeds, table)-{os.getpid()}
        # A bare process-group number may have been reused. Descendants are
        # instead recovered through the subreaper and exact recorded identities.
        result = {pid: table[pid] for pid in members if table[pid]["state"] != "Z"}
        for pid, item in result.items():
            self.state["all_seen_processes"][f'{pid}:{item["start_ticks"]}'] = item
        return result

    def _poll(self, task, record):
        attempt = record["attempts"][-1]
        directory = Path(attempt["path"])
        is_npu = task.get("device", "npu:0").startswith("npu")
        q = read(directory/"queue.json", {}) if is_npu else None
        if is_npu:
            start, end, seconds = queue_interval(directory/"stdout.log")
            if start is None and q.get("status") in {"running", "completed"}:
                # A damaged log must neither authorize completion nor erase
                # consumed device time; conservatively charge the full wait.
                seconds = max(0, time.time()-attempt["started_unix"])
                record["observation_error"] = "queue_allocation_time_missing"
                attempt["accounting_estimated"] = True
            if start is not None and end is None and q.get("status") == "completed":
                seconds = max(seconds, time.time()-attempt["started_unix"])
                record["observation_error"] = "queue_release_time_missing"
                attempt["accounting_estimated"] = True
            attempt.update(allocation_unix=start, release_unix=end, device_seconds=seconds)
            record["queue"] = q
        record["device_seconds"] = sum(v.get("device_seconds", 0) for v in record["attempts"])
        heartbeat = read(directory/"heartbeat.json")
        if heartbeat:
            record["heartbeat"] = heartbeat
        process = self.processes.get(task["id"])
        code = process.poll() if process else None
        if live(record.get("process")):
            return False
        if process:
            code = process.wait()
            self.processes.pop(task["id"], None)
        handle = self.handles.pop(task["id"], None)
        if handle:
            handle.close()
        if is_npu and code is None:
            code = q.get("return_code")
        result_path = directory/"result.json"
        try:
            result = read(result_path)
        except (OSError, ValueError, TypeError) as exc:
            result = None
            record["observation_error"] = f"unreadable_worker_result:{type(exc).__name__}"
        if is_npu and result is None and cancelled_before_allocation(directory, q):
            status, reason = "interrupted", "cancelled_before_allocation"
        else:
            status, reason = evaluate_outcome(task, result, code, q)
        if result is not None and result.get("task_sha256") != sha256(record["task_path"]):
            status, reason = "failed", "worker_task_identity_mismatch"
            record["observation_error"] = reason
        if (result is not None and task.get("evaluation_budget_key")
                and result.get("evaluation", {}).get("tokens", 0) > evaluation_bound(task)):
            status, reason = "failed", "worker_evaluation_exceeds_declared_bound"
            record["observation_error"] = reason
        identity = record.get("process")
        if identity and any(p["pgrp"] == identity["pgrp"] for p in self.owned_members().values()):
            status, reason = "failed", "worker_descendants_still_alive"
        if record.get("observation_error"):
            status, reason = "failed", record["observation_error"]
        attempt.update(status=status, return_code=code, finished_unix=time.time(), reason=reason)
        record.update(status=status, reason=reason, result_path=str(result_path),
                      result_sha256=sha256(result_path) if result_path.exists() else None)
        # Retrying source/model/optimizer errors is forbidden. A retry keeps
        # the immutable request and uses the last verified full checkpoint.
        self.retry_transient(task, record, result)
        self._save()
        return True

    def run(self, stage, tasks, *, allowed_failures=()):
        if set(allowed_failures)-{"capacity_error"} or allowed_failures and any(
                task.get("phase") != "preflight" or task.get("purpose") != "capacity_calibration" for task in tasks):
            raise ValueError("only declared capacity calibration may tolerate capacity errors")
        if len({t["id"] for t in tasks}) != len(tasks):
            raise ValueError("duplicate task id")
        for task in tasks:
            if Path(task["id"]).name != task["id"] or task["id"] in {".", ".."}:
                raise ValueError("unsafe task id")
            if task["target_tokens"] <= 0:
                raise ValueError("positive cumulative token target required")
            freeze_json(self.output/"tasks"/task["id"]/"task.json", task)
            previous = self.state["tasks"].get(task["id"])
            if previous is None:
                self.state["tasks"][task["id"]] = {"status": "pending", "stage": stage,
                    "model_id": task["model_id"], "device": task.get("device", "npu:0"),
                    "preparation": bool(self.config.get("preparation") or task.get("phase") == "preflight"),
                    "attempts": [], "device_seconds": 0.0}
            elif previous["stage"] != stage:
                raise ValueError("task reused across controller stages")
            elif previous["status"] == "completed":
                self.verify_completed(task, previous)
        self.state.update(status="running", stage=stage)
        self._save()
        try:
            while True:
                self.supervision_tick()
                for task in tasks:
                    record = self.state["tasks"][task["id"]]
                    if record["status"] == "running":
                        self._poll(task, record)
                records = [self.state["tasks"][t["id"]] for t in tasks]
                def allowed(record):
                    if not allowed_failures or record["status"] != "failed" or not record.get("result_path"):
                        return False
                    if sha256(record["result_path"]) != record.get("result_sha256"):
                        return False
                    result = read(record["result_path"], {})
                    return (result.get("task_sha256") == sha256(record["task_path"])
                            and result.get("failure_class") in allowed_failures
                            and result.get("source_identity") == read(record["task_path"]).get("source_identity")
                            and not record.get("observation_error"))
                failure = next((r for r in records if r["status"] in {"failed", "stopped_budget", "interrupted"}
                                and not allowed(r)), None)
                if failure:
                    self.stop_reason = f'{stage}: {failure.get("reason")}'
                self.stop_reason = self.stop_reason or self.resource_reason()
                if self.stop_reason:
                    raise Blocked(self.stop_reason)
                active = sum(r["status"] == "running" for r in records)
                allocated = self.actual_npu_concurrency(records)
                self.state["actual_npu_concurrency"] = allocated
                self.state["peak_actual_npu_concurrency"] = max(self.state["peak_actual_npu_concurrency"], allocated)
                slots = worker_limit(self.config)-active
                self.start_ready_tasks(tasks, slots)
                self._save()
                if all(r["status"] in TERMINAL for r in records) and self.stage_drained():
                    if self.owned_members():
                        raise Blocked("stage_has_live_descendants")
                    return {t["id"]: read(self.state["tasks"][t["id"]]["result_path"]) for t in tasks}
                time.sleep(self.config.get("poll_seconds", 2))
        except BaseException as error:
            self.stop_reason = self.stop_reason or f"{type(error).__name__}: {error}"
            self.stop_all()
            raise

    def run_dynamic(self, stage, producer):
        """Drive a finite dependency producer through one resource supervisor.

        Per-task stage names and immutable requests survive restart. Only
        verified terminal results reach the producer; no concurrent run()
        calls or independent resource pools are introduced.
        """
        requests, permissions, verified = {}, {}, {}
        self.state.update(status="running", stage=stage)
        try:
            while True:
                self.supervision_tick()
                for task_id, task in requests.items():
                    record = self.state["tasks"][task_id]
                    if record["status"] == "running":
                        self._poll(task, record)
                    if task_id in verified:
                        continue
                    if record["status"] == "completed":
                        self.verify_completed(task, record)
                        verified[task_id] = read(record["result_path"])
                    elif record["status"] == "recipe_rejected":
                        result = read(record["result_path"])
                        if record.get("imported_from") and not record.get("attempts"):
                            from .resume3 import imported_entry
                            imported, original, inherited = imported_entry(self.config, task_id)
                            if (imported.get("disposition") != "recipe_rejected" or original != task
                                    or inherited != result or imported.get("result") != {
                                        "path": record["result_path"], "sha256": record.get("result_sha256")}):
                                raise Blocked("dynamic_imported_recipe_rejection_invalid")
                            rejection_code = imported.get("worker_return_code")
                        else:
                            rejection_code = record["attempts"][-1].get("return_code") if record.get("attempts") else None
                        if (record.get("observation_error") or sha256(record["result_path"]) != record.get("result_sha256")
                                or result.get("task_sha256") != sha256(record["task_path"])
                                or evaluate_outcome(task, result, rejection_code)[0] != "recipe_rejected"):
                            raise Blocked("dynamic_recipe_rejection_evidence_invalid")
                        verified[task_id] = result
                    elif record["status"] in {"failed", "stopped_budget", "interrupted"}:
                        result = read(record["result_path"], {}) if record.get("result_path") else {}
                        allowed = (record["status"] == "failed" and not record.get("observation_error")
                            and result.get("failure_class") in permissions[task_id]
                            and sha256(record["result_path"]) == record.get("result_sha256")
                            and result.get("task_sha256") == sha256(record["task_path"])
                            and result.get("source_identity") == task.get("source_identity"))
                        if not allowed:
                            raise Blocked(f'{record["stage"]}: {record.get("reason")}')
                        verified[task_id] = result
                update = producer(dict(verified))
                if not isinstance(update, dict) or type(update.get("done")) is not bool:
                    raise ValueError("dynamic producer requires requests and explicit done")
                for request in update.get("requests", []):
                    tasks = request["tasks"]
                    allowed = frozenset(request.get("allowed_failures", ()))
                    if allowed - {"capacity_error"} or allowed and any(
                            t.get("phase") != "preflight" or t.get("purpose") != "capacity_calibration" for t in tasks):
                        raise ValueError("only declared capacity calibration may tolerate capacity errors")
                    for task in tasks:
                        task_id = task["id"]
                        if Path(task_id).name != task_id or task_id in {".", ".."} or task["target_tokens"] <= 0:
                            raise ValueError("unsafe dynamic task or nonpositive exposure")
                        if task_id in requests and (requests[task_id] != task or permissions[task_id] != allowed):
                            raise ValueError("dynamic request or failure permission changed")
                        record = self.state["tasks"].get(task_id)
                        if record is None:
                            self.state["tasks"][task_id] = {"status": "pending", "stage": request["stage"],
                                "model_id": task["model_id"], "device": task.get("device", "npu:0"),
                                "preparation": bool(self.config.get("preparation") or task.get("phase") == "preflight"),
                                "attempts": [], "device_seconds": 0.}
                        elif record["stage"] != request["stage"]:
                            raise ValueError("task reused across controller stages")
                        placement = getattr(self, "dispatch_placement", None)
                        if placement:
                            placement.register(task, self.state["tasks"][task_id])
                        else:
                            freeze_json(self.output/"tasks"/task_id/"task.json", task)
                        requests[task_id], permissions[task_id] = task, allowed
                self.stop_reason = self.stop_reason or self.resource_reason()
                if self.stop_reason:
                    raise Blocked(self.stop_reason)
                if any(self.state["tasks"][key]["status"] in TERMINAL and key not in verified for key in requests):
                    self._save()
                    continue
                records = [self.state["tasks"][key] for key in requests]
                active = sum(row["status"] == "running" for row in self.state["tasks"].values())
                allocated = self.actual_npu_concurrency(records)
                self.state["actual_npu_concurrency"] = allocated
                self.state["peak_actual_npu_concurrency"] = max(self.state["peak_actual_npu_concurrency"], allocated)
                self.start_ready_tasks(list(requests.values()), max(0, worker_limit(self.config)-active))
                self._save()
                if update["done"] and len(verified) == len(requests) and self.stage_drained():
                    if self.owned_members():
                        raise Blocked("dynamic_stage_has_live_descendants")
                    return verified
                if not update["done"] and len(verified) == len(requests):
                    raise Blocked("dynamic_producer_stalled_without_work")
                # Reused terminal evidence can advance a dependency without
                # paying a poll interval for every imported stage.
                if any(self.state["tasks"][key]["status"] in TERMINAL and key not in verified for key in requests):
                    continue
                time.sleep(self.config.get("poll_seconds", 2))
        except BaseException as error:
            self.stop_reason = self.stop_reason or f"{type(error).__name__}: {error}"
            self.stop_all()
            raise

    def stop_all(self):
        # Publish the closed gate before fallible lifecycle observation.
        self.state.update(status="a_blocked_needs_repair", stage_b_gate="closed",
                          blocking_reason=self.stop_reason or "controller_exception")
        try:
            self._save()
        except Exception as error:
            # Broken observation or a full filesystem must never prevent the
            # already-registered process tree from receiving stop signals.
            self.state["cleanup_save_error"] = f"{type(error).__name__}: {error}"
        def members():
            try:
                return self.owned_members()
            except Exception as exc:
                self.state["cleanup_observation_error"] = f"{type(exc).__name__}:{exc}"
                return {}
        records = [r for r in self.state["tasks"].values() if r["status"] == "running"]
        for process in self.processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for record in records:
            if live(record.get("process")):
                try:
                    os.killpg(record["process"]["pgrp"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for pid in members():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        stop = time.monotonic()+90
        while time.monotonic() < stop and members():
            for process in self.processes.values():
                process.poll()
            time.sleep(0.5)
        for pid in members():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in self.processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    self.state["cleanup_observation_error"] = "registered process did not terminate"
        for record in records:
            try:
                task = read(record["task_path"])
                if not live(record.get("process")) and record.get("attempts"):
                    self._poll(task, record)
                elif not record.get("attempts"):
                    record.update(status="failed", reason="spawn_identity_not_established")
            except Exception as exc:
                record.update(status="failed", reason=f"cleanup_evidence_error:{type(exc).__name__}:{exc}")
                try:
                    requested_device = read(record["task_path"]).get("device", "npu:0")
                except Exception:
                    requested_device = record.get("device", "npu:0")
                for attempt in record.get("attempts", [])[-1:] if requested_device.startswith("npu") and not record.get("shared_group") else []:
                    attempt["device_seconds"] = max(attempt.get("device_seconds", 0),
                        time.time()-attempt["started_unix"])
                    attempt.update(accounting_estimated=True, accounting_reason="unreadable_current_attempt_accounting")
                record["device_seconds"] = sum(v.get("device_seconds", 0) for v in record.get("attempts", []))
        self.state["remaining_descendants"] = list(members().values())
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        self.state.update(status="stopped_budget" if self.stop_reason and any(
            x in self.stop_reason for x in ("deadline", "limit", "storage", "budget")) else "a_blocked_needs_repair",
            blocking_reason=self.stop_reason or "controller_exception", stage_b_gate="closed")
        try:
            self._save()
        except Exception as error:
            self.state["cleanup_save_error"] = f"{type(error).__name__}: {error}"

    def finish(self, status):
        self.state.update(status=status, finished_unix=time.time())
        self._save()

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = read(args.plan)
    supervisor = Supervisor(plan["config"], args.output)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    try:
        supervisor.run(plan.get("stage", "preflight"), plan["tasks"])
        supervisor.finish("completed")
        return 0
    except Exception as exc:
        supervisor.stop_reason = supervisor.stop_reason or f"{type(exc).__name__}: {exc}"
        supervisor.stop_all()
        return 1
    finally:
        supervisor.close()


if __name__ == "__main__":
    raise SystemExit(main())
