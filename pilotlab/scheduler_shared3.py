"""Campaign integration for bounded CPU groups and shared NPU leases."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .io import sha256
from .scheduler import proc_identity
from .scheduler_dynamic3 import DynamicAdmission
from .scheduler3 import (Blocked, Supervisor, evaluate_outcome, evaluation_bound,
                         freeze_json, live, read)

GIB = 2**30


def finite_time(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def reservations(task):
    rss = task.get("reserved_rss_bytes", 16 * GIB if task.get("purpose") == "capacity_calibration" else 4 * GIB)
    threads = task.get("cpu_threads", 4)
    checkpoint = task.get("checkpoint_reservation_bytes", 512 * 2**20)
    if any(type(value) is not int or value <= 0 for value in (rss, threads, checkpoint)):
        raise Blocked("invalid_shared_worker_resource_reservation")
    return {"reserved_rss_bytes": rss, "cpu_threads": threads,
            "checkpoint_reservation_bytes": max(16 * 2**20, checkpoint)}


class SharedSupervisor(DynamicAdmission, Supervisor):
    def worker_reservations(self, task):
        resource = reservations(task)
        if self.config.get("resource_dispatch") and task.get("device") == "cpu" and "status" not in task:
            # Accelerator profiles report host RSS separately from tensors in
            # HBM. Both allocations live in host memory on a CPU trajectory.
            resource["reserved_rss_bytes"] += task.get("reserved_hbm_bytes", 8 * GIB)
        return resource

    def group_observation(self, group_id, *, fresh=False):
        cache = getattr(self, "_group_observation_cache", None)
        if not fresh and cache is not None and group_id in cache:
            return cache[group_id]
        entry = self.state.get("shared_groups", {})[group_id]
        terminal_cache = getattr(self, "_terminal_group_cache", {})
        previous = terminal_cache.get(group_id)
        if (not fresh and entry.get("release_verified") and previous is not None
                and time.monotonic()-previous[0] < self.config.get("terminal_group_recheck_seconds", 30.)):
            if cache is not None:
                cache[group_id] = previous[1]
            return previous[1]
        observation = read(Path(entry["path"]) / "group.json", {})
        if not isinstance(observation, dict):
            raise Blocked("shared_group_record_is_not_an_object")
        if observation:
            if (not isinstance(observation, dict) or observation.get("job_id") != entry["job_id"]
                    or observation.get("manifest_sha256") != entry["manifest_sha256"]):
                raise Blocked("shared_group_identity_mismatch")
            if observation.get("backend") not in {"cpu", "npu"} or entry.get("backend", observation["backend"]) != observation["backend"]:
                raise Blocked("shared_group_backend_mismatch")
            acquired, released = observation.get("allocation_acquired_unix"), observation.get("allocation_released_unix")
            seconds = observation.get("physical_device_seconds", 0.)
            if (not finite_time(seconds) or acquired is not None and not finite_time(acquired)
                    or released is not None and (not finite_time(released) or acquired is None or released < acquired)
                    or not isinstance(observation.get("workers"), dict)
                    or any(not isinstance(worker, dict) for worker in observation.get("workers", {}).values())):
                raise Blocked("shared_group_accounting_invalid")
            if entry.get("dynamic_admission"):
                self.dynamic_observation(entry, observation)
                if observation.get("admission", {}).get("closed"):
                    entry.setdefault("close_requested", observation["admission"].get("close_reason") or "owner_closed_admission")
            elif entry.get("task_ids") is not None and set(observation["workers"]) != set(entry["task_ids"]):
                raise Blocked("shared_group_worker_set_mismatch")
            entry.update(status=observation["status"],
                backend=observation["backend"], allocation_unix=acquired, release_unix=released,
                physical_id=observation.get("physical_id"),
                device_seconds=max(entry.get("device_seconds", 0.), seconds),
                worker_wall_seconds=observation.get("worker_wall_seconds", 0.),
                peak_parallel_workers=observation.get("peak_parallel_workers", 0), observed_group=True)
            owner = observation.get("owner")
            if owner and not entry.get("process"):
                if entry.get("provisional_pid", owner["pid"]) != owner["pid"]:
                    raise Blocked("shared_group_owner_identity_mismatch")
                entry["process"] = owner
                self.state["all_seen_processes"][f'{owner["pid"]}:{owner["start_ticks"]}'] = owner
            self.state["all_seen_processes"].update(observation.get("all_seen_processes", {}))
            for item in observation.get("workers", {}).values():
                identity = item.get("process")
                if identity:
                    self.state["all_seen_processes"][f'{identity["pid"]}:{identity["start_ticks"]}'] = identity
        if cache is not None:
            cache[group_id] = observation
        if entry.get("release_verified") and observation:
            if not hasattr(self, "_terminal_group_cache"):
                self._terminal_group_cache = {}
            self._terminal_group_cache[group_id] = time.monotonic(), observation
        return observation

    def observe_owner(self, group_id):
        group = self.group_observation(group_id)
        if self.owner_running(group_id):
            return group, True
        self.reap_owner(group_id)
        if not self.state["shared_groups"][group_id].get("release_verified"):
            # The owner can publish its last worker/lease receipts and exit
            # after the tick's read. Finalize only from a post-exit read.
            group = self.group_observation(group_id, fresh=True)
        return group, False

    def group_release_error(self, group_id, group, *, require_completed=True):
        entry = self.state["shared_groups"][group_id]
        statuses = {"completed"} if require_completed else {"completed", "failed"}
        if not group or group.get("status") not in statuses:
            return "shared_group_not_completed:" + str(group.get("reason", group.get("status")))
        if group.get("remaining_descendants") != [] or live(entry.get("process")):
            return "shared_group_release_not_verified"
        expected_code = 0 if group["status"] == "completed" else 1
        if entry.get("owner_return_code", expected_code) != expected_code:
            return "shared_group_owner_failed"
        if entry.get("observation_error") or group.get("queue_release_error") or group.get("cleanup_observation_error"):
            return "shared_group_cleanup_or_queue_error"
        manifest_path = entry.get("manifest_path")
        if manifest_path and sha256(manifest_path) != entry["manifest_sha256"]:
            return "shared_group_manifest_changed"
        if group.get("backend") == "npu":
            queue = read(Path(entry["path"]) / "queue.json", {})
            from .lifecycle3 import empty_dispatch_ticket_closed
            cancelled_ticket = empty_dispatch_ticket_closed(entry, group, queue)
            if (not (group.get("allocation_released_unix") or cancelled_ticket) or not group.get("queue_released_unix")
                    or queue.get("job_id") != entry["job_id"]
                    or queue.get("status") != group["status"] or queue.get("return_code") != expected_code):
                return "shared_group_lease_release_missing"
        return None

    def verify_completed(self, task, record):
        if record.get("imported_completion"):
            return super().verify_completed(task, record)
        if not record.get("shared_group"):
            return super().verify_completed(task, record)
        group_id = record["shared_group"]
        group, running = self.observe_owner(group_id)
        if self.state["shared_groups"][group_id].get("dynamic_admission"):
            if group.get("status") == "failed" or group.get("reason"):
                raise Blocked("dynamic_owner_failed_after_worker_completion")
            checked = self.dynamic_worker_result(task, record, group)
            if (checked is None or checked[0] != "completed" or str(checked[3]) != record.get("result_path")
                    or sha256(checked[3]) != record.get("result_sha256")):
                raise Blocked("completed_dynamic_worker_evidence_changed")
            if not running:
                reason = self.group_release_error(group_id, group)
                if reason:
                    raise Blocked(reason)
            return
        reason = self.group_release_error(group_id, group)
        result_path = Path(record["result_path"])
        if reason:
            raise Blocked(reason)
        if not result_path.is_file() or sha256(result_path) != record.get("result_sha256"):
            raise Blocked("completed_result_missing_or_changed")
        worker = group.get("workers", {}).get(task["id"], {})
        result = read(result_path)
        status, reason = evaluate_outcome(task, result, worker.get("return_code"))
        if (worker.get("status") != "completed" or worker.get("result_sha256") != record["result_sha256"]
                or result.get("task_sha256") != sha256(record["task_path"])
                or Path(worker.get("output", "")).resolve() != result_path.parent.resolve()):
            raise Blocked("completed_shared_worker_evidence_changed")
        if status != "completed":
            raise Blocked(f"completed_result_invalid:{reason}")
        from .resume3 import verify_adapter_result
        verify_adapter_result(self.config, task, record, result)

    def device_seconds(self):
        groups = self.state.get("shared_groups", {})
        for key, entry in groups.items():
            if (entry.get("launch_state") == "prepared"
                    and entry.get("launch_controller") == self.state.get("controller_identity")):
                continue  # This controller has not attempted Popen yet.
            try:
                observation = self.group_observation(key)
            except (OSError, ValueError, TypeError, KeyError, Blocked) as error:
                entry["observation_error"] = f"{type(error).__name__}: {error}"
                self.stop_reason = self.stop_reason or "shared_group_observation_failed"
                observation = {}
            if not observation and entry.get("observed_group"):
                entry["observation_error"] = entry.get("observation_error", "shared_group_record_missing")
                self.stop_reason = self.stop_reason or "shared_group_observation_failed"
            if entry.get("backend", observation.get("backend")) != "npu":
                continue
            acquired = entry.get("allocation_unix")
            queue = {}
            if acquired is None:
                try:
                    queue = read(Path(entry["path"]) / "queue.json", {})
                    if not isinstance(queue, dict):
                        raise ValueError("queue record is not an object")
                    if queue.get("job_id") == entry["job_id"]:
                        candidate = queue.get("accounting_started_unix")
                        if finite_time(candidate):
                            acquired = candidate
                            entry["allocation_unix"] = acquired
                except (OSError, ValueError, TypeError):
                    pass
            unknown = (entry.get("observation_error") or not observation and not live(entry.get("process"))
                       or observation and observation.get("status") == "failed" and not entry.get("release_unix"))
            if acquired is None and unknown and not entry.get("launch_aborted_before_spawn"):
                origins = [entry.get("started_unix")]
                origins += [attempt.get("started_unix") for record in self.state["tasks"].values()
                            for attempt in record.get("attempts", []) if attempt.get("shared_group") == key]
                origins.append(self.state.get("started_unix"))
                # No valid origin is never interpreted as zero device cost.
                # A complete absence of timestamps uses the epoch as an
                # explicit conservative upper bound and closes admission.
                acquired = next((value for value in origins if finite_time(value)), 0.)
                entry["accounting_estimated"] = True
                entry["accounting_reason"] = "allocation origin unavailable; charge from group launch through verified cleanup"
                self.stop_reason = self.stop_reason or "shared_group_accounting_origin_unverified"
            if acquired is not None and (not entry.get("release_unix") or entry.get("observation_error")
                    or observation.get("queue_release_error")):
                end = entry.get("cleanup_verified_unix", time.time())
                entry["device_seconds"] = max(entry.get("device_seconds", 0.), max(0., end - acquired))
                if not observation or observation.get("queue_release_error") or not live(entry.get("process")):
                    entry["accounting_estimated"] = True
                    entry["accounting_reason"] = "lease_release_missing_or_unverified; charge through verified descendant cleanup"
        return (sum(g.get("device_seconds", 0.) for g in groups.values())
                + sum(r.get("device_seconds", 0.) for r in self.state["tasks"].values()
                      if not r.get("shared_group")))

    def stop_all(self):
        # Dispatch caches are invalid across process termination. Reading a
        # cached running owner here loses its final result/cleanup receipts.
        self._group_observation_cache = None
        self._terminal_group_cache = {}
        self._stopping = True
        try:
            super().stop_all()
        finally:
            try:
                remaining = self.owned_members()
                for group_id in list(self.processes):
                    if self.processes[group_id].poll() is not None:
                        self.reap_owner(group_id)
                if not remaining:
                    for entry in self.state.get("shared_groups", {}).values():
                        if not live(entry.get("process")):
                            entry.setdefault("cleanup_verified_unix", time.time())
                self.state["actual_npu_concurrency"] = self.actual_npu_concurrency([])
                if not remaining:
                    self.state["actual_npu_worker_concurrency"] = 0
                    self.state["actual_cpu_worker_concurrency"] = 0
                self._save()
            except Exception as error:
                self.state["cleanup_save_error"] = f"{type(error).__name__}: {error}"

    def preparation_device_seconds(self):
        return (sum(g.get("device_seconds", 0.) for g in self.state.get("shared_groups", {}).values()
                    if g.get("preparation"))
                + sum(r.get("device_seconds", 0.) for r in self.state["tasks"].values()
                      if not r.get("shared_group") and (self.config.get("preparation") or r.get("preparation"))))

    def actual_npu_concurrency(self, records):
        physical = set()
        logical = 0
        cpu_logical = 0
        for group_id in self.state.get("shared_groups", {}):
            group = self.group_observation(group_id)
            if group.get("backend") == "npu" and group.get("allocation_acquired_unix") is not None and group.get("allocation_released_unix") is None:
                physical.add(group["physical_id"])
                logical += sum(w.get("status") == "running" for w in group["workers"].values())
            elif group.get("backend") == "cpu":
                cpu_logical += sum(w.get("status") == "running" for w in group.get("workers", {}).values())
        self.state["actual_npu_worker_concurrency"] = logical
        self.state["peak_npu_worker_concurrency"] = max(self.state.get("peak_npu_worker_concurrency", 0), logical)
        self.state["actual_cpu_worker_concurrency"] = cpu_logical
        self.state["peak_cpu_worker_concurrency"] = max(self.state.get("peak_cpu_worker_concurrency", 0), cpu_logical)
        return len(physical)

    def start_ready_tasks(self, tasks, slots):
        if self.dynamic_enabled():
            return self.start_dynamic_ready_tasks(tasks, slots)
        if slots <= 0:
            return
        pending = [t for t in tasks if self.state["tasks"][t["id"]]["status"] == "pending"]
        active_records = [r for r in self.state["tasks"].values() if r["status"] == "running"]
        active_reservations = [reservations(r) for r in active_records]
        write_limit = self.available_checkpoint_writes()
        reserved = sum(r["checkpoint_reservation_bytes"] for r in active_reservations) if write_limit is not None else 0
        reserved_rss = sum(r["reserved_rss_bytes"] for r in active_reservations)
        reserved_threads = sum(r["cpu_threads"] for r in active_reservations)
        cpu_reserved_threads = sum(reservations(r)["cpu_threads"] for r in active_records if r.get("device") == "cpu")
        group_limit = self.config.get("max_npu_devices")
        if group_limit is not None and (type(group_limit) is not int or not 1 <= group_limit <= 16):
            raise Blocked("invalid_max_npu_devices")
        active_npu_groups = {r["shared_group"] for r in active_records
                             if r.get("device", "npu:0").startswith("npu") and r.get("shared_group")}
        npu_slots = max(0, group_limit-len(active_npu_groups)) if group_limit is not None else None
        active_cpu = sum(r.get("device") == "cpu" for r in active_records)
        cpu_slots = max(0, self.config.get("max_cpu_workers", 32) - active_cpu)
        # Reserve the bounded CPU pool first so queued device groups cannot
        # occupy every logical slot while host cores sit idle.
        for backend in ("cpu", "npu"):
            if backend == "npu" and npu_slots == 0:
                continue
            ready = [t for t in pending if ("npu" if t.get("device", "npu:0").startswith("npu") else "cpu") == backend]
            if backend == "cpu":
                ready = ready[:cpu_slots]
            ready = ready[:slots]
            if not ready:
                continue
            target = max(1, self.config.get("target_npu_devices", 7)) if backend == "npu" else max(1, self.config.get("max_cpu_workers", 32))
            if backend == "npu" and npu_slots is not None:
                target = min(target, npu_slots)
            cap = self.config.get("npu_workers_per_device", 4) if backend == "npu" else 1
            size = min(cap, max(1, math.ceil(len(ready) / target)))
            # New capacity measurements have no trustworthy HBM reservation yet.
            # Measure them exclusively before admitting their later trajectories.
            if any(t.get("purpose") == "capacity_calibration" for t in ready):
                size = 1
            while ready and slots and (backend != "npu" or npu_slots is None or npu_slots > 0):
                group = []
                group_hbm = 4 * GIB
                for task in ready[:size]:
                    resources = reservations(task)
                    estimate = resources["checkpoint_reservation_bytes"]
                    reason = None
                    if write_limit is not None and reserved + estimate > write_limit:
                        reason = "checkpoint"
                    elif reserved_rss + resources["reserved_rss_bytes"] > self.config.get("max_rss_bytes", 512 * GIB):
                        reason = "rss"
                    elif reserved_threads + resources["cpu_threads"] > self.config.get("max_cpu_threads", len(os.sched_getaffinity(0))):
                        reason = "cpu_thread"
                    elif backend == "cpu" and cpu_reserved_threads + resources["cpu_threads"] > self.config.get("max_cpu_task_threads", 160):
                        reason = "cpu_task_thread"
                    if reason:
                        break
                    hbm = task.get("reserved_hbm_bytes", 8 * GIB)
                    if backend == "npu" and group and group_hbm + hbm > self.config.get("shared_max_hbm_bytes", 48 * GIB):
                        break
                    group.append(task)
                    if write_limit is not None:
                        reserved += estimate
                    reserved_rss += resources["reserved_rss_bytes"]
                    reserved_threads += resources["cpu_threads"]
                    if backend == "cpu":
                        cpu_reserved_threads += resources["cpu_threads"]
                    group_hbm += hbm
                if not group:
                    if not active_records and not any(r["status"] == "running" for r in self.state["tasks"].values()):
                        raise Blocked(f"single_task_{reason}_reservation_exceeds_limit")
                    break
                self.start_group(group, backend)
                if backend == "npu" and npu_slots is not None:
                    npu_slots -= 1
                slots -= len(group)
                ready = ready[len(group):]

    def worker_plans(self, tasks, backend):
        workers, plans = [], []
        originals = {task["id"]: copy.deepcopy(self.state["tasks"][task["id"]]) for task in tasks}
        worker_source = self.config.get("worker_source", self.config["source"])
        worker_foundation = self.config.get("worker_foundation", self.config["foundation"])
        for task in tasks:
            record = self.state["tasks"][task["id"]]
            if record["status"] != "pending":
                raise Blocked("shared_group_task_is_not_pending")
            attempt = self.output / "tasks" / task["id"] / f'attempt-{len(record["attempts"])}'
            task_path = attempt.parent / "task.json"
            command = [self.config["python"], "-u", "-m", self.config.get("worker_module", "pilotlab.worker3"),
                "--task", str(task_path), "--output", str(attempt), "--data", self.config["data"],
                "--device", task.get("device", "npu:0"), "--deadline-unix", str(self.config["deadline_unix"])]
            if record.get("retry_checkpoint"):
                expected = record.get("retry_checkpoint_sha256")
                if expected is None or sha256(record["retry_checkpoint"]) != expected:
                    raise Blocked("retry_checkpoint_changed_before_launch")
                command += ["--resume-checkpoint", record["retry_checkpoint"]]
            if record.get("resume_receipt"):
                from .resume3 import bound
                adapter = self.config["resume_adapter"]
                if sha256(adapter["path"]) != adapter["sha256"]:
                    raise Blocked("resume_adapter_changed_before_launch")
                receipt = bound(record["resume_receipt"])
                if (receipt["task_id"] != task["id"] or receipt["checkpoint"]["path"] != record.get("retry_checkpoint")
                        or receipt["checkpoint"]["sha256"] != record.get("retry_checkpoint_sha256")):
                    raise Blocked("resume_checkpoint_receipt_mismatch")
                command = [self.config["python"], "-u", adapter["path"], *command[4:],
                    "--receipt", record["resume_receipt"]["path"], "--receipt-sha256", record["resume_receipt"]["sha256"],
                    "--adapter-sha256", adapter["sha256"]]
            elif task.get("stage_a_migration") and not record.get("retry_checkpoint"):
                from .resume3 import bound
                ref = task["stage_a_migration"]
                receipt = bound(ref)
                adapter = self.config["resume_adapter"]
                if receipt["task_id"] != task["id"] or sha256(adapter["path"]) != adapter["sha256"]:
                    raise Blocked("stage_a_migration_command_binding_changed")
                command = [self.config["python"], "-u", adapter["path"], *command[4:],
                    "--resume-checkpoint", receipt["checkpoint"]["path"], "--receipt", ref["path"],
                    "--receipt-sha256", ref["sha256"], "--adapter-sha256", adapter["sha256"]]
            resources = self.worker_reservations(task)
            threads = resources["cpu_threads"]
            worker = {"id": task["id"], "argv": command, "cwd": worker_source, "output": str(attempt),
                "cpu_threads": threads, "reserved_rss_bytes": resources["reserved_rss_bytes"],
                "phase": task.get("phase"), "purpose": task.get("purpose"),
                "reserved_hbm_bytes": task.get("reserved_hbm_bytes", 8 * GIB),
                "env": {"PYTHONPATH": os.pathsep.join([worker_source, worker_foundation + "/src"]),
                    "PYTHONDONTWRITEBYTECODE": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1"}}
            if task.get("purpose") == "capacity_calibration":
                worker["allowed_failure_classes"] = ["capacity_error"]
            workers.append(worker)
            plans.append((task, record, attempt, task_path, command, resources))
        return workers, plans, originals

    def start_group(self, tasks, backend, *, dispatch_lease=False):
        from .shared_npu3 import validate_manifest
        if dispatch_lease and (tasks or backend != "npu" or not self.dynamic_enabled() or not getattr(self, "dispatch_placement", None)):
            raise Blocked("empty_lease_requires_live_dispatch_policy")
        if not (tasks or dispatch_lease) or backend not in {"cpu", "npu"} or any(
                ("npu" if task.get("device", "npu:0").startswith("npu") else "cpu") != backend for task in tasks):
            raise Blocked("shared_group_backend_contract_invalid")
        if backend == "npu" and len(tasks) > 1 and any(
                task.get("purpose") == "capacity_calibration" or task.get("phase") == "preflight" for task in tasks):
            raise Blocked("shared_capacity_probe_requires_exclusive_owner")
        groups = self.state.setdefault("shared_groups", {})
        group_id = f"group-{len(groups):06d}"
        directory = self.output / "groups" / group_id
        job_id = self.config["run_id"] + "-" + group_id
        if len(job_id) > 128:
            job_id = "r3-" + hashlib.sha256(job_id.encode()).hexdigest()
        workers, plans, originals = self.worker_plans(tasks, backend)
        shared_limit = self.config.get("shared_max_hbm_bytes", 48 * GIB)
        exclusive_limit = self.config.get("exclusive_max_hbm_bytes")
        exclusive = backend == "npu" and exclusive_limit is not None and any(
            w["reserved_hbm_bytes"] + 4 * GIB > shared_limit for w in workers)
        if exclusive and len(workers) != 1:
            raise Blocked("large_hbm_worker_requires_exclusive_owner")
        dynamic = (self.dynamic_enabled() and backend == "npu" and not exclusive and all(
            task.get("phase") != "preflight" and task.get("purpose") != "capacity_calibration" for task in tasks))
        ledger = read(self.config["budget_ledger"])
        manifest = {"schema": "pilot-r3-shared-group-v1", "job_id": job_id, "project": "tide-learning-lab-r3",
            "backend": backend, "sharing_authorization": "round3-user-approved", "priority": 0,
            "external_occupant_policy": self.config.get("external_occupant_policy", "stop"),
            "deadline_unix": self.config["deadline_unix"], "max_device_seconds": max(1., ledger["max_device_seconds"] - ledger["device_seconds"]),
            "max_parallel": len(workers), "max_cpu_threads": sum(w["cpu_threads"] for w in workers),
            "max_rss_bytes": min(self.config.get("max_rss_bytes", 512 * GIB),
                                 GIB + sum(w["reserved_rss_bytes"] for w in workers)),
            "max_hbm_bytes": exclusive_limit if exclusive else shared_limit,
            "workers": workers, "resource_interval_seconds": 5., "stop_grace_seconds": 45.}
        if backend == "npu" and exclusive_limit is not None:
            manifest.update(exclusive_hbm=exclusive, release_on_idle_hbm_wait=True)
        if backend == "npu" and self.config.get("temporary_npu"):
            policy = self.config["temporary_npu"]
            manifest["candidate_ids"] = policy["candidate_ids"]
            if dynamic:
                manifest["temporary_npu"] = policy
        if backend == "cpu" and self.config.get("resource_dispatch"):
            manifest["max_rss_bytes"] = self.config["max_rss_bytes"]
        if dynamic:
            cap = min(16, self.config.get("npu_workers_per_device", 16))
            owner_threads = self.config.get("dynamic_owner_max_cpu_threads", 64)
            host_threads = self.config.get("max_cpu_threads", 320)
            if host_threads is not None:
                owner_threads = min(owner_threads, host_threads)
            manifest.update(max_parallel=cap, max_cpu_threads=owner_threads,
                            max_rss_bytes=self.config.get("max_rss_bytes", 1024*GIB))
            total = self.config.get("dynamic_owner_max_total_workers", 32768)
            manifest["dynamic_admission"] = {"inbox_dir": str(directory/"inbox"),
                "max_total_workers": total, "max_commands": self.config.get("dynamic_owner_max_commands", total+1),
                "controller_identity": copy.deepcopy(self.state["controller_identity"]),
                "idle_timeout_seconds": self.config.get("dynamic_owner_idle_seconds", 30.),
                "stable_window_seconds": self.config.get("dynamic_stable_window_seconds", 30.),
                "min_gain_ratio": self.config.get("dynamic_min_gain_ratio", .05),
                "heartbeat_max_age_seconds": self.config.get("dynamic_heartbeat_max_age_seconds", 15.)}
            if self.config.get("resource_dispatch"):
                manifest["dynamic_admission"]["resource_dispatch"] = True
        if dispatch_lease:
            manifest["dispatch_lease"] = True
        if not dynamic:
            validate_manifest(manifest)
        # Reserve the entire group before creating attempts. A later model's
        # exhausted budget must not leave earlier models partially reserved.
        try:
            for task, record, *_ in plans:
                self.reserve_task(task, record)
        except BaseException:
            for task, record, *_ in plans:
                record.clear()
                record.update(originals[task["id"]])
            raise
        path = directory / "manifest.json"
        command = [self.config["python"], "-u", "-m", "pilotlab.shared_npu3", "--manifest", str(path), "--output", str(directory)]
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "PYTHONPATH": os.pathsep.join([self.config["source"], self.config["foundation"] + "/src"])}
        started = time.time()
        entry = groups[group_id] = {"path": str(directory), "job_id": job_id, "manifest_path": str(path),
            "process": None, "status": "starting", "device_seconds": 0., "started_unix": started,
            "launch_state": "prepared", "launch_controller": self.state.get("controller_identity"),
            "backend": backend, "task_ids": [t["id"] for t in tasks],
            "dynamic_admission": dynamic, "dispatch_lease": dispatch_lease,
            "preparation": bool(self.config.get("preparation") or tasks and all(t["phase"] == "preflight" for t in tasks))}
        for task, record, attempt, task_path, worker_command, resources in plans:
            record.update(status="running", reason=None, task_path=str(task_path), shared_group=group_id,
                          process=None, purpose=task.get("purpose"),
                          reserved_hbm_bytes=task.get("reserved_hbm_bytes", 8*GIB), **resources)
            record["attempts"].append({"path": str(attempt), "shared_group": group_id,
                "group_path": str(directory), "command": worker_command, "started_unix": started,
                "status": "starting", "device_seconds": 0.})
        process, log = None, None
        try:
            for task, _, attempt, task_path, *_ in plans:
                attempt.mkdir(parents=True, exist_ok=False)
                freeze_json(task_path, task)
            if dynamic or exclusive and all(w["phase"] != "preflight" and w["purpose"] != "capacity_calibration" for w in workers):
                from .shared_npu3 import worker_spec_digest
                for worker, (_, _, _, task_path, *_) in zip(workers, plans):
                    worker.update(task_path=str(task_path), task_sha256=sha256(task_path))
                entry["worker_spec_sha256"] = {worker["id"]: worker_spec_digest(worker) for worker in workers}
                validate_manifest(manifest)
            freeze_json(path, manifest)
            entry["manifest_sha256"] = sha256(path)
            self._save()  # Restart sees a complete, reserved group before spawn.
            log = (directory / "stdout.log").open("a", buffering=1)
            entry["launch_state"] = "spawning"
            process = subprocess.Popen(command, cwd=self.config["source"], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            # Register every owner and member before any fallible /proc read.
            self.processes[group_id], self.handles[group_id] = process, log
            entry.update(provisional_pid=process.pid, launch_state="spawned")
            for _, record, *_ in plans:
                record["provisional_pid"] = process.pid
                record["attempts"][-1].update(provisional_pid=process.pid, status="running")
            identity = proc_identity(process.pid, strict=True)
            if identity is None:
                raise Blocked("shared_owner_exited_before_identity_capture")
            entry["process"] = identity
            self.state["all_seen_processes"][f'{identity["pid"]}:{identity["start_ticks"]}'] = identity
            for _, record, *_ in plans:
                record["process"] = identity
                record["attempts"][-1]["process"] = identity
            self._save()
        except BaseException as error:
            entry["launch_error"] = f"{type(error).__name__}: {error}"
            if process is None:
                if log:
                    log.close()
                entry.update(status="failed", launch_state="aborted", launch_aborted_before_spawn=True,
                             cleanup_verified_unix=time.time())
                for task, record, *_ in plans:
                    for field in ("preparation_reserved_tokens", "evaluation_reserved_tokens", "evaluation_budget_key"):
                        if field in originals[task["id"]]:
                            record[field] = originals[task["id"]][field]
                        else:
                            record.pop(field, None)
                    record.update(status="failed", reason="shared_group_not_spawned",
                                  observation_error=entry["launch_error"])
                    record["attempts"][-1].update(status="not_started", reason=entry["launch_error"], finished_unix=time.time())
            else:
                for _, record, *_ in plans:
                    record["observation_error"] = entry["launch_error"]
            try:
                self._save()
            except Exception:
                pass  # The caller still has registered handles for cleanup.
            raise
        return group_id

    def poll_hbm_deferral(self, task, record, group, worker, owner_running):
        if worker.get("status") != "not_started" or worker.get("reason") != "hbm_admission_deferred":
            return None
        group_id = record["shared_group"]
        if owner_running:
            return False
        error = self.group_release_error(group_id, group)
        if error:
            raise Blocked(error)
        entry = self.state["shared_groups"][group_id]
        attempt = record["attempts"][-1]
        path = Path(attempt["path"])
        if (not group.get("capacity_deferral") or not read(entry["manifest_path"]).get("release_on_idle_hbm_wait")
                or any(worker.get(k) is not None for k in ("pid", "process", "started_unix", "return_code"))
                or Path(worker["output"]).resolve() != path.resolve() or any(path.iterdir())):
            raise Blocked("hbm_deferred_worker_has_execution_or_missing_release_evidence")
        entry["release_verified"] = True
        count = record.get("hbm_deferrals", 0) + 1
        attempt.update(status="not_started", reason="hbm_admission_deferred", finished_unix=time.time(),
                       release_unix=group["allocation_released_unix"])
        record.update(status="pending", reason="hbm_admission_deferred", hbm_deferrals=count,
                      hbm_retry_after_unix=time.time() + min(900, 60 * 2 ** min(4, count - 1)))
        self._save()
        return True

    def _poll(self, task, record):
        group_id = record.get("shared_group")
        if not group_id:
            return super()._poll(task, record)
        entry = self.state["shared_groups"][group_id]
        if entry.get("dynamic_admission"):
            return self.poll_dynamic_worker(task, record)
        try:
            group, running = self.observe_owner(group_id)
        except (OSError, ValueError, TypeError, KeyError, Blocked) as error:
            group = {}
            running = self.owner_running(group_id)
            if not running:
                self.reap_owner(group_id)
            record["observation_error"] = f"shared_group_observation_failed:{type(error).__name__}:{error}"
            entry["observation_error"] = record["observation_error"]
            self.stop_reason = self.stop_reason or "shared_group_observation_failed"
        worker = group.get("workers", {}).get(task["id"], {})
        attempt = record["attempts"][-1]
        directory = Path(attempt["path"])
        if worker.get("process"):
            attempt["worker_process"] = worker["process"]
        attempt["worker_wall_seconds"] = worker.get("wall_seconds", 0.)
        attempt["allocation_unix"] = group.get("allocation_acquired_unix")
        attempt["release_unix"] = group.get("allocation_released_unix")
        physical_id = group.get("physical_id")
        record["queue"] = {"status": "running" if group.get("allocation_acquired_unix") and not group.get("allocation_released_unix") else
            "completed" if group.get("status") == "completed" else "queued" if group.get("status") in {"queued", "preparing"} else group.get("status", "queued"),
            "shared_group": group_id, "physical_ids": [] if physical_id is None else [physical_id],
            "group_path": entry["path"], "accounting_scope": "physical lease charged once in shared_groups"}
        try:
            heartbeat = read(directory / "heartbeat.json")
        except (OSError, ValueError, TypeError) as error:
            heartbeat = None
            record["observation_error"] = f"shared_worker_heartbeat_unreadable:{type(error).__name__}:{error}"
            self.stop_reason = self.stop_reason or "shared_worker_observation_failed"
        if heartbeat:
            record["heartbeat"] = heartbeat
        if running:
            return False
        deferred = self.poll_hbm_deferral(task, record, group, worker, running)
        if deferred is not None:
            return deferred
        result_path = directory / "result.json"
        try:
            result = read(result_path)
            if result is not None and not isinstance(result, dict):
                raise ValueError("worker result is not an object")
        except (OSError, ValueError, TypeError) as error:
            result = None
            record["observation_error"] = f"shared_worker_result_unreadable:{type(error).__name__}:{error}"
        code = worker.get("return_code")
        status, reason = evaluate_outcome(task, result, code)
        if worker.get("status") == "not_started" and result is None:
            status, reason = "interrupted", "shared_group_worker_not_started"
        elif worker.get("status") not in {"completed", "allowed_failure", "failed", "terminated"}:
            status, reason = "failed", "shared_worker_terminal_evidence_missing"
        if not group or group.get("remaining_descendants") != []:
            status, reason = "failed", "shared_group_release_not_verified"
        if group.get("backend") == "npu" and group.get("allocation_acquired_unix") and not group.get("allocation_released_unix"):
            status, reason = "failed", "shared_group_lease_release_missing"
        if result is not None and result.get("task_sha256") != sha256(record["task_path"]):
            status, reason = "failed", "worker_task_identity_mismatch"
            record["observation_error"] = reason
        if result is not None and result.get("status") == "completed" and record.get("resume_receipt"):
            try:
                from .resume3 import verify_adapter_result
                verify_adapter_result(self.config, task, record, result)
            except (OSError, ValueError, KeyError, Blocked) as error:
                record["observation_error"] = f"resume_adapter_result_invalid:{error}"
        if result and task.get("evaluation_budget_key") and result.get("evaluation", {}).get("tokens", 0) > evaluation_bound(task):
            status, reason = "failed", "worker_evaluation_exceeds_declared_bound"
            record["observation_error"] = reason
        if result and worker.get("result_sha256") != sha256(result_path):
            status, reason = "failed", "shared_worker_result_digest_mismatch"
            record["observation_error"] = reason
        try:
            cleanup_error = self.group_release_error(group_id, group, require_completed=False)
            group_error = self.group_release_error(group_id, group)
        except (OSError, ValueError, TypeError, KeyError) as error:
            cleanup_error = group_error = f"shared_group_release_unreadable:{type(error).__name__}:{error}"
        retryable_group_failure = (not cleanup_error and group.get("status") == "failed"
            and group.get("reason") == f"worker_process_failed:{task['id']}:{code}"
            and isinstance(result, dict) and result.get("failure_class") == "infrastructure_transient")
        if group_error:
            status, reason = "failed", group_error
            if not retryable_group_failure:
                record["observation_error"] = reason
        if record.get("observation_error"):
            status, reason = "failed", record["observation_error"]
        if cleanup_error is None:
            entry["release_verified"] = True
        attempt.update(status=status, return_code=code, finished_unix=time.time(), reason=reason)
        record.update(status=status, reason=reason, result_path=str(result_path),
            result_sha256=sha256(result_path) if result_path.exists() else None, device_seconds=0.)
        if retryable_group_failure:
            self.retry_transient(task, record, result)
        self._save()
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = read(args.plan)
    supervisor = SharedSupervisor(plan["config"], args.output)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    try:
        supervisor.run(plan.get("stage", "preflight"), plan["tasks"], allowed_failures=plan.get("allowed_failures", ()))
        supervisor.finish("completed")
        return 0
    except BaseException as error:
        supervisor.stop_reason = supervisor.stop_reason or f"{type(error).__name__}: {error}"
        supervisor.stop_all()
        return 1
    finally:
        supervisor.close()


if __name__ == "__main__":
    raise SystemExit(main())
