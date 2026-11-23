"""Finite groups of independent workers sharing one cooperatively owned NPU.

The published queue helper is loaded read-only and owns the usual FIFO,
two-sample admission and whole-device locks. Initial admission requires an
empty device. An explicit policy permits external occupants arriving later;
they are observed, never adopted into this group's ownership or cleanup.
CPU groups use the same supervision without importing the NPU helper.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from .io import atomic_json, sha256
from .scheduler import QUEUE_HELPER, child_subreaper, family_members, proc_identity, process_table
from .sharing_policy3 import SharingPolicy

SCHEMA = "pilot-r3-shared-group-v1"
TERMINAL = {"completed", "allowed_failure", "failed", "terminated", "not_started"}
VISIBILITY_KEYS = {"ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}
EXTERNAL_OCCUPANT_POLICIES = {"stop", "continue_running"}
ADMISSION_SCHEMA = "pilot-r3-shared-admission-v1"


class GroupBlocked(RuntimeError):
    pass


def validate_manifest(value):
    if value.get("schema") != SCHEMA or value.get("backend") not in {"cpu", "npu"}:
        raise ValueError("explicit shared-group schema and cpu/npu backend required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value.get("job_id", "")):
        raise ValueError("valid unique job_id required")
    if value.get("external_occupant_policy", "stop") not in EXTERNAL_OCCUPANT_POLICIES:
        raise ValueError("invalid external_occupant_policy")
    for key in ("max_parallel", "max_cpu_threads", "max_rss_bytes"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError(f"positive integer {key} required")
    if not isinstance(value.get("deadline_unix"), (int, float)) or not math.isfinite(value["deadline_unix"]):
        raise ValueError("finite absolute deadline_unix required")
    if value["backend"] == "npu":
        if value.get("sharing_authorization") != "round3-user-approved":
            raise ValueError("explicit current-round shared-device authorization required")
        if type(value.get("max_hbm_bytes")) is not int or value["max_hbm_bytes"] <= 0:
            raise ValueError("positive max_hbm_bytes required")
        if not isinstance(value.get("max_device_seconds"), (int, float)) or not 0 < value["max_device_seconds"] < math.inf:
            raise ValueError("finite positive remaining max_device_seconds required")
    for key, default in (("poll_seconds", 0.25), ("resource_interval_seconds", 1.),
                         ("sharing_sample_seconds", 0.2), ("admission_sample_seconds", 3.),
                         ("stop_grace_seconds", 20.), ("kill_grace_seconds", 10.)):
        if not isinstance(value.get(key, default), (int, float)) or not 0 < value.get(key, default) <= 60:
            raise ValueError(f"{key} must be positive and at most 60 seconds")
    empty_lease = value.get("dispatch_lease") is True and value["backend"] == "npu" and value.get("dynamic_admission")
    if not isinstance(value.get("workers"), list) or not (value["workers"] or empty_lease):
        raise ValueError("a finite nonempty worker list is required")
    if value.get("exclusive_hbm") and (value["backend"] != "npu" or len(value["workers"]) != 1
            or value["max_parallel"] != 1 or "dynamic_admission" in value
            or value["max_hbm_bytes"] > 52 * 2**30):
        raise ValueError("exclusive HBM requires a non-appendable single-worker NPU lease of at most 52 GiB")
    if type(value.get("priority", 0)) is not int:
        raise ValueError("queue priority must be an integer")
    candidates = value.get("candidate_ids")
    if candidates is not None and (not isinstance(candidates, list) or not candidates
            or any(type(item) is not int or item < 0 for item in candidates)):
        raise ValueError("candidate_ids must contain nonnegative physical device ids")
    for key, default in (("admission_max_hbm_mib", 4096), ("admission_max_aicore_percent", 5)):
        if type(value.get(key, default)) is not int or value.get(key, default) < 0:
            raise ValueError("admission thresholds must be nonnegative integers")
    ids, outputs = set(), set()
    permitted_affinity = set(os.sched_getaffinity(0))
    for worker in value["workers"]:
        if not isinstance(worker, dict) or not re.fullmatch(r"[A-Za-z0-9_.-]+", worker.get("id", "")):
            raise ValueError("worker id is invalid")
        if worker["id"] in ids:
            raise ValueError("duplicate worker id")
        ids.add(worker["id"])
        if not isinstance(worker.get("argv"), list) or not worker["argv"] or any(not isinstance(arg, str) for arg in worker["argv"]):
            raise ValueError("worker argv must be an explicit nonempty argument list")
        if "--device" in worker["argv"]:
            index = worker["argv"].index("--device")
            expected_device = "cpu" if value["backend"] == "cpu" else "npu:0"
            if index + 1 >= len(worker["argv"]) or worker["argv"][index + 1] != expected_device:
                raise ValueError("worker device must match the group's logical backend")
        for key in ("output", "cwd"):
            if not isinstance(worker.get(key), str) or not Path(worker[key]).is_absolute():
                raise ValueError(f"worker {key} must be absolute")
        output = str(Path(worker["output"]).resolve())
        if output in outputs:
            raise ValueError("workers require separate output directories")
        outputs.add(output)
        for key in ("cpu_threads", "reserved_rss_bytes"):
            if type(worker.get(key)) is not int or worker[key] <= 0:
                raise ValueError(f"worker {key} must be positive")
        if worker["cpu_threads"] > value["max_cpu_threads"] or worker["reserved_rss_bytes"] > value["max_rss_bytes"]:
            raise ValueError("a single worker exceeds group CPU/RSS budget")
        if value["backend"] == "npu" and (type(worker.get("reserved_hbm_bytes")) is not int
                or not 0 < worker["reserved_hbm_bytes"] <= value["max_hbm_bytes"]):
            raise ValueError("worker must fit an explicit positive HBM reservation")
        environment = worker.get("env", {})
        if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items()):
            raise ValueError("worker env must map strings to strings")
        if VISIBILITY_KEYS.intersection(environment):
            raise ValueError("worker env may not override group device visibility")
        if value["backend"] == "cpu" and environment.get("TORCH_DEVICE_BACKEND_AUTOLOAD", "0") != "0":
            raise ValueError("CPU workers must disable automatic vendor backend loading")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            if key in environment and (not environment[key].isdigit() or not 1 <= int(environment[key]) <= worker["cpu_threads"]):
                raise ValueError("worker environment threads exceed its reservation")
        affinity = worker.get("cpu_affinity", value.get("cpu_affinity"))
        if affinity is not None and (not isinstance(affinity, list) or not affinity
                or any(type(cpu) is not int for cpu in affinity) or not set(affinity) <= permitted_affinity
                or len(set(affinity)) < worker["cpu_threads"]):
            raise ValueError("CPU affinity must be within current allowance and cover compute threads")
        if set(worker.get("allowed_failure_classes", [])) - {"capacity_error"}:
            raise ValueError("only explicit capacity_error may be a nonfatal worker failure")
    if "dynamic_admission" in value:
        dynamic = value["dynamic_admission"]
        if not isinstance(dynamic, dict) or value["backend"] != "npu" or value["max_parallel"] > 16:
            raise ValueError("dynamic admission requires a one-card NPU group with at most 16 workers")
        if not isinstance(dynamic.get("inbox_dir"), str) or not Path(dynamic["inbox_dir"]).is_absolute():
            raise ValueError("dynamic inbox_dir must be absolute")
        for name in ("max_total_workers", "max_commands"):
            if type(dynamic.get(name)) is not int or dynamic[name] <= 0:
                raise ValueError(f"finite positive dynamic {name} required")
        if len(value["workers"]) > dynamic["max_total_workers"]:
            raise ValueError("initial workers exceed the finite dynamic worker limit")
        owner = dynamic.get("controller_identity", {})
        if any(type(owner.get(name)) is not int or owner[name] <= 0 for name in ("pid", "start_ticks")):
            raise ValueError("dynamic admission requires the controller PID/start identity")
        if not 0 < dynamic.get("idle_timeout_seconds", 30.) <= 3600:
            raise ValueError("invalid dynamic idle timeout")
        if not 0 < dynamic.get("integrity_interval_seconds", 30.) <= 3600:
            raise ValueError("invalid dynamic integrity interval")
        SharingPolicy(dynamic, value["max_parallel"])
        for worker in value["workers"]:
            validate_dynamic_worker(value, worker)
    return value


def worker_spec_digest(worker):
    return hashlib.sha256(json.dumps(worker, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_dynamic_worker(manifest, worker):
    if worker.get("allowed_failure_classes") or worker.get("purpose") == "capacity_calibration":
        raise ValueError("capacity calibration is forbidden in a dynamic group")
    path, expected = worker.get("task_path"), worker.get("task_sha256")
    if not isinstance(path, str) or not Path(path).is_absolute() or not expected or sha256(path) != expected:
        raise ValueError("dynamic worker requires an intact absolute task_path/task_sha256")
    task = json.loads(Path(path).read_text())
    if (task.get("id") != worker["id"] or task.get("phase") == "preflight"
            or task.get("purpose") == "capacity_calibration" or task.get("cpu_threads") != worker["cpu_threads"]):
        raise ValueError("dynamic worker task identity, phase or thread reservation mismatch")
    argv = worker["argv"]
    for flag, expected_value in (("--task", path), ("--output", worker["output"]),
                                 ("--device", "npu:0"), ("--deadline-unix", manifest["deadline_unix"])):
        if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
            raise ValueError(f"dynamic worker requires one explicit {flag}")
        observed = argv[argv.index(flag) + 1]
        matches = (float(observed) == expected_value if flag == "--deadline-unix" else
                   Path(observed).resolve() == Path(expected_value).resolve() if flag in {"--task", "--output"}
                   else observed == expected_value)
        if not matches:
            raise ValueError(f"dynamic worker {flag} differs from its bound request")
    return task


def validate_dynamic_paths(workers):
    outputs = [Path(worker["output"]).resolve() for worker in workers]
    inputs = [Path(worker["task_path"]).resolve() for worker in workers]
    input_ancestors = {parent for path in inputs for parent in (path, *path.parents)}
    seen, ancestors = set(), set()
    for output in outputs:
        if output in input_ancestors:
            raise ValueError("dynamic worker output would overwrite a frozen task input")
        if output in ancestors or output in seen or any(parent in seen for parent in output.parents):
            raise ValueError("dynamic workers have overlapping output directories")
        seen.add(output)
        ancestors.update(output.parents)


def read_admission_chain(manifest, manifest_path, group=None):
    """Read and verify a published command chain and an optional acknowledged prefix.

    A controller may have durably published a valid tail not yet observed by the
    owner.  It is not a missing-worker error; only acknowledged workers belong
    in that group snapshot.  Callers decide whether a terminal tail is allowed.
    """
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if json.loads(raw) != manifest:
        raise ValueError("initial manifest content differs from its frozen file")
    validate_manifest(manifest)
    dynamic = manifest.get("dynamic_admission")
    initial = list(manifest["workers"])
    if not dynamic:
        return {"published_workers": initial, "acknowledged_workers": initial, "commands": [],
                "closed": True, "acknowledged_closed": True, "next_sequence": 0,
                "last_sha256": manifest_sha, "manifest_sha256": manifest_sha}
    inbox = Path(dynamic["inbox_dir"])
    validate_dynamic_paths(initial)
    paths = sorted(inbox.glob("*.json")) if inbox.exists() else []
    if len(paths) > dynamic["max_commands"]:
        raise ValueError("dynamic admission command limit exceeded")
    workers, commands = list(initial), []
    ids = {worker["id"] for worker in initial}
    outputs = {str(Path(worker["output"]).resolve()) for worker in initial}
    last, closed = manifest_sha, False
    for sequence, path in enumerate(paths):
        if path.name != f"{sequence:08d}.json":
            raise ValueError("admission commands are not a contiguous canonical sequence")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        command = json.loads(data)
        if (not isinstance(command, dict) or closed or command.get("schema") != ADMISSION_SCHEMA or type(command.get("sequence")) is not int
                or command["sequence"] != sequence or command.get("previous_sha256") != last
                or command.get("manifest_sha256") != manifest_sha or command.get("job_id") != manifest["job_id"]
                or command.get("deadline_unix") != manifest["deadline_unix"]):
            raise ValueError("admission command identity, predecessor or deadline mismatch; commands after close are forbidden")
        kind = command.get("kind")
        additions = command.get("workers", [])
        if kind == "admit":
            if not isinstance(additions, list) or not additions:
                raise ValueError("admit requires a nonempty immutable worker batch")
            validate_manifest({**manifest, "workers": additions})
            for worker in additions:
                output = str(Path(worker["output"]).resolve())
                if worker["id"] in ids or output in outputs:
                    raise ValueError("admission repeats a task identity or overwrites an output")
                ids.add(worker["id"])
                outputs.add(output)
                workers.append(worker)
            if len(workers) > dynamic["max_total_workers"]:
                raise ValueError("admission exceeds the finite total worker limit")
        elif kind == "close" and isinstance(additions, list) and not additions:
            closed = True
        else:
            raise ValueError("admission command kind/workers is invalid")
        commands.append({"sequence": sequence, "path": str(path.resolve()), "sha256": digest,
                         "kind": kind, "worker_ids": [worker["id"] for worker in additions], "command": command})
        last = digest
    validate_dynamic_paths(workers)
    acknowledged = 0
    if group is not None:
        admission = group.get("admission", {})
        acknowledged = admission.get("next_sequence")
        if (group.get("manifest_sha256") != manifest_sha or type(acknowledged) is not int
                or not 0 <= acknowledged <= len(commands)):
            raise ValueError("group admission acknowledgement is not a published prefix")
        expected = [{key: value for key, value in row.items() if key != "command"} for row in commands[:acknowledged]]
        if (admission.get("commands") != expected or admission.get("last_sha256") !=
                (commands[acknowledged-1]["sha256"] if acknowledged else manifest_sha)):
            raise ValueError("acknowledged admission command was changed")
    accepted = initial + [worker for row in commands[:acknowledged] for worker in row["command"].get("workers", [])]
    ack_closed = any(row["kind"] == "close" for row in commands[:acknowledged])
    if group is not None:
        admission = group["admission"]
        autonomous_close = admission.get("closed") and admission.get("close_reason") == "controller_lost"
        if (admission.get("accepted_task_ids") != [worker["id"] for worker in accepted]
                or set(group.get("workers", {})) != {worker["id"] for worker in accepted}
                or bool(admission.get("closed")) != bool(ack_closed or autonomous_close)):
            raise ValueError("acknowledged workers/close differ from the published prefix")
        bindings = {worker["id"]: (None, manifest_sha) for worker in initial}
        for row in commands[:acknowledged]:
            bindings.update({key: (row["sequence"], row["sha256"]) for key in row["worker_ids"]})
        for worker in accepted:
            record = group["workers"][worker["id"]]
            sequence, digest = bindings[worker["id"]]
            if (record.get("worker_spec_sha256") != worker_spec_digest(worker)
                    or record.get("admission_sequence") != sequence or record.get("admission_command_sha256") != digest):
                raise ValueError("group worker spec/admission binding differs from immutable evidence")
    return {"published_workers": workers, "acknowledged_workers": accepted, "commands": commands,
            "closed": closed, "acknowledged_closed": ack_closed, "next_sequence": len(commands),
            "last_sha256": last, "manifest_sha256": manifest_sha}


def load_queue_helper():
    name = "_pilot_r3_readonly_ascend_queue"
    spec = importlib.util.spec_from_file_location(name, QUEUE_HELPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Compile read bytes directly: importing this shared asset must not create
    # or refresh __pycache__ under the published skill directory.
    exec(compile(QUEUE_HELPER.read_bytes(), str(QUEUE_HELPER), "exec"), module.__dict__)
    return module


class QueueLease:
    """Use the shared queue's existing functions and retain their real locks."""
    def __init__(self, manifest, output, stop):
        self.manifest, self.output, self.stop = manifest, output, stop
        self.helper = load_queue_helper()
        self.helper_identity = {"path": str(QUEUE_HELPER), "sha256": sha256(QUEUE_HELPER)}
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        self.queue_dir, self.device_dir = runtime / "ascend-job-queue", runtime / "ascend-device-locks"
        self.job_lock, self.device_locks, self.physical_id = None, (), None
        self.mapping = self.helper.parse_mapping(self.helper.run_npu_smi("info", "-m"))
        self.acquired_unix = self.acquired_monotonic = None
        self.released_unix = self.released_monotonic = None

    def emit(self, status, **details):
        self.helper.emit(status, self.output / "queue.json", job_id=self.manifest["job_id"],
            project="tide-learning-lab-r3-shared", priority=self.manifest.get("priority", 0), npu_count=1, **details)

    def acquire(self):
        candidates = self.manifest.get("candidate_ids")
        candidates = set(candidates) if candidates is not None else None
        if candidates is not None and not candidates <= set(self.mapping.values()):
            raise ValueError("candidate ids absent from the current physical mapping")
        self.job_lock, _ = self.helper.register_job(self.queue_dir, job_id=self.manifest["job_id"],
            project="tide-learning-lab-r3-shared", priority=self.manifest.get("priority", 0),
            npu_count=1, candidate_ids=candidates)
        while not self.stop.is_set():
            if self.manifest.get("dispatch_lease") and self.cancelled():
                return
            if self.manifest.get("dynamic_admission"):
                controller = self.manifest["dynamic_admission"]["controller_identity"]
                current = proc_identity(controller["pid"], strict=True)
                if current is None or current["start_ticks"] != controller["start_ticks"]:
                    raise GroupBlocked("controller_lost_while_queued")
            if time.time() >= self.manifest["deadline_unix"]:
                raise GroupBlocked("deadline_while_queued")
            before_unix, before_mono = time.time(), time.monotonic()
            decision = self.helper.attempt_schedule(queue_dir=self.queue_dir, device_lock_dir=self.device_dir,
                job_id=self.manifest["job_id"], mapping=self.mapping, candidate_ids=candidates, npu_count=1,
                max_hbm_mib=self.manifest.get("admission_max_hbm_mib", 4096),
                max_aicore_percent=self.manifest.get("admission_max_aicore_percent", 5),
                sample_seconds=self.manifest.get("admission_sample_seconds", 3.), stop_event=self.stop)
            if decision.physical_ids:
                self.device_locks = decision.device_locks
                self.physical_id = decision.physical_ids[0]
                # Charge from the successful admission attempt, including its
                # double-sample lock-holding interval. This is conservative.
                self.acquired_unix, self.acquired_monotonic = before_unix, before_mono
                self.emit("running", physical_ids=[self.physical_id], physical_id=self.physical_id,
                          accounting_started_unix=self.acquired_unix,
                          accounting_scope="one device; includes complete successful admission attempt")
                return
            self.emit("queued", reason=decision.reason, queue_position=decision.queue_position,
                      head_job_id=decision.head_job_id)
            self.stop.wait(min(2., max(0., self.manifest["deadline_unix"] - time.time())))
        raise GroupBlocked("cancelled_while_queued")

    def snapshot(self):
        status = self.helper.snapshot(self.mapping)[self.physical_id]
        return {"physical_id": status.physical_id, "health": status.health,
                "used_bytes": status.hbm_used_mib * 2**20, "total_bytes": status.hbm_total_mib * 2**20,
                "process_ids": list(status.process_ids)}

    def pass_fds(self):
        return tuple(handle.fileno() for handle in (*self.device_locks, *([self.job_lock] if self.job_lock else [])))

    def release(self, status):
        # Caller proves descendants gone first. Explicit LOCK_UN must never
        # run while a child still holds an inherited copy of this lock fd.
        if self.device_locks:
            self.helper.release_locks(self.device_locks)
            self.device_locks = ()
            self.released_unix, self.released_monotonic = time.time(), time.monotonic()
        if self.job_lock is not None:
            try:
                self.helper.remove_job(self.queue_dir, self.manifest["job_id"])
            finally:
                self.helper.release_locks([self.job_lock])
                self.job_lock = None
        self.emit("completed" if status == "completed" else "failed", physical_ids=[] if self.physical_id is None else [self.physical_id],
                  return_code=0 if status == "completed" else 1)


def admission_reason(manifest, worker, active, owned, device, *, owner_rss=0):
    """Separate new-worker capacity waits from running-worker stop conditions."""
    coexist = manifest.get("external_occupant_policy", "stop") == "continue_running"
    rss = owner_rss + sum(row.get("rss_bytes", 0) for row in owned.values())
    if rss > manifest["max_rss_bytes"]:
        raise GroupBlocked("group_rss_budget_exceeded")
    if device is not None:
        if device.get("health") != "OK":
            raise GroupBlocked("allocated_device_unhealthy")
        if not coexist and any(pid not in owned for pid in device.get("process_ids", [])):
            raise GroupBlocked("unknown_device_occupant")
        # npu-smi reports total card usage, including other users. Under the
        # coexistence policy this is an admission ceiling, not a kill switch.
        if not coexist and device["used_bytes"] > min(manifest["max_hbm_bytes"], device["total_bytes"]):
            raise GroupBlocked("group_hbm_budget_exceeded")
    if worker is None:
        return None
    if sum(item["cpu_threads"] for item in active) + worker["cpu_threads"] > manifest["max_cpu_threads"]:
        return "cpu_thread_reservations_full"
    projected_rss = max(rss, owner_rss + sum(item["reserved_rss_bytes"] for item in active)) + worker["reserved_rss_bytes"]
    if projected_rss > manifest["max_rss_bytes"]:
        return "rss_reservations_full"
    if device is not None:
        if manifest.get("exclusive_hbm"):
            if active or device["process_ids"]:
                return "exclusive_hbm_requires_empty_card"
            if device["total_bytes"] < 64 * 2**30:
                return "exclusive_hbm_requires_64_gib"
        projected_hbm = max(device["used_bytes"], device.get("baseline_used_bytes", 0)
                            + sum(item["reserved_hbm_bytes"] for item in active)) + worker["reserved_hbm_bytes"]
        if coexist and any(pid not in owned for pid in device.get("process_ids", [])):
            # No per-PID memory attribution is available. Reserve room for
            # existing workers' future allocations above the entire observed
            # card usage; conservative double counting is preferable to
            # treating an external allocation as one of our reservations.
            projected_hbm = max(projected_hbm, device["used_bytes"]
                + sum(item["reserved_hbm_bytes"] for item in active) + worker["reserved_hbm_bytes"])
        if projected_hbm > min(manifest["max_hbm_bytes"], device["total_bytes"]):
            return "hbm_reservations_full"
    return None


def child_setup(parent_pid, affinity):
    """Direct workers die with the owner; all retain the group's device locks."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
    if affinity is not None:
        os.sched_setaffinity(0, affinity)


class GroupRunner:
    def __init__(self, manifest, output, *, manifest_path=None, lease_factory=QueueLease):
        self.manifest = validate_manifest(manifest)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "group.json").exists():
            raise FileExistsError("existing group attempt must be inspected; use a new attempt directory")
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.manifest_sha = sha256(self.manifest_path) if self.manifest_path else None
        self.dynamic = manifest.get("dynamic_admission")
        if self.dynamic and self.manifest_path is None:
            raise ValueError("dynamic admission requires a frozen manifest file")
        self.policy = SharingPolicy(self.dynamic, manifest["max_parallel"]) if self.dynamic else None
        self.last_admission_check = 0.
        self.inbox_stamp = None
        self.stop = threading.Event()
        self.reason = None
        self.processes, self.logs, self.monotonic_starts, self.identities = {}, {}, {}, {}
        self.worker_identities = {}
        self.lease_factory, self.lease = lease_factory, None
        self.device_cache, self.device_cache_time = None, 0.
        self.device_baseline_bytes = None
        self.last_device_conditions = None
        self.started_monotonic = time.monotonic()
        self.state = {"schema": "pilot-r3-shared-group-state-v1", "job_id": manifest["job_id"],
            "backend": manifest["backend"], "status": "preparing", "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "manifest_sha256": self.manifest_sha, "started_unix": time.time(),
            "owner": proc_identity(os.getpid(), strict=True),
            "owner_cgroup": Path("/proc/self/cgroup").read_text().strip(), "all_seen_processes": {},
            "allocation_acquired_unix": None, "allocation_released_unix": None,
            "physical_device_seconds": 0., "worker_wall_seconds": 0., "peak_parallel_workers": 0,
            "peak_observed_npu_processes": 0, "peak_rss_bytes": 0, "peak_hbm_bytes": 0,
            "external_occupant_policy": manifest.get("external_occupant_policy", "stop"),
            "external_contention_observed": False, "peak_external_processes": 0,
            "device_observation_events": 0,
            "workers": {worker["id"]: {"status": "pending", "output": worker["output"], "return_code": None,
                "pid": None, "start_ticks": None, "started_unix": None, "finished_unix": None}
                for worker in manifest["workers"]},
            "accounting_scope": "charge physical_device_seconds exactly once per group; worker_wall_seconds is diagnostic and must not be added to device ledger"}
        self.workers = {worker["id"]: worker for worker in manifest["workers"]}
        for worker in manifest["workers"]:
            self.state["workers"][worker["id"]].update(worker_spec_sha256=worker_spec_digest(worker),
                admission_sequence=None, admission_command_sha256=self.manifest_sha)
        if self.dynamic:
            self.state["admission"] = {"next_sequence": 0, "last_sha256": self.manifest_sha,
                "closed": False, "close_reason": None, "commands": [],
                "accepted_task_ids": list(self.workers), "idle_since_unix": None}
            self.state["lifecycle_memory_observations"] = {"log": str(self.output / "resource-samples.jsonl"),
                "samples": 0, "whole_card_high_water_by_worker_states": {}, "final_result_device_memory": {},
                "scope": "whole-card HBM correlated with all active heartbeat states; mixed workers/external occupants cannot be apportioned; final result allocator peaks retain their original scope"}
            read_admission_chain(manifest, self.manifest_path, self.state)
        self.save()

    def save(self):
        self.state["updated_unix"] = time.time()
        if self.lease and self.lease.acquired_monotonic is not None and self.state["allocation_released_unix"] is None:
            self.state["physical_device_seconds"] = max(0., time.monotonic() - self.lease.acquired_monotonic)
        self.state["worker_wall_seconds"] = sum(row.get("wall_seconds", 0.) for row in self.state["workers"].values())
        if self.policy:
            self.state["policy"] = self.state["sharing_policy"] = self.policy.snapshot()
        atomic_json(self.output / "group.json", self.state)

    def refresh_admissions(self):
        if not self.dynamic:
            return
        admission = self.state["admission"]
        next_path = Path(self.dynamic["inbox_dir"]) / f'{admission["next_sequence"]:08d}.json'
        now = time.monotonic()
        inbox = next_path.parent
        stamp = (inbox.stat().st_ino, inbox.stat().st_mtime_ns) if inbox.exists() else None
        if (next_path.exists() or stamp != self.inbox_stamp
                or now - self.last_admission_check >= self.dynamic.get("integrity_interval_seconds", 30.)):
            chain = read_admission_chain(self.manifest, self.manifest_path, self.state)
            self.last_admission_check = now
            self.inbox_stamp = stamp
            if not admission["closed"]:
                for row in chain["commands"][admission["next_sequence"]:]:
                    for worker in row["command"].get("workers", []):
                        self.workers[worker["id"]] = worker
                        self.state["workers"][worker["id"]] = {"status": "pending", "output": worker["output"],
                            "return_code": None, "pid": None, "start_ticks": None, "started_unix": None,
                            "finished_unix": None, "worker_spec_sha256": worker_spec_digest(worker),
                            "admission_sequence": row["sequence"], "admission_command_sha256": row["sha256"]}
                        admission["accepted_task_ids"].append(worker["id"])
                    admission["commands"].append({key: value for key, value in row.items() if key != "command"})
                    admission.update(next_sequence=row["sequence"] + 1, last_sha256=row["sha256"])
                    if row["kind"] == "close":
                        admission.update(closed=True, close_reason="controller_close")
                self.save()
        controller = self.dynamic["controller_identity"]
        current = proc_identity(controller["pid"], strict=True)
        if current is None or current["start_ticks"] != controller["start_ticks"]:
            admission.update(closed=True, close_reason="controller_lost", controller_lost_unix=time.time())
            for record in self.state["workers"].values():
                if record["status"] == "pending":
                    record.update(status="not_started", reason="controller_lost_before_worker_start",
                                  cleanup_verified=True, descendant_cleanup_verified=True,
                                  cleanup_verified_unix=time.time(), remaining_descendants=[])
            self.save()

    def heartbeat_observations(self):
        observations = {}
        now = time.time()
        for identity in self.processes:
            record = self.state["workers"][identity]
            path = Path(record["output"]) / "heartbeat.json"
            try:
                value = json.loads(path.read_text())
                if value.get("task_id") != identity or value.get("pid") != record["pid"]:
                    raise ValueError("heartbeat task/process identity mismatch")
                timestamp = value.get("updated_unix")
                if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                    raise ValueError("heartbeat timestamp unavailable")
                observations[identity] = {"status": value.get("status", value.get("state")),
                    "tokens": value.get("tokens"), "updated_unix": timestamp, "age_seconds": now - timestamp,
                    "path": str(path), "pid": record["pid"], "start_ticks": record["start_ticks"]}
            except (OSError, ValueError, TypeError) as error:
                observations[identity] = {"status": "unavailable", "reason": str(error), "tokens": None,
                                          "age_seconds": None, "path": str(path)}
        return observations

    def observe_policy(self, device):
        if not self.policy:
            return
        observations = self.heartbeat_observations()
        external = [(row["pid"], row.get("start_ticks"), row.get("uid"))
                    for row in (device or {}).get("external_occupants", [])]
        external.append(("whole_card_hbm_above_admission_limit", (device or {}).get("hbm_above_admission_limit", False)))
        self.policy.observe(observations, external)
        for event in self.policy.take_events():
            with (self.output / "throughput-windows.jsonl").open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
        self.state["sharing_policy_log"] = str(self.output / "throughput-windows.jsonl")
        self.state["worker_heartbeats"] = observations

    def request_stop(self, signum=None, frame=None):
        self.reason = self.reason or (f"signal_{signum}" if signum else "controller_stop")
        self.stop.set()

    def owned(self):
        table = process_table(strict=True)
        seeds = {os.getpid()}
        for identity in self.identities.values():
            current = table.get(identity["pid"])
            if current and current["start_ticks"] == identity["start_ticks"]:
                seeds.add(identity["pid"])
        members = family_members(seeds, table) - {os.getpid()}
        found = {pid: table[pid] for pid in members if table[pid]["state"] != "Z"}
        for pid, identity in found.items():
            key = f'{pid}:{identity["start_ticks"]}'
            self.identities[key] = identity
            self.state["all_seen_processes"][key] = identity
        # Associate only descendants of our recorded PID/start identities.
        # A device observation is never an ownership seed.
        for identity, record in self.state["workers"].items():
            known = self.worker_identities.setdefault(identity, {})
            direct = record.get("process")
            if direct:
                known[f'{direct["pid"]}:{direct["start_ticks"]}'] = direct
            live_seeds = {row["pid"] for row in known.values() if row["pid"] in table
                          and table[row["pid"]]["start_ticks"] == row["start_ticks"]}
            for pid in family_members(live_seeds, table):
                if pid in found:
                    row = found[pid]
                    known[f'{pid}:{row["start_ticks"]}'] = row
        return found

    def resources(self, *, force=False):
        owned = self.owned()
        owner = proc_identity(os.getpid(), strict=True)
        if self.lease and (force or self.device_cache is None
                or time.monotonic() - self.device_cache_time >= self.manifest.get("resource_interval_seconds", 1.)
                or self.manifest.get("external_occupant_policy", "stop") == "stop"
                    and any(pid not in owned for pid in self.device_cache.get("process_ids", []))):
            self.device_cache = self.lease.snapshot()
            if self.device_baseline_bytes is None:
                self.device_baseline_bytes = self.device_cache["used_bytes"]
            else:
                self.device_baseline_bytes = min(self.device_baseline_bytes, self.device_cache["used_bytes"])
            self.device_cache["baseline_used_bytes"] = self.device_baseline_bytes
            self.device_cache["observed_unix"] = time.time()
            self.device_cache_time = time.monotonic()
            self.record_device_observation(owned, self.device_cache)
            if self.dynamic:
                observations = self.heartbeat_observations()
                memory = self.state["lifecycle_memory_observations"]
                states = "+".join(sorted({row["status"] or "unknown" for row in observations.values()})) or "idle"
                peaks = memory["whole_card_high_water_by_worker_states"]
                peaks[states] = max(peaks.get(states, 0), self.device_cache["used_bytes"])
                memory["samples"] += 1
                sample = {"observed_unix": time.time(), "physical_id": self.lease.physical_id,
                    "whole_card_used_bytes": self.device_cache["used_bytes"],
                    "external_occupants": self.device_cache.get("external_occupants", []), "workers": observations}
                with Path(memory["log"]).open("a") as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
                    stream.flush()
        device = self.device_cache
        self.state["resources"] = {"observed_unix": time.time(), "rss_bytes": owner["rss_bytes"] + sum(p["rss_bytes"] for p in owned.values()),
                                   "device": device}
        self.state["peak_rss_bytes"] = max(self.state["peak_rss_bytes"], self.state["resources"]["rss_bytes"])
        if device is not None:
            self.state["peak_hbm_bytes"] = max(self.state["peak_hbm_bytes"], device["used_bytes"])
            self.state["peak_observed_npu_processes"] = max(self.state["peak_observed_npu_processes"], len(device["process_ids"]))
        # Retain the decisive sample even when it triggers a stop.
        admission_reason(self.manifest, None, [], owned, device, owner_rss=owner["rss_bytes"])
        return owned, device, owner["rss_bytes"]

    def record_device_observation(self, owned, device):
        external = []
        for pid in sorted(set(device.get("process_ids", [])) - set(owned)):
            identity = proc_identity(pid)
            row = {"pid": pid, "start_ticks": identity["start_ticks"] if identity else None,
                   "identity_status": "observed" if identity else "unavailable"}
            try:
                row["uid"] = Path(f"/proc/{pid}").stat().st_uid
            except OSError:
                row["uid"] = None
            external.append(row)
        device["external_occupants"] = external
        limit = min(self.manifest["max_hbm_bytes"], device["total_bytes"])
        device["hbm_above_admission_limit"] = device["used_bytes"] > limit
        self.state["external_contention_observed"] |= bool(external)
        self.state["peak_external_processes"] = max(self.state["peak_external_processes"], len(external))
        conditions = (device.get("health"), device["hbm_above_admission_limit"],
                      tuple((row["pid"], row["start_ticks"], row["uid"]) for row in external))
        if conditions != self.last_device_conditions:
            path = self.output / "device-observations.jsonl"
            event = {"external_occupant_policy": self.state["external_occupant_policy"],
                     "scope": "sampled whole-card usage and external identities; no per-process HBM attribution",
                     "admission_limit_bytes": limit, **device}
            with path.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.state["device_observation_log"] = str(path)
            self.state["device_observation_events"] += 1
            self.last_device_conditions = conditions

    def capacity_wait(self, worker, active, reason):
        self.state["workers"][worker["id"]]["waiting_reason"] = reason
        if self.policy:
            self.policy.resource_wait(reason, self.processes)
        if (not active and self.manifest.get("release_on_idle_hbm_wait") and reason in
                {"hbm_reservations_full", "exclusive_hbm_requires_empty_card", "exclusive_hbm_requires_64_gib"}):
            self.state["capacity_deferral"] = {"task_id": worker["id"], "reason": reason,
                "device": self.state["resources"]["device"], "observed_unix": time.time()}
            return
        # External HBM pressure can subside even when the last own worker has
        # finished. Wait inside the original deadline/card-time budget.
        if (not active and not (reason == "hbm_reservations_full"
                and self.manifest.get("external_occupant_policy", "stop") == "continue_running")):
            raise GroupBlocked("single_worker_cannot_fit:" + reason)

    def start_worker(self, worker):
        if self.stop.is_set() or time.time() >= self.manifest["deadline_unix"]:
            raise GroupBlocked(self.reason or "deadline_before_worker_launch")
        if self.lease and time.monotonic() - self.lease.acquired_monotonic >= self.manifest["max_device_seconds"]:
            raise GroupBlocked("group_device_seconds_budget")
        if self.manifest_path and sha256(self.manifest_path) != self.manifest_sha:
            raise GroupBlocked("group_manifest_changed")
        if self.dynamic or self.manifest.get("exclusive_hbm") and worker.get("task_sha256"):
            validate_dynamic_worker(self.manifest, worker)
            record = self.state["workers"][worker["id"]]
            if worker_spec_digest(worker) != record["worker_spec_sha256"]:
                raise GroupBlocked("dynamic_worker_spec_changed")
            sequence = record["admission_sequence"]
            if sequence is not None and sha256(Path(self.dynamic["inbox_dir"]) / f"{sequence:08d}.json") != record["admission_command_sha256"]:
                raise GroupBlocked("dynamic_admission_command_changed")
        output = Path(worker["output"])
        output.mkdir(parents=True, exist_ok=True)
        if any((output / name).exists() for name in ("result.json", "heartbeat.json", "checkpoint.json")):
            raise FileExistsError("refusing existing worker artifacts")
        environment = {**os.environ, "OMP_NUM_THREADS": str(worker["cpu_threads"]),
                       "MKL_NUM_THREADS": str(worker["cpu_threads"]), "OPENBLAS_NUM_THREADS": "1",
                       "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                       "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", **worker.get("env", {})}
        if self.lease:
            environment["ASCEND_RT_VISIBLE_DEVICES"] = str(self.lease.physical_id)
        affinity = worker.get("cpu_affinity", self.manifest.get("cpu_affinity"))
        log = (output / "stdout.log").open("x", buffering=1)
        parent = os.getpid()
        try:
            process = subprocess.Popen(worker["argv"], cwd=worker["cwd"], env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=self.lease.pass_fds() if self.lease else (),
                preexec_fn=lambda: child_setup(parent, affinity))
        except BaseException:
            log.close()
            raise
        identity = worker["id"]
        # Register immediately; even a failed /proc read must leave the child
        # reachable for cleanup through its Popen and subreaper parentage.
        self.processes[identity], self.logs[identity] = process, log
        self.monotonic_starts[identity] = time.monotonic()
        record = self.state["workers"][identity]
        record.pop("waiting_reason", None)
        record.update(status="running", pid=process.pid, started_unix=time.time(), log=str(output / "stdout.log"))
        observed = proc_identity(process.pid, strict=True)
        if observed is None:
            raise GroupBlocked("worker_exited_before_identity_capture")
        record.update(process=observed, start_ticks=observed["start_ticks"], cpu_threads=worker["cpu_threads"],
                      cpu_affinity_requested=affinity, cpu_affinity_observed=sorted(os.sched_getaffinity(process.pid)),
                      logical_device="npu:0" if self.lease else "cpu",
                      physical_id=self.lease.physical_id if self.lease else None)
        self.identities[f'{observed["pid"]}:{observed["start_ticks"]}'] = observed
        self.state["peak_parallel_workers"] = max(self.state["peak_parallel_workers"], len(self.processes))
        self.save()

    def poll_workers(self):
        self.owned()
        for identity, process in list(self.processes.items()):
            code = process.poll()
            record = self.state["workers"][identity]
            if code is None:
                record["wall_seconds"] = time.monotonic() - self.monotonic_starts[identity]
                continue
            process.wait()
            record.update(return_code=code, finished_unix=time.time(),
                          wall_seconds=time.monotonic() - self.monotonic_starts[identity])
            self.logs.pop(identity).close()
            del self.processes[identity]
            result_path = Path(record["output"]) / "result.json"
            result = None
            try:
                result = json.loads(result_path.read_text())
                record.update(result_path=str(result_path), result_sha256=sha256(result_path),
                              result_status=result.get("status"), failure_class=result.get("failure_class"))
            except (OSError, ValueError, AttributeError) as error:
                record["result_observation_error"] = str(error)
                result = None
            allowed = (code == 1 and isinstance(result, dict) and result.get("status") == "failed" and result.get("failure_class")
                       in self.workers[identity].get("allowed_failure_classes", []))
            if (self.dynamic or self.manifest.get("exclusive_hbm") and self.workers[identity].get("task_sha256")) and code == 1 and isinstance(result, dict):
                task = validate_dynamic_worker(self.manifest, self.workers[identity])
                allowed |= (result.get("status") == "recipe_rejected"
                    and result.get("failure_class") == "declared_optimizer_instability"
                    and result.get("task_sha256") == self.workers[identity]["task_sha256"]
                    and result.get("id") == task["id"] and result.get("model_id") == task.get("model_id")
                    and result.get("phase") == task.get("phase")
                    and bool(task.get("source_identity")) and result.get("source_identity") == task["source_identity"])
            record["exit_status"] = "completed" if code == 0 else "allowed_failure" if allowed else "failed"
            record["status"] = "exited_waiting_cleanup"
            if self.dynamic and result is None:
                record["exit_status"] = "failed"
                self.reason = self.reason or f"worker_result_unreadable:{identity}"
                self.stop.set()
            if self.dynamic and isinstance(result, dict):
                self.state["lifecycle_memory_observations"]["final_result_device_memory"][identity] = result.get("device_memory", {})
                tokens = result.get("training", {}).get("segment_tokens", 0)
                self.policy.progress_seen |= type(tokens) is int and tokens > 0
            if code and not allowed and not self.reason:
                self.reason = f"worker_process_failed:{identity}:{code}"
                self.stop.set()
        self.finalize_worker_cleanup()

    def finalize_worker_cleanup(self):
        owned = self.owned()
        attributed = {row["pid"] for known in self.worker_identities.values() for row in known.values()
                      if row["pid"] in owned and owned[row["pid"]]["start_ticks"] == row["start_ticks"]}
        unknown = set(owned) - attributed
        for identity, record in self.state["workers"].items():
            if record["status"] != "exited_waiting_cleanup":
                continue
            known = self.worker_identities.get(identity, {})
            remaining = [owned[row["pid"]] for row in known.values() if row["pid"] in owned
                         and owned[row["pid"]]["start_ticks"] == row["start_ticks"]]
            record["remaining_descendants"] = remaining
            if remaining or unknown:
                record.update(cleanup_verified=False, descendant_cleanup_verified=False)
                continue
            record.update(status=record["exit_status"], cleanup_verified=True, descendant_cleanup_verified=True,
                          cleanup_verified_unix=time.time(), remaining_descendants=[],
                          all_seen_processes=dict(known))

    def _reap_orphans(self):
        # Direct children remain owned by their Popen objects until polled.
        direct = {process.pid for process in self.processes.values()}
        table = process_table(strict=True)
        for pid, row in table.items():
            if row["ppid"] == os.getpid() and row["state"] == "Z" and pid not in direct:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass

    def cleanup(self):
        # Registered direct children remain signalable even if a subsequent
        # /proc observation fails. No unknown device occupant is signalled.
        for process in self.processes.values():
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        def signal_owned(sig):
            for pid, identity in self.owned().items():
                current = proc_identity(pid, strict=True)
                if current and current["start_ticks"] == identity["start_ticks"]:
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
        signal_owned(signal.SIGTERM)
        for sig, seconds in ((signal.SIGTERM, self.manifest.get("stop_grace_seconds", 20.)),
                             (signal.SIGKILL, self.manifest.get("kill_grace_seconds", 10.))):
            if sig == signal.SIGKILL:
                signal_owned(sig)
            until = time.monotonic() + seconds
            next_signal = time.monotonic() + 1.
            while self.owned() and time.monotonic() < until:
                self.poll_workers()
                self._reap_orphans()
                # A worker can install/reset its signal flag while importing
                # its runtime. Reassert the same stop request during grace so
                # that such a race still reaches a checkpoint-safe boundary.
                if sig == signal.SIGTERM and time.monotonic() >= next_signal:
                    signal_owned(signal.SIGTERM)
                    next_signal = time.monotonic() + 1.
                time.sleep(min(.1, self.manifest.get("poll_seconds", .25)))
            if not self.owned():
                break
        self.poll_workers()
        self._reap_orphans()
        self.state["remaining_descendants"] = list(self.owned().values())
        return not self.state["remaining_descendants"]

    def cancel_empty_ticket(self):
        self.refresh_admissions()
        if self.workers or self.state["admission"].get("close_reason") == "controller_lost":
            raise GroupBlocked("invalid_empty_queue_ticket")
        return self.state["admission"]["closed"]

    def run(self):
        previous_subreaper = child_subreaper(True)
        previous_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous_signals:
            signal.signal(sig, self.request_stop)
        try:
            if self.manifest["backend"] == "npu":
                self.state["status"] = "queued"
                self.save()
                self.lease = self.lease_factory(self.manifest, self.output, self.stop)
                if self.manifest.get("dispatch_lease"):
                    self.lease.cancelled = self.cancel_empty_ticket
                self.lease.acquire()
                if self.manifest.get("dispatch_lease") and self.lease.physical_id is None:
                    if not self.cancel_empty_ticket():
                        raise GroupBlocked("empty_ticket_lacks_verified_close")
                    self.state.update(status="completed", cancelled_before_allocation=True)
                    return self.state
                self.state.update(allocation_acquired_unix=self.lease.acquired_unix, physical_id=self.lease.physical_id,
                                  queue_helper=getattr(self.lease, "helper_identity", None))
            self.state["status"] = "running"
            self.save()
            while True:
                self.poll_workers()
                self._reap_orphans()
                if time.time() >= self.manifest["deadline_unix"]:
                    raise GroupBlocked("absolute_deadline")
                if self.lease and time.monotonic() - self.lease.acquired_monotonic >= self.manifest["max_device_seconds"]:
                    raise GroupBlocked("group_device_seconds_budget")
                if self.stop.is_set():
                    raise GroupBlocked(self.reason or "stop_requested")
                owned, device, owner_rss = self.resources()
                direct_pids = {process.pid for process in self.processes.values()}
                if any(row["ppid"] == os.getpid() and pid not in direct_pids for pid, row in owned.items()):
                    raise GroupBlocked("worker_left_live_orphan_descendant")
                self.refresh_admissions()
                self.observe_policy(device)
                if self.state.get("capacity_deferral"):
                    # Drain even an admission already committed by the controller
                    # while the device sample was in flight. No child is active.
                    for record in self.state["workers"].values():
                        if record["status"] == "pending":
                            record.update(status="not_started", reason="hbm_admission_deferred")
                pending = [worker for worker in self.workers.values() if self.state["workers"][worker["id"]]["status"] == "pending"]
                if self.policy:
                    waiting = len(pending) > max(0, self.policy.limit-len(self.processes))
                    headroom = bool(pending) and admission_reason(self.manifest, pending[0],
                        [self.workers[key] for key in self.processes], owned, device, owner_rss=owner_rss) is None
                    self.policy.backlog(waiting, headroom)
                if not pending and not self.processes:
                    if owned:
                        raise GroupBlocked("worker_descendants_outlived_group")
                    if not self.dynamic or self.state["admission"]["closed"]:
                        if self.dynamic:
                            chain = read_admission_chain(self.manifest, self.manifest_path, self.state)
                            if chain["next_sequence"] != self.state["admission"]["next_sequence"]:
                                raise GroupBlocked("unacknowledged_commands_at_group_drain")
                        self.state["status"] = "completed"
                        break
                    self.state["admission"]["idle_since_unix"] = self.state["admission"].get("idle_since_unix") or time.time()
                elif self.dynamic:
                    self.state["admission"]["idle_since_unix"] = None
                for worker in pending:
                    limit = self.policy.limit if self.policy else self.manifest["max_parallel"]
                    if len(self.processes) >= limit:
                        break
                    active = [self.workers[key] for key in self.processes]
                    # Cached pressure can defer admission without querying
                    # npu-smi every poll. Every actual launch still gets two
                    # fresh samples below.
                    reason = admission_reason(self.manifest, worker, active, owned, device, owner_rss=owner_rss)
                    if reason:
                        self.capacity_wait(worker, active, reason)
                        break
                    owned, device, owner_rss = self.resources(force=True)
                    reason = admission_reason(self.manifest, worker, active, owned, device, owner_rss=owner_rss)
                    if reason:
                        self.capacity_wait(worker, active, reason)
                        break
                    if self.lease:
                        first_device = device
                        self.stop.wait(self.manifest.get("sharing_sample_seconds", .2))
                        if self.stop.is_set():
                            raise GroupBlocked(self.reason or "stop_requested")
                        owned, device, owner_rss = self.resources(force=True)
                        reason = admission_reason(self.manifest, worker, active, owned, device, owner_rss=owner_rss)
                        if reason:
                            self.capacity_wait(worker, active, reason)
                            break
                        if self.manifest.get("exclusive_hbm"):
                            self.state["workers"][worker["id"]]["hbm_admission_samples"] = [
                                {**d, "projected_bytes": max(d["used_bytes"], d["baseline_used_bytes"])
                                 + worker["reserved_hbm_bytes"]} for d in (first_device, device)]
                    self.start_worker(worker)
                self.save()
                self.stop.wait(self.manifest.get("poll_seconds", .25))
        except BaseException as error:
            self.reason = self.reason or f"{type(error).__name__}: {error}"
            self.state.update(status="failed", reason=self.reason)
        finally:
            # Never release the lease while a known descendant is still alive.
            # A kernel-uninterruptible process leaves a visible cleanup state;
            # the durable service's cgroup remains the outer stop boundary.
            try:
                while True:
                    try:
                        if self.cleanup():
                            break
                    except (OSError, RuntimeError, ValueError) as error:
                        self.state["cleanup_observation_error"] = f"{type(error).__name__}: {error}"
                    self.state.update(status="cleanup_blocked", reason=self.reason or "descendants_survived_sigkill")
                    self.save()
                    time.sleep(1.)
                for record in self.state["workers"].values():
                    if record["status"] == "pending":
                        record.update(status="not_started", reason=self.reason)
                if self.lease:
                    try:
                        self.lease.release(self.state["status"])
                        self.state["queue_released_unix"] = time.time()
                    except Exception as error:
                        self.state.update(status="failed", reason=self.reason or "queue_release_unverified",
                            queue_release_error=f"{type(error).__name__}: {error}")
                    self.state["allocation_released_unix"] = getattr(self.lease, "released_unix", None)
                    if self.lease.acquired_monotonic is not None:
                        finish = getattr(self.lease, "released_monotonic", None) or time.monotonic()
                        self.state["physical_device_seconds"] = max(0., finish - self.lease.acquired_monotonic)
                self.state.update(finished_unix=time.time(), remaining_descendants=[])
                self.save()
            finally:
                for sig, previous in previous_signals.items():
                    signal.signal(sig, previous)
                child_subreaper(previous_subreaper)
        return self.state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    runner = GroupRunner(manifest, args.output, manifest_path=args.manifest)
    result = runner.run()
    print(json.dumps({"status": result["status"], "reason": result.get("reason"),
                      "physical_device_seconds": result["physical_device_seconds"], "group": str(args.output / "group.json")}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
