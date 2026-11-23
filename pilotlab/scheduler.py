"""Bounded first-cycle task scheduling; this module does not import Torch.

Run this controller inside a detached ``background.slice`` user service.  NPU
admission remains owned by the installed cooperative queue helper.  A task's
worker result, subprocess status, and process termination are separate facts.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


GIB = 1024 ** 3
QUEUE_HELPER = Path("/mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py")
TERMINAL = {"completed", "failed", "budget_exhausted", "not_started"}
SUCCESSFUL_PARENT = {"completed", "budget_exhausted"}
PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child_subreaper(enabled: bool | None = None) -> bool:
    """Adopt daemonized descendants without changing any cgroup configuration."""
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    if enabled is not None and libc.prctl(36, int(enabled), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    return bool(previous.value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def immutable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class Quota:
    filesystem: str
    used_bytes: int
    soft_bytes: int
    hard_bytes: int
    grace_epoch: int


def storage_headroom(quota: Quota, filesystem_free: int) -> int:
    """Bytes available under the account hard limit and the output filesystem."""
    if quota.hard_bytes == 0:
        return filesystem_free
    return min(filesystem_free, quota.hard_bytes - quota.used_bytes)


def parse_quota(text: str) -> list[Quota]:
    """Parse ``quota -w -v -p``; quota block units are 1024 bytes."""
    rows: list[Quota] = []
    pending: str | None = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith("/") or ":/" in parts[0]:
            pending, parts = parts[0], parts[1:]
        elif pending is None:
            continue
        if not parts:
            continue
        if len(parts) < 4:
            raise ValueError("incomplete quota row")
        try:
            numbers = [int(p.rstrip("*")) for p in parts[:4]]
        except ValueError as exc:
            raise ValueError("quota output must use numeric -p grace values") from exc
        if any(n < 0 for n in numbers):
            raise ValueError("negative quota value")
        rows.append(Quota(pending, *(n * 1024 for n in numbers[:3]), numbers[3]))
        pending = None
    if not rows:
        raise ValueError("no usable account quota row")
    return rows


def current_quota(filesystem: str | None = None) -> Quota:
    result = subprocess.run(["quota", "-w", "-v", "-p"], capture_output=True,
                            text=True, timeout=10, env={**os.environ, "LC_ALL": "C"})
    # quota(1) reports limit warnings through its exit status as well as stdout.
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"quota query failed ({result.returncode}): {result.stderr.strip()}")
    rows = parse_quota(result.stdout)
    if filesystem is not None:
        rows = [row for row in rows if row.filesystem == filesystem]
    if len(rows) != 1:
        raise RuntimeError("quota filesystem is ambiguous or absent; set plan.quota_filesystem")
    if result.returncode == 1 and not (
        rows[0].soft_bytes and rows[0].used_bytes >= rows[0].soft_bytes
        or rows[0].hard_bytes and rows[0].used_bytes >= rows[0].hard_bytes
    ):
        raise RuntimeError(f"quota query failed without an explained limit excess: {result.stderr.strip()}")
    return rows[0]


def tree_bytes(root: Path) -> int:
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                path = Path(directory) / name
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass  # A worker may atomically replace a checkpoint during inspection.
    return total


def storage_snapshot(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    quota = current_quota(plan.get("quota_filesystem"))
    free = shutil.disk_usage(root).free
    used = tree_bytes(root)
    margin = int(plan.get("quota_reserve_bytes", 16 * GIB))
    cap = int(plan.get("max_output_bytes", 128 * GIB))
    expired = bool(quota.soft_bytes and quota.used_bytes > quota.soft_bytes
                   and (quota.grace_epoch == 0 or quota.grace_epoch <= time.time()))
    return {"observed_unix": time.time(), "quota": asdict(quota),
            "filesystem_free_bytes": free, "output_bytes": used,
            "reserve_bytes": margin, "output_limit_bytes": cap,
            "quota_limit_basis": "hard",
            "available_for_admission_bytes": min(storage_headroom(quota, free) - margin, cap - used),
            "soft_grace_expired_or_unknown": expired}


def proc_identity(pid: int, *, strict: bool = False) -> dict[str, int | str] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        result = {"pid": pid, "ppid": int(fields[1]), "pgrp": int(fields[2]),
                  "start_ticks": int(fields[19]), "state": fields[0],
                  "rss_bytes": int(fields[21]) * PAGE_BYTES}
        if result["start_ticks"] <= 0 or result["rss_bytes"] < 0:
            raise ValueError("invalid process start identity or RSS")
        return result
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, IndexError) as exc:
        if strict:
            raise RuntimeError(f"cannot measure process {pid}: {exc}") from exc
        return None


def process_table(*, strict: bool = False) -> dict[int, dict[str, Any]]:
    found = {}
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:
                # All supported workers keep the scheduler's uid.  Other
                # accounts need not expose their stat files to this user.
                if path.stat().st_uid != os.getuid():
                    continue
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError:
                if strict:
                    raise
                continue
            item = proc_identity(int(path.name), strict=strict)
            if item is not None:
                found[item["pid"]] = item
    return found


def family_members(seeds: set[int], table: dict[int, dict[str, Any]]) -> set[int]:
    members = set(seeds)
    while True:
        children = {pid for pid, row in table.items() if row["ppid"] in members}
        if children <= members:
            return members
        members.update(children)


def host_memory() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if parts[0] in {"MemTotal:", "MemAvailable:"}:
            if len(parts) != 3 or parts[2] != "kB":
                raise ValueError("unexpected host memory units")
            values[parts[0][:-1]] = int(parts[1]) * 1024
    if (set(values) != {"MemTotal", "MemAvailable"}
            or not 0 <= values["MemAvailable"] <= values["MemTotal"]
            or values["MemTotal"] <= 0):
        raise ValueError("invalid or missing MemAvailable/MemTotal")
    return {"total_bytes": values["MemTotal"], "available_bytes": values["MemAvailable"]}


def memory_policy(plan: dict[str, Any]) -> dict[str, Any]:
    requested = plan.get("memory_guard", {})
    if not isinstance(requested, dict):
        raise ValueError("memory_guard must be an object")
    result = {"limit_bytes": 256 * GIB, "reserve_bytes": 16 * GIB,
              "system_reserve_bytes": 16 * GIB, "poll_seconds": 1.0,
              "admission_factor": 1.25, **requested}
    if set(result) != {"limit_bytes", "reserve_bytes", "system_reserve_bytes",
                       "poll_seconds", "admission_factor"}:
        raise ValueError("unknown memory_guard setting")
    for key in ("limit_bytes", "reserve_bytes", "system_reserve_bytes"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"memory_guard.{key} must be a nonnegative integer")
    if not 0 <= result["reserve_bytes"] < result["limit_bytes"]:
        raise ValueError("memory reserve must be below the positive limit")
    for key, lower in (("poll_seconds", 0), ("admission_factor", 1)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid memory_guard.{key}")
        if value < lower or (key == "poll_seconds" and value == 0):
            raise ValueError(f"invalid memory_guard.{key}")
    return result


def dependency_ids(task: dict[str, Any]) -> list[str]:
    dependencies = list(task.get("depends_on", []))
    if task.get("resume_task") is not None and task["resume_task"] not in dependencies:
        dependencies.append(task["resume_task"])
    return dependencies


def validate_graph(plan: dict[str, Any], tasks: list[dict[str, Any]],
                   ceilings: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    phases = plan.get("phases", [{"id": "default", **{k + "_slots": v for k, v in ceilings.items()}}])
    if not isinstance(phases, list) or not phases:
        raise ValueError("phases must be a nonempty ordered list")
    normalized = []
    for phase in phases:
        if not isinstance(phase, dict) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", phase.get("id", "")):
            raise ValueError("each phase needs a safe unique id")
        limits = {backend + "_slots": phase.get(backend + "_slots", ceiling)
                  for backend, ceiling in ceilings.items()}
        for backend, ceiling in ceilings.items():
            value = limits[backend + "_slots"]
            if type(value) is not int or not 0 <= value <= ceiling:
                raise ValueError("phase slots must be between zero and the launcher ceiling")
        normalized.append({"id": phase["id"], **limits})
    order = {phase["id"]: i for i, phase in enumerate(normalized)}
    if len(order) != len(normalized):
        raise ValueError("duplicate phase id")
    task_phase = {task["id"]: task.get("phase", normalized[0]["id"]) for task in tasks}
    graph = {}
    for task in tasks:
        phase = task_phase[task["id"]]
        if phase not in order:
            raise ValueError("task refers to an unknown phase")
        if not normalized[order[phase]][task["backend"] + "_slots"]:
            raise ValueError("task backend has no slots in its phase")
        depends = task.get("depends_on", [])
        if (not isinstance(depends, list) or any(not isinstance(d, str) for d in depends)
                or len(set(depends)) != len(depends)):
            raise ValueError("depends_on must contain distinct task ids")
        if "resume_task" in task and not isinstance(task["resume_task"], str):
            raise ValueError("resume_task must be a task id")
        if "resume_task" in task and "resume_from" in task:
            raise ValueError("declare resume_task or resume_from, not both")
        if "resume_select" in task:
            if not isinstance(task["resume_select"], dict) or not depends or "resume_task" in task or "resume_from" in task:
                raise ValueError("resume_select needs dependencies and no preselected parent")
        allowed = task.get("allow_parent_status", ["completed"])
        if (not isinstance(allowed, list) or not allowed or any(not isinstance(v, str) for v in allowed)
                or not set(allowed) <= SUCCESSFUL_PARENT):
            raise ValueError("allow_parent_status supports completed and budget_exhausted only")
        dependencies = dependency_ids(task)
        for parent in dependencies:
            if parent not in task_phase or parent == task["id"]:
                raise ValueError("unknown or self dependency")
            if order[task_phase[parent]] > order[phase]:
                raise ValueError("dependencies cannot point to a later phase")
        graph[task["id"]] = dependencies
    visiting, visited = set(), set()
    def visit(key: str) -> None:
        if key in visiting:
            raise ValueError("task dependency cycle")
        if key in visited:
            return
        visiting.add(key)
        for parent in graph[key]:
            visit(parent)
        visiting.remove(key)
        visited.add(key)
    for key in graph:
        visit(key)
    return normalized, task_phase


def same_live(identity: dict[str, Any], table: dict[int, dict[str, Any]]) -> bool:
    current = table.get(identity["pid"])
    return bool(current and current["start_ticks"] == identity["start_ticks"]
                and current["state"] not in {"Z", "X"})


def unix_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return None


def task_outcome(return_code: int | None, result: dict[str, Any] | None,
                 termination_reason: str | None) -> tuple[str, str]:
    # An inner failure is never overwritten by an outer timeout/success.
    if result and result.get("status") == "failed":
        return "failed", "worker_reported_failure"
    if termination_reason:
        if termination_reason in {"task_deadline", "queue_deadline", "round_deadline", "signal"}:
            return "budget_exhausted", termination_reason
        return "failed", termination_reason
    if result is None:
        return "failed", "missing_worker_result"
    if return_code != 0:
        return "failed", "nonzero_process_exit"
    if result.get("status") in {"completed", "budget_exhausted"}:
        return result["status"], "worker_result"
    return "failed", "invalid_worker_status"


@dataclass
class Running:
    task: dict[str, Any]
    directory: Path
    process: subprocess.Popen
    log: Any
    cpus: list[int]
    deadline: float
    record: dict[str, Any]
    descendants: dict[int, dict[str, Any]] = field(default_factory=dict)
    term_sent: float | None = None
    kill_sent: float | None = None
    termination_reason: str | None = None


class Scheduler:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.plan_path = Path(args.plan).resolve()
        self.plan_sha256 = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        self.plan = read_json(self.plan_path)
        if self.plan is None:
            raise ValueError("plan is missing")
        self.tasks = self.plan.get("tasks")
        if not isinstance(self.tasks, list) or not self.tasks:
            raise ValueError("plan.tasks must be a nonempty list")
        identifiers = []
        self.affinity = sorted(os.sched_getaffinity(0))
        for task in self.tasks:
            if not isinstance(task, dict) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", task.get("id", "")):
                raise ValueError("each task needs a safe unique id")
            if task.get("backend") not in {"cpu", "npu"}:
                raise ValueError("task backend must be cpu or npu")
            threads = task.get("cpu_threads", 8)
            if type(threads) is not int or not 1 <= threads <= len(self.affinity):
                raise ValueError("task cpu_threads exceeds the effective affinity")
            fixed = task.get("cpu_affinity")
            if fixed is not None and (not isinstance(fixed, list) or len(fixed) != threads
                                      or any(type(cpu) is not int for cpu in fixed)
                                      or len(set(fixed)) != threads or not set(fixed) <= set(self.affinity)):
                raise ValueError("task cpu_affinity must contain cpu_threads distinct allowed CPUs")
            wall = float(task.get("wall_seconds", 600))
            if not math.isfinite(wall) or wall <= 0:
                raise ValueError("task wall_seconds must be positive")
            if int(task.get("estimated_output_bytes", 2 * GIB)) < 0:
                raise ValueError("negative task output reservation")
            estimate = task.get("estimated_rss_bytes", 4 * GIB)
            if type(estimate) is not int or estimate <= 0:
                raise ValueError("task estimated_rss_bytes must be a positive integer")
            candidates = task.get("candidate_ids")
            if candidates is not None and (task["backend"] != "npu" or not isinstance(candidates, list)
                                           or not candidates or any(type(v) is not int or v < 0 for v in candidates)
                                           or len(set(candidates)) != len(candidates)):
                raise ValueError("candidate_ids must contain distinct nonnegative NPU ids")
            identifiers.append(task["id"])
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate task id")
        if args.npu_slots < 0 or args.cpu_slots < 0 or args.duration_seconds <= 0:
            raise ValueError("invalid slots or duration")
        if args.npu_slots == args.cpu_slots == 0:
            raise ValueError("at least one backend slot is required")
        self.phases, self.task_phase = validate_graph(
            self.plan, self.tasks, {"cpu": args.cpu_slots, "npu": args.npu_slots})
        self.phase_index = 0
        self.phase_records = [{**phase, "status": "pending"} for phase in self.phases]
        self.memory_policy = memory_policy(self.plan)
        self.source = Path(args.source).resolve()
        self.foundation_source_requested = {"plan": self.plan.get("foundation_source"),
                                            "cli": getattr(args, "foundation_source", None)}
        foundations = []
        for origin, value in self.foundation_source_requested.items():
            if value is None:
                continue
            if not isinstance(value, str) or not Path(value).is_absolute() or os.pathsep in value:
                raise ValueError(f"{origin} foundation_source must be a single absolute source directory")
            path = Path(value).resolve(strict=True)
            if not path.is_dir():
                raise ValueError(f"{origin} foundation_source is not a directory: {path}")
            foundations.append(path)
        if len(set(foundations)) > 1:
            raise ValueError("--foundation-source conflicts with plan.foundation_source")
        self.foundation_source = foundations[0] if foundations else None
        # A wrapper's arbitrary PYTHONPATH is not part of a frozen plan.  The
        # worker and every fresh-interpreter recovery child inherit only these
        # explicit source roots, in this order.
        source_roots = [self.source] + ([] if self.foundation_source is None else [self.foundation_source])
        self.resolved_environment = {"PYTHONPATH": os.pathsep.join(str(path) for path in source_roots)}
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "scheduler.json").exists() or (self.root / "plan.json").exists():
            raise FileExistsError("scheduler output already exists; use a new output directory")
        immutable_json(self.root / "plan.json", self.plan)
        # Resolving a venv's python symlink would silently escape its packages.
        self.python = os.path.abspath(args.python)
        self.data = Path(args.data).resolve()
        self.queue_helper = Path(self.plan.get("queue_helper", str(QUEUE_HELPER))).resolve()
        self.worker_module = self.plan.get("worker_module", "pilotlab.worker")
        self.started = time.time()
        self.deadline = self.started + args.duration_seconds
        self.dispatch_deadline = self.deadline - float(self.plan.get("dispatch_reserve_seconds", 300))
        self.finish_reserve = float(self.plan.get("finish_reserve_seconds", 60))
        self.grace = float(self.plan.get("term_grace_seconds", 20))
        self.poll = float(self.plan.get("poll_seconds", 1))
        if min(self.finish_reserve, self.grace) < 0 or self.poll <= 0:
            raise ValueError("invalid reserve/grace/poll settings")
        self.pending = list(self.tasks)
        self.running: dict[str, Running] = {}
        self.records: dict[str, dict[str, Any]] = {
            t["id"]: {"id": t["id"], "backend": t["backend"], "status": "not_started",
                      "phase": self.task_phase[t["id"]], "depends_on": dependency_ids(t)}
            for t in self.tasks}
        self.stop_reason: str | None = None
        self.signal_number: int | None = None
        self.last_storage: dict[str, Any] | None = None
        self.last_memory: dict[str, Any] | None = None
        self.memory_checked = -math.inf
        self.peak_rss_bytes = 0
        self.memory_samples = 0
        self.memory_error: str | None = None
        self.round_descendants: dict[int, dict[str, Any]] = {}
        self.orphans: dict[int, dict[str, Any]] = {}
        self.storage_checked = -math.inf
        self.last_persist = -math.inf
        self.events_path = self.root / "events.jsonl"
        self.memory_path = self.root / "memory.jsonl"
        self.round_id = self.plan.get("round_id", "pilot") + "-" + str(os.getpid())
        self.round_id = re.sub(r"[^A-Za-z0-9_.-]", "-", self.round_id)[:90]
        self.event("round_started", deadline_unix=self.deadline, affinity=self.affinity)

    def event(self, kind: str, **details: Any) -> None:
        event = {"event": kind, "timestamp": utc_now(), "unix": time.time(), **details}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def persist(self, status: str = "running") -> None:
        counts: dict[str, int] = {}
        for record in self.records.values():
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        atomic_json(self.root / "scheduler.json", {
            "schema_version": 2, "status": status, "started_unix": self.started,
            "updated_unix": time.time(), "deadline_unix": self.deadline,
            "dispatch_deadline_unix": self.dispatch_deadline, "stop_reason": self.stop_reason,
            "signal": self.signal_number, "scheduler_identity": proc_identity(os.getpid()),
            "cgroup": Path("/proc/self/cgroup").read_text(), "python": self.python,
            "source": str(self.source), "data": str(self.data),
            "foundation_source": None if self.foundation_source is None else str(self.foundation_source),
            "foundation_source_requested": self.foundation_source_requested,
            "resolved_environment": self.resolved_environment,
            "plan_sha256": self.plan_sha256,
            "effective_cpu_affinity": self.affinity,
            "slots": {"cpu": self.args.cpu_slots, "npu": self.args.npu_slots},
            "phases": self.phase_records,
            "active_phase": self.phases[self.phase_index]["id"] if self.phase_index < len(self.phases) else None,
            "memory_guard": {"requested": self.plan.get("memory_guard", {}),
                             "effective": self.memory_policy,
                             "enforcement": "sampled_process_rss_software_watchdog",
                             "kernel_memory_limit_established": False,
                             "rss_scope": "scheduler_and_all_observed_descendant_processes",
                             "limitations": ["RSS sums count shared resident pages more than once",
                                             "allocations between samples can exceed the limit",
                                             "RSS excludes filesystem page cache and NPU device memory"],
                             "samples": self.memory_samples, "peak_rss_bytes": self.peak_rss_bytes,
                             "last_observation": self.last_memory, "error": self.memory_error},
            "storage": self.last_storage, "counts": counts,
            "tasks": list(self.records.values()),
            "orphan_processes": list(self.orphans.values()),
            "all_task_processes_terminated": not self.running and not self.orphans,
        })
        self.last_persist = time.monotonic()

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.signal_number = signum
        self.stop_reason = self.stop_reason or "signal"

    def allocated_cpus(self) -> set[int]:
        return {cpu for run in self.running.values() for cpu in run.cpus}

    def observe_processes(self, table: dict[int, dict[str, Any]]) -> set[int]:
        """Track trees, retained identities and adopted children, not just PGIDs."""
        seeds = {os.getpid()} | {pid for pid, row in self.round_descendants.items() if same_live(row, table)}
        members = family_members(seeds, table)
        assigned: set[int] = set()
        for run in self.running.values():
            leader = run.record["process_identity"]
            current = table.get(run.process.pid)
            task_seeds = {pid for pid, row in run.descendants.items() if same_live(row, table)}
            if leader and current and current["start_ticks"] == leader["start_ticks"]:
                task_seeds.add(run.process.pid)
            # The original PGID remains valid after its leader exits.  A
            # reused leader PID with a new identity must not claim that group.
            if current is None or leader and current["start_ticks"] == leader["start_ticks"]:
                task_seeds.update(pid for pid, row in table.items() if row["pgrp"] == run.process.pid)
            task_members = family_members(task_seeds, table)
            members.update(task_members)
            for pid in task_members:
                if pid in table:
                    run.descendants[pid] = table[pid]
            assigned.update(task_members)
        # A child can double-fork and create a new session between samples.
        # Subreaping keeps it in this round; inherited markers attribute it.
        for pid in members - assigned - {os.getpid()}:
            row = table.get(pid)
            if row is None or row["state"] in {"Z", "X"}:
                continue
            try:
                raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
                env = dict(value.split(b"=", 1) for value in raw if b"=" in value)
                task_id = env.get(b"PILOTLAB_TASK_ID", b"").decode()
                observed = proc_identity(pid)
                if (env.get(b"PILOTLAB_ROUND_ID") == self.round_id.encode()
                        and task_id in self.running and observed is not None
                        and observed["start_ticks"] == row["start_ticks"]):
                    self.running[task_id].descendants[pid] = row
                    assigned.add(pid)
            except (OSError, UnicodeError, ValueError):
                pass
        for pid in members - {os.getpid()}:
            if pid in table:
                self.round_descendants[pid] = table[pid]
            row = table.get(pid)
            if pid not in assigned and row is not None and row["state"] not in {"Z", "X"}:
                if pid not in self.orphans or self.orphans[pid]["identity"]["start_ticks"] != row["start_ticks"]:
                    self.orphans[pid] = {"identity": row, "term_sent_unix": None, "kill_sent_unix": None}
                    self.stop_reason = self.stop_reason or "unattributed_descendants"
                    self.event("unattributed_descendant", identity=row)
        return members

    def monitor_memory(self, *, force: bool = False) -> None:
        if not force and time.monotonic() - self.memory_checked < self.memory_policy["poll_seconds"]:
            return
        began = time.time()
        try:
            table = process_table(strict=True)
            if os.getpid() not in table:
                raise RuntimeError("scheduler process is absent from memory observation")
            members = self.observe_processes(table)
            task_by_pid = {pid: key for key, run in self.running.items()
                           for pid, row in run.descendants.items() if same_live(row, table)}
            rows = [{**table[pid], "task_id": task_by_pid.get(pid)} for pid in sorted(members)
                    if pid in table and table[pid]["state"] not in {"Z", "X"}]
            for row in rows:
                if type(row.get("rss_bytes")) is not int or row["rss_bytes"] < 0:
                    raise ValueError(f"invalid RSS for process {row['pid']}")
            host = host_memory()
            total = sum(row["rss_bytes"] for row in rows)
            by_task = {key: sum(row["rss_bytes"] for row in rows if row["task_id"] == key)
                       for key in self.running}
            now = time.time()
            snapshot = {"observed_unix": now, "sample_started_unix": began,
                        "scan_seconds": now - began, "total_rss_bytes": total,
                        "scheduler_rss_bytes": table[os.getpid()]["rss_bytes"],
                        "host": host, "task_rss_bytes": by_task, "processes": rows,
                        "interval_since_previous_seconds": (None if self.last_memory is None else
                                                            began - self.last_memory["sample_started_unix"])}
            self.last_memory = snapshot
            self.peak_rss_bytes = max(self.peak_rss_bytes, total)
            self.memory_samples += 1
            for key, rss in by_task.items():
                record = self.running[key].record
                record["last_rss_bytes"] = rss
                record["peak_rss_bytes"] = max(record.get("peak_rss_bytes", 0), rss)
                record["rss_observed_unix"] = now
                record["rss_samples"] = record.get("rss_samples", 0) + 1
            with self.memory_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            reason = None
            if total > self.memory_policy["limit_bytes"]:
                reason = "rss_memory_limit"
            elif host["available_bytes"] < self.memory_policy["system_reserve_bytes"]:
                reason = "host_memory_pressure"
            if reason and self.stop_reason is None:
                self.stop_reason = reason
                self.event("memory_limit_observed", reason=reason, total_rss_bytes=total, host=host)
        except Exception as exc:
            self.last_memory = None
            self.memory_error = f"{type(exc).__name__}: {exc}"
            self.stop_reason = self.stop_reason or "memory_observation_failed"
            self.event("memory_probe_failed", error=self.memory_error)
        self.memory_checked = time.monotonic()

    def cleanup_orphans(self, table: dict[int, dict[str, Any]]) -> None:
        now = time.time()
        for pid, record in list(self.orphans.items()):
            identity = record["identity"]
            if not same_live(identity, table):
                self.event("unattributed_descendant_terminated", **record)
                del self.orphans[pid]
                continue
            sig = None
            if record["term_sent_unix"] is None:
                sig, record["term_sent_unix"] = signal.SIGTERM, now
            elif now - record["term_sent_unix"] >= self.grace and record["kill_sent_unix"] is None:
                sig, record["kill_sent_unix"] = signal.SIGKILL, now
            if sig is not None:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        leaders = {run.process.pid for run in self.running.values()}
        for pid, identity in self.round_descendants.items():
            current = table.get(pid)
            if (pid not in leaders and current and current["start_ticks"] == identity["start_ticks"]
                    and current["state"] == "Z"):
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass

    def memory_admission(self, task: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if (self.last_memory is not None and self.memory_error is None
                and time.monotonic() - self.memory_checked > self.memory_policy["poll_seconds"] * 2):
            self.monitor_memory(force=True)
        if self.last_memory is None or self.memory_error is not None:
            return False, {"reason": "missing_valid_memory_observation"}
        age = time.monotonic() - self.memory_checked
        if age > self.memory_policy["poll_seconds"] * 2:
            return False, {"reason": "stale_memory_observation", "age_seconds": age}
        estimate = math.ceil(task.get("estimated_rss_bytes", 4 * GIB) * self.memory_policy["admission_factor"])
        unmaterialized = sum(max(0, run.record["reserved_rss_bytes"] -
                                 self.last_memory["task_rss_bytes"].get(key, 0))
                             for key, run in self.running.items())
        growth = unmaterialized + estimate
        projected = self.last_memory["total_rss_bytes"] + growth
        round_capacity = self.memory_policy["limit_bytes"] - self.memory_policy["reserve_bytes"]
        host_capacity = self.last_memory["host"]["available_bytes"] - self.memory_policy["system_reserve_bytes"]
        evidence = {"observed_unix": self.last_memory["observed_unix"],
                    "observed_rss_bytes": self.last_memory["total_rss_bytes"],
                    "task_reserved_rss_bytes": estimate, "unmaterialized_reservations_bytes": unmaterialized,
                    "projected_rss_bytes": projected, "round_admission_capacity_bytes": round_capacity,
                    "projected_additional_bytes": growth, "host_admission_capacity_bytes": host_capacity}
        return projected <= round_capacity and growth <= host_capacity, evidence

    def advance_phase(self) -> None:
        while self.phase_index < len(self.phases):
            phase = self.phase_records[self.phase_index]
            if phase["status"] == "pending":
                phase.update(status="running", started_unix=time.time())
                self.event("phase_started", **phase)
            outstanding = [task for task in self.pending if self.task_phase[task["id"]] == phase["id"]]
            active = [run for run in self.running.values() if self.task_phase[run.task["id"]] == phase["id"]]
            if outstanding or active or self.orphans:
                return
            records = [record for record in self.records.values() if record["phase"] == phase["id"]]
            phase.update(status=("completed" if all(r["status"] == "completed" for r in records)
                                 else "completed_with_limits_or_failures"), ended_unix=time.time())
            self.event("phase_terminal", **phase)
            self.phase_index += 1

    def dependencies_ready(self, task: dict[str, Any]) -> bool:
        dependencies = dependency_ids(task)
        unfinished = {t["id"] for t in self.pending} | set(self.running)
        if any(parent in unfinished for parent in dependencies):
            return False
        # Candidate eligibility is the frozen selector's responsibility.  All
        # candidates must be terminal, but a failed candidate need not prevent
        # selecting a different, fully qualified pair.
        if "resume_select" in task:
            return True
        allowed = set(task.get("allow_parent_status", ["completed"]))
        invalid = {parent: self.records[parent]["status"] for parent in dependencies
                   if self.records[parent]["status"] not in allowed
                   or not self.records[parent].get("descendants_terminated", False)}
        if invalid:
            self.records[task["id"]].update(status="not_started", outcome_reason="dependency_failed",
                                            failed_dependencies=invalid, descendants_terminated=True)
            self.pending.remove(task)
            self.event("task_dependency_failed", id=task["id"], parents=invalid)
            return False
        return True

    def resolve_task(self, declared: dict[str, Any], directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        task = dict(declared)
        resolution: dict[str, Any] = {"declared_task_sha256": content_sha256(declared),
                                      "hash_scope": "canonical_json_sort_keys_utf8"}
        if "resume_select" in task:
            from pilotlab.selection import resolve_selection
            task, selection = resolve_selection(dict(task), self.records, self.root)
            if not isinstance(task, dict) or not isinstance(selection, dict):
                raise ValueError("selector must return a task and a selection record")
            # Selection may change the learned recipe only as declared by the
            # fixed selector.  It must not enlarge already admitted resources.
            protected = ("id", "backend", "phase", "cpu_threads", "cpu_affinity", "candidate_ids",
                         "wall_seconds", "queue_wait_seconds", "estimated_output_bytes", "estimated_rss_bytes",
                         "depends_on", "allow_parent_status")
            if any(task.get(key) != declared.get(key) for key in protected):
                raise ValueError("selector changed a scheduler resource or dependency field")
            if task.get("resume_task") not in dependency_ids(declared) or "resume_select" in task:
                raise ValueError("selector must choose a declared dependency")
            immutable_json(directory / "selection.json", selection)
            resolution["selection"] = {"path": str(directory / "selection.json"),
                                       "sha256": content_sha256(selection)}
        parent_id = task.get("resume_task")
        if parent_id is not None:
            record = self.records[parent_id]
            if (parent_id in self.running or any(t["id"] == parent_id for t in self.pending)
                    or record["status"] not in task.get("allow_parent_status", ["completed"])
                    or not record.get("descendants_terminated")):
                raise ValueError("resume parent is not an allowed, fully terminated task")
            parent_dir = self.root / "tasks" / parent_id
            result = read_json(parent_dir / "result.json")
            checkpoint = None if result is None else result.get("checkpoint")
            if not isinstance(checkpoint, dict) or result.get("status") != record.get("worker_result_status"):
                raise ValueError("resume parent has no consistent checkpoint result")
            path = Path(checkpoint.get("path", ""))
            if not path.is_absolute() or path.is_symlink():
                raise ValueError("parent checkpoint path must be an absolute regular file")
            path = path.resolve(strict=True)
            if not path.is_relative_to(parent_dir.resolve()):
                raise ValueError("parent checkpoint escapes its task directory")
            before = path.stat()
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise ValueError("parent checkpoint is empty or not a regular file")
            expected = checkpoint.get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
                raise ValueError("parent checkpoint has no valid SHA256 identity")
            observed = file_sha256(path)
            after = path.stat()
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or observed != expected.lower()
                    or checkpoint.get("bytes", before.st_size) != before.st_size):
                raise ValueError("parent checkpoint content identity mismatch")
            task.update(resume_from=str(path), resume_checkpoint_sha256=observed)
            resolution["resume_parent"] = {"task_id": parent_id, "status": record["status"],
                                            "checkpoint": str(path), "sha256": observed,
                                            "bytes": before.st_size, "step": checkpoint.get("step")}
        resolution["resolved_task_sha256"] = content_sha256(task)
        return task, resolution

    def launch(self, task: dict[str, Any], cpus: list[int], memory_admission: dict[str, Any] | None = None) -> None:
        if self.foundation_source is not None and not self.foundation_source.is_dir():
            raise FileNotFoundError(f"declared foundation source disappeared: {self.foundation_source}")
        directory = self.root / "tasks" / task["id"]
        directory.mkdir(parents=True, exist_ok=False)
        immutable_json(directory / "declared-task.json", task)
        task, resolution = self.resolve_task(task, directory)
        task_path = directory / "task.json"
        immutable_json(task_path, task)
        global_task_deadline = self.deadline - self.finish_reserve
        now = time.time()
        task_deadline = (global_task_deadline if task["backend"] == "npu" else
                         min(global_task_deadline, now + float(task.get("wall_seconds", 600))))
        queue_deadline = min(global_task_deadline, now + float(
            task.get("queue_wait_seconds", max(0, global_task_deadline - now))))
        worker = [self.python, "-u", "-m", self.worker_module,
                  "--task", str(task_path), "--data", str(self.data),
                  "--output", str(directory), "--deadline-unix", str(task_deadline),
                  "--device", "npu:0" if task["backend"] == "npu" else "cpu"]
        command = worker
        suffix = hashlib.sha256(task["id"].encode()).hexdigest()[:8]
        job_id = f"{self.round_id[:40]}-{task['id'][:60]}-{suffix}"
        if task["backend"] == "npu":
            wait_seconds = max(1, math.ceil(queue_deadline - time.time()))
            command = [self.python, "-u", str(self.queue_helper),
                       "--job-id", job_id, "--project", "tide-learning-lab",
                       "--priority", "0", "--npu-count", "1",
                       "--max-wait-seconds", str(wait_seconds),
                       "--state-file", str(directory / "queue.json")]
            if task.get("candidate_ids") is not None:
                command += ["--candidate-ids", ",".join(str(value) for value in task["candidate_ids"])]
            command += ["--", *worker]
        env = os.environ.copy()
        env.update(self.resolved_environment)
        env.update(PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS=str(len(cpus)), MKL_NUM_THREADS=str(len(cpus)),
                   OPENBLAS_NUM_THREADS=str(len(cpus)), TOKENIZERS_PARALLELISM="false",
                   PILOTLAB_ROUND_ID=self.round_id, PILOTLAB_TASK_ID=task["id"])
        if task["backend"] == "cpu":
            env["ASCEND_RT_VISIBLE_DEVICES"] = ""
        log = (directory / "worker.log").open("ab", buffering=0)
        try:
            process = subprocess.Popen(command, cwd=self.source, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       preexec_fn=lambda: os.sched_setaffinity(0, cpus))
        except BaseException:
            log.close()
            raise
        now = time.time()
        record = {"id": task["id"], "backend": task["backend"],
                  "phase": self.task_phase[task["id"]], "depends_on": dependency_ids(task),
                  "task_resolution": resolution,
                  "foundation_source": None if self.foundation_source is None else str(self.foundation_source),
                  "resolved_environment": dict(self.resolved_environment),
                  "task_file_sha256": file_sha256(task_path),
                  "requested_cpu_affinity": task.get("cpu_affinity"),
                  "requested_candidate_ids": task.get("candidate_ids"),
                  "status": "queued" if task["backend"] == "npu" else "running",
                  "process_started_unix": now, "worker_started_unix": now if task["backend"] == "cpu" else None,
                  "deadline_unix": task_deadline, "cpu_affinity": cpus,
                  "supplied_deadline_unix": task_deadline,
                  "queue_deadline_unix": queue_deadline if task["backend"] == "npu" else None,
                  "queue_admitted_unix": None,
                  "pid": process.pid, "pgid": process.pid, "process_identity": proc_identity(process.pid),
                  "command": command, "directory": str(directory),
                  "estimated_output_bytes": int(task.get("estimated_output_bytes", 2 * GIB)),
                  "estimated_rss_bytes": task.get("estimated_rss_bytes", 4 * GIB),
                  "reserved_rss_bytes": math.ceil(task.get("estimated_rss_bytes", 4 * GIB) * self.memory_policy["admission_factor"]),
                  "memory_admission": memory_admission}
        self.records[task["id"]] = record
        run = Running(task, directory, process, log, cpus, task_deadline, record)
        if record["process_identity"] is not None:
            run.descendants[process.pid] = record["process_identity"]
            self.round_descendants[process.pid] = record["process_identity"]
        self.running[task["id"]] = run
        atomic_json(directory / "process.json", record)
        self.event("task_dispatched", id=task["id"], pid=process.pid, cpu_affinity=cpus,
                   deadline_unix=task_deadline)

    def signal_task(self, run: Running, sig: int, table: dict[int, dict[str, Any]]) -> None:
        # Once a wrapper has been reaped, its former PGID can be reused.  Only
        # signal the group while a saved, live task identity still anchors it.
        if any(same_live(identity, table) and table[identity["pid"]]["pgrp"] == run.process.pid
               for identity in run.descendants.values()):
            try:
                os.killpg(run.process.pid, sig)
            except ProcessLookupError:
                pass
        # A known descendant that created its own process group still belongs
        # to this task. Verify the saved start identity before signalling it.
        for identity in run.descendants.values():
            if same_live(identity, table) and table[identity["pid"]]["pgrp"] != run.process.pid:
                try:
                    os.kill(identity["pid"], sig)
                except ProcessLookupError:
                    pass

    def update_running(self) -> None:
        table = process_table()
        self.observe_processes(table)
        self.cleanup_orphans(table)
        now = time.time()
        for key, run in list(self.running.items()):
            try:
                queue = read_json(run.directory / "queue.json")
                worker = read_json(run.directory / "worker.json")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                run.record["observation_error"] = str(exc)
                queue = worker = None
            if queue:
                run.record["queue"] = queue
                if queue.get("status") == "running" and run.record["queue_admitted_unix"] is None:
                    run.record["queue_admitted_unix"] = unix_timestamp(queue.get("timestamp")) or now
                    run.record["status"] = "running"
                    self.event("task_admitted", id=key, queue=queue)
                if queue.get("status") == "timed_out":
                    run.termination_reason = run.termination_reason or "queue_deadline"
            if worker and unix_timestamp(worker.get("started_unix")) is not None:
                run.record["worker_started_unix"] = unix_timestamp(worker["started_unix"])
                run.record["worker_start_source"] = "worker"
                run.record["status"] = "running"
                if run.task["backend"] == "npu":
                    # Device waiting belongs to the round budget.  The worker's
                    # own runtime budget starts only once the admitted Python
                    # process enters its recorded entry point.
                    run.deadline = min(self.deadline - self.finish_reserve,
                                       run.record["worker_started_unix"] + float(run.task.get("wall_seconds", 600)))
                    run.record["deadline_unix"] = run.deadline
            # A finished child must be reaped before deciding it needs a
            # deadline signal; otherwise a polling delay can turn an on-time
            # completed result into a false budget exhaustion.
            return_code = run.process.poll()
            reason = self.stop_reason or ("task_deadline" if now >= run.deadline else None)
            if (not reason and run.task["backend"] == "npu"
                    and run.record["worker_started_unix"] is None
                    and run.record["queue_admitted_unix"] is None
                    and now >= run.record["queue_deadline_unix"]):
                reason = "queue_deadline"
            if now >= self.deadline:
                reason = reason or "round_deadline"
            if reason and run.term_sent is None and return_code is None:
                run.termination_reason = reason
                run.term_sent = now
                self.signal_task(run, signal.SIGTERM, table)
                self.event("task_term_sent", id=key, reason=reason)
            return_code = run.process.poll()
            live = [identity for identity in run.descendants.values() if same_live(identity, table)]
            if return_code is not None:
                # poll() may have reaped the leader after the table snapshot;
                # refresh the few recorded descendants before calling them orphans.
                fresh = {identity["pid"]: proc_identity(identity["pid"])
                         for identity in live if identity["pid"] != run.process.pid}
                fresh = {pid: row for pid, row in fresh.items() if row is not None}
                live = [identity for identity in live if same_live(identity, fresh)]
            if return_code is not None and live and run.term_sent is None:
                run.termination_reason = "orphaned_descendants"
                run.term_sent = now
                self.signal_task(run, signal.SIGTERM, table)
                self.event("orphan_cleanup", id=key, live=live)
            if run.term_sent is not None and live and now - run.term_sent >= self.grace and run.kill_sent is None:
                self.signal_task(run, signal.SIGKILL, table)
                run.kill_sent = now
                self.event("task_kill_sent", id=key, live=live)
            run.record["descendant_identities"] = list(run.descendants.values())
            if return_code is None or live:
                continue
            result_error = None
            try:
                result = read_json(run.directory / "result.json")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                result, result_error = None, str(exc)
            status, why = task_outcome(return_code, result, run.termination_reason)
            if result and unix_timestamp(result.get("started_unix")) is not None:
                run.record["worker_started_unix"] = unix_timestamp(result["started_unix"])
                run.record["worker_start_source"] = "worker_result"
            run.record.update(status=status, outcome_reason=why, return_code=return_code,
                              process_ended_unix=now, worker_result_status=None if result is None else result.get("status"),
                              worker_ended_unix=None if result is None else unix_timestamp(result.get("ended_unix")),
                              descendants_terminated=True, termination_reason=run.termination_reason,
                              term_sent_unix=run.term_sent, kill_sent_unix=run.kill_sent)
            if run.record["worker_started_unix"] is not None:
                run.record["launch_to_worker_seconds"] = run.record["worker_started_unix"] - run.record["process_started_unix"]
                if run.record["worker_ended_unix"] is not None:
                    run.record["worker_elapsed_seconds"] = run.record["worker_ended_unix"] - run.record["worker_started_unix"]
            if result_error:
                run.record["result_read_error"] = result_error
            atomic_json(run.directory / "process.json", run.record)
            run.log.close()
            del self.running[key]
            self.event("task_terminal", id=key, status=status, reason=why, return_code=return_code)

    def dispatch(self) -> None:
        self.advance_phase()
        if self.stop_reason or time.time() >= self.dispatch_deadline:
            return
        if self.last_storage is None or self.last_memory is None or self.orphans:
            return
        while self.phase_index < len(self.phases):
            phase_index = self.phase_index
            phase = self.phases[phase_index]
            before_pending = len(self.pending)
            counts = {backend: sum(r.task["backend"] == backend for r in self.running.values())
                      for backend in ("cpu", "npu")}
            reservations = sum(r.record["estimated_output_bytes"] for r in self.running.values())
            for task in list(self.pending):
                if self.task_phase[task["id"]] != phase["id"]:
                    continue
                if not self.dependencies_ready(task):
                    continue
                backend, threads = task["backend"], task.get("cpu_threads", 8)
                estimate = int(task.get("estimated_output_bytes", 2 * GIB))
                if counts[backend] >= phase[backend + "_slots"]:
                    continue
                free = [cpu for cpu in self.affinity if cpu not in self.allocated_cpus()]
                fixed = task.get("cpu_affinity")
                if fixed is not None:
                    if not set(fixed) <= set(free):
                        continue
                    cpus = list(fixed)
                else:
                    reserved = {cpu for pending in self.pending
                                if self.task_phase[pending["id"]] == phase["id"]
                                for cpu in pending.get("cpu_affinity", [])}
                    automatic_free = [cpu for cpu in free if cpu not in reserved]
                    if threads > len(automatic_free):
                        continue
                    cpus = automatic_free[:threads]
                if estimate + reservations > self.last_storage["available_for_admission_bytes"]:
                    continue
                memory_ok, memory_evidence = self.memory_admission(task)
                if not memory_ok:
                    self.records[task["id"]]["memory_admission_blocked"] = memory_evidence
                    continue
                if self.stop_reason or time.time() >= self.dispatch_deadline:
                    return
                try:
                    self.launch(task, cpus, memory_evidence)
                except Exception as exc:
                    if task["id"] in self.running:
                        self.stop_reason = self.stop_reason or "launch_record_failure"
                    else:
                        self.records[task["id"]].update(status="failed", outcome_reason="launch_failure",
                                                        error=str(exc), descendants_terminated=True)
                    self.event("launch_failed", id=task["id"], error=str(exc))
                else:
                    reservations += estimate
                    counts[backend] += 1
                self.pending.remove(task)
                if self.stop_reason:
                    return
            self.advance_phase()
            if self.phase_index != phase_index:
                continue
            # Propagate failed dependencies even if plan order lists children
            # before parents.  A barrier does not depend on task list ordering.
            if not self.running and len(self.pending) < before_pending:
                continue
            return

    def run(self) -> int:
        old_handlers = {s: signal.signal(s, self.request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        fatal: str | None = None
        previous_subreaper: bool | None = None
        try:
            previous_subreaper = child_subreaper(True)
            self.event("subreaper_enabled", previous=previous_subreaper)
            self.persist()
            while self.pending or self.running or self.orphans:
                now = time.time()
                if now >= self.deadline:
                    self.stop_reason = self.stop_reason or "round_deadline"
                if time.monotonic() - self.storage_checked >= float(self.plan.get("storage_poll_seconds", 15)):
                    try:
                        self.last_storage = storage_snapshot(self.root, self.plan)
                    except Exception as exc:
                        self.stop_reason = self.stop_reason or "quota_unavailable"
                        fatal = str(exc)
                        self.event("storage_probe_failed", error=fatal)
                    else:
                        if self.last_storage["available_for_admission_bytes"] < 0:
                            self.stop_reason = self.stop_reason or "storage_budget"
                    self.storage_checked = time.monotonic()
                self.update_running()
                self.monitor_memory()
                if self.stop_reason and (self.running or self.orphans):
                    self.update_running()
                self.dispatch()
                if not self.running and not self.orphans:
                    if not self.pending:
                        break
                    if self.stop_reason or time.time() >= self.dispatch_deadline:
                        break
                    # No task can be admitted with all slots/CPUs free: waiting
                    # cannot fix an oversized declared output requirement.
                    self.stop_reason = "admission_capacity"
                    break
                if time.monotonic() - self.last_persist >= min(5.0, self.poll * 5):
                    self.persist()
                time.sleep(self.poll)
        except BaseException as exc:
            fatal = f"{type(exc).__name__}: {exc}"
            self.stop_reason = self.stop_reason or "scheduler_error"
            self.event("scheduler_error", error=fatal)
        finally:
            # Even an exception or SIGTERM must finish process-group cleanup.
            # A final full scan also catches a child which detached just as its
            # wrapper exited.  Subreaping keeps such children attributable to
            # this scheduler rather than letting them escape to init.
            self.monitor_memory(force=True)
            self.update_running()
            while self.running or self.orphans:
                self.stop_reason = self.stop_reason or "scheduler_error"
                self.update_running()
                if self.running or self.orphans:
                    self.persist()
                    time.sleep(min(self.poll, 0.25))
            for task in self.pending:
                self.records[task["id"]].update(status="not_started", outcome_reason=self.stop_reason or "dispatch_cutoff",
                                                descendants_terminated=True)
            self.pending.clear()
            for phase in self.phase_records:
                if phase["status"] in {"pending", "running"}:
                    phase.update(status="not_started" if phase["status"] == "pending" else "stopped",
                                 ended_unix=time.time(), outcome_reason=self.stop_reason or "dispatch_cutoff")
            fatal = fatal or self.memory_error
            if fatal:
                atomic_json(self.root / "scheduler-error.json", {"error": fatal, "timestamp": utc_now()})
            if self.stop_reason in {"signal", "round_deadline"}:
                status = "budget_exhausted"
            elif self.stop_reason and self.stop_reason != "admission_capacity":
                status = "failed"
            elif any(r["status"] == "failed" for r in self.records.values()):
                status = "completed_with_failures"
            elif any(r["status"] != "completed" for r in self.records.values()):
                status = "completed_with_limits"
            else:
                status = "completed"
            self.event("round_terminal", status=status, stop_reason=self.stop_reason)
            self.persist(status)
            if previous_subreaper is not None:
                child_subreaper(previous_subreaper)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        return 1 if status in {"failed", "completed_with_failures"} else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output", "data", "python", "source"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--foundation-source", help="explicit frozen foundation import root; must agree with plan.foundation_source")
    p.add_argument("--duration-seconds", type=float, default=3600)
    p.add_argument("--npu-slots", type=int, default=16)
    p.add_argument("--cpu-slots", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return Scheduler(args).run()
    except Exception as exc:
        print(f"scheduler setup failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
