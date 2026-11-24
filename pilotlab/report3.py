"""Read-only observation of a round-three scheduler and its recorded evidence.

No training, gate creation, device initialization or checkpoint deserialization
is performed. JSON is printed unless an explicit new --output file is supplied.
This report preserves separate execution, exposure, health, state-loading,
trajectory and learning observations; it is not a fresh scientific gate audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .lifecycle3 import cancelled_before_allocation, empty_dispatch_ticket_closed


MI = 2**20
FORMAL_TOKENS = 12*MI
RECIPE_FIELDS = ("model_id", "model", "seed", "recipe_id", "init_recipe", "lr", "batch_size",
                 "sequence_length", "stream_chunks", "cpu_threads", "precision", "deterministic",
                 "schedule_tokens", "warmup_tokens", "lr_floor_ratio", "weight_decay", "grad_clip",
                 "auxiliary_loss_weight", "source_identity")
TERMINAL = {"completed", "failed", "recipe_rejected", "stopped_budget", "interrupted", "budget_exhausted"}


def number(value, *, nonnegative=True):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def process_observation(identity):
    """Probe only an explicitly recorded PID, protecting against PID reuse."""
    if not isinstance(identity, dict) or not isinstance(identity.get("pid"), int):
        return {"alive": None, "reason": "no recorded process identity"}
    try:
        path = Path(f'/proc/{identity["pid"]}')
        raw = (path/"stat").read_text()
        fields = raw[raw.rfind(")")+2:].split()
        same = int(fields[19]) == identity.get("start_ticks")
        return {"pid": identity["pid"], "alive": same and fields[0] != "Z", "same_start_ticks": same,
                "state": fields[0], "cgroup": (path/"cgroup").read_text().strip() if same else None}
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": identity["pid"], "alive": False, "reason": "recorded PID absent"}
    except (OSError, ValueError, IndexError) as exc:
        return {"pid": identity["pid"], "alive": None, "reason": f"process observation unavailable: {type(exc).__name__}"}


class Snapshot:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.errors = []
        self.cache = {}
        self.hashes = {}
        self.external_attempts = set()
        self.external_files = set()

    def error(self, path, reason):
        item = {"path": str(path), "reason": str(reason)}
        if item not in self.errors:
            self.errors.append(item)

    def path(self, value, *, inside=True):
        path = Path(value)
        path = (path if path.is_absolute() else self.root/path).resolve()
        if (inside and not path.is_relative_to(self.root) and path not in self.external_files
                and path not in self.external_attempts and self.external_attempts.isdisjoint(path.parents)):
            self.error(path, "artifact reference leaves the requested round")
            return None
        return path

    def read(self, value, *, required=False, inside=True):
        path = self.path(value, inside=inside)
        if path is None:
            return None
        if path in self.cache:
            return self.cache[path]
        try:
            raw = path.read_bytes()
            result = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(result, dict):
                raise ValueError("JSON root is not an object")
            self.hashes[path] = hashlib.sha256(raw).hexdigest()
        except FileNotFoundError:
            if required:
                self.error(path, "missing required artifact")
            result = None
        except (OSError, ValueError, TypeError) as exc:
            self.error(path, f"unreadable artifact: {type(exc).__name__}: {exc}")
            result = None
        self.cache[path] = result
        return result

    def bound(self, value, expected, *, inside=True):
        path = self.path(value, inside=inside)
        self.read(value, required=True, inside=inside)
        return bool(expected and path is not None and self.hashes.get(path) == expected)

    def read_import_reference(self, reference):
        """Validate each immutable reference against the bytes in this report."""
        path = Path(reference["path"]).resolve()
        value = self.read(path, required=True, inside=False)
        if value is None or self.hashes[path] != reference["sha256"]:
            raise ValueError(f"resume_reference_changed:{path}")
        return value


def union_length(intervals):
    end = None
    length = 0
    for start, stop in sorted(intervals):
        if end is None or start > end:
            length += stop-start
            end = stop
        elif stop > end:
            length += stop-end
            end = stop
    return length


def concurrency(intervals):
    """Half-open allocation intervals; releases precede simultaneous acquires."""
    events = defaultdict(int)
    for start, end in intervals:
        if end > start:
            events[start] += 1
            events[end] -= 1
    active = peak = 0
    for _, delta in sorted(events.items()):
        active += delta
        peak = max(peak, active)
    seconds = sum(end-start for start, end in intervals)
    span = max((end for _, end in intervals), default=0)-min((start for start, _ in intervals), default=0)
    return {"peak": peak, "device_seconds": seconds, "device_hours": seconds/3600,
            "allocation_window_seconds": span, "any_device_allocated_seconds": union_length(intervals),
            "mean_over_allocation_window": seconds/span if span else None}


def queue_intervals(snapshot, directory, attempt, queue, *, now):
    """Read acquisition/release events, including attempts between poll ticks.

    A live, still-allocated wrapper gives a censored interval ending at `now`.
    A terminal attempt without release does not get an invented end timestamp.
    Its conservative scheduler debit remains available in the resource ledger.
    """
    if "runq_id" in attempt:
        if not queue.get("started_unix") and attempt.get("status") == "starting":
            return [], False, []
        begin, end = queue.get("started_unix"), queue.get("finished_unix")
        bound = (type(queue.get("lease_id")) is int and queue.get("job_id") == attempt["runq_id"]
                 and attempt.get("process") and queue.get("process") == attempt.get("process")
                 and queue.get("unit") == attempt.get("unit") and queue.get("physical_ids") == [attempt.get("physical_id")])
        if bound and number(begin):
            if number(end) and end >= begin:
                return [(begin,end)], False, []
            if queue.get("status") == "running" and process_observation(queue["process"])["alive"] is True:
                return [(begin,now)], True, []
            if attempt.get("status") == "running" and queue.get("status") == "running":
                return [], False, []  # publishing the independent worker's exit receipt
        snapshot.error(directory/"queue.json", "runq allocation or process binding incomplete")
        return [], False, ["incomplete allocation"]
    path = directory/"stdout.log"
    intervals, problems, acquired = [], [], None
    try:
        with path.open() as stream:
            for line in stream:
                if '"ascend_queue"' not in line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event") != "ascend_queue":
                        continue
                    status = event.get("status")
                    if status not in {"running", "completed", "failed", "cancelled"}:
                        continue
                    stamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                    if stamp.tzinfo is None:
                        raise ValueError("queue timestamp has no timezone")
                    value = stamp.timestamp()
                    if not number(value):
                        raise ValueError("invalid timestamp")
                    if status == "running":
                        if acquired is None:
                            acquired = value
                        elif value != acquired:
                            problems.append("repeated acquire without release")
                    elif acquired is not None:
                        if value < acquired:
                            problems.append("release precedes acquire")
                        else:
                            intervals.append((acquired, value))
                        acquired = None
                    elif not (status == "cancelled" and cancelled_before_allocation(directory, queue)):
                        problems.append("release has no acquire")
                except (ValueError, KeyError, TypeError, AttributeError):
                    problems.append("invalid queue event")
    except FileNotFoundError:
        if queue.get("status") in {"running", "completed", "failed", "cancelled"}:
            problems.append("queue event log missing")
    except OSError as exc:
        problems.append(f"queue log unavailable: {type(exc).__name__}")
    open_interval = False
    if acquired is not None:
        alive = process_observation(attempt.get("process"))["alive"]
        if queue.get("status") == "running" and alive is True and now >= acquired:
            intervals.append((acquired, now))
            open_interval = True
        else:
            problems.append("acquire has no verified release or live allocation")
    if attempt.get("accounting_estimated"):
        problems.append("scheduler device accounting is estimated")
    if not intervals and (queue.get("status") == "completed" or attempt.get("status") == "completed"):
        problems.append("completed queue has no complete allocation interval")
    for reason in sorted(set(problems)):
        snapshot.error(path, reason)
    return intervals, open_interval, sorted(set(problems))


def inspect_gate(snapshot, filename):
    gate = snapshot.read(filename)
    if gate is None:
        return {"status": "absent", "binding_verified": False}
    reasons = []
    if gate.get("status") != "passed":
        reasons.append("gate is not passed")
    for kind in ("manifest", "report"):
        path = gate.get(kind+"_path")
        if not path or not snapshot.bound(path, gate.get(kind+"_sha256")):
            reasons.append(f"{kind} artifact hash mismatch or unavailable")
    report = snapshot.read(gate["report_path"]) if gate.get("report_path") else None
    if not report or report.get("status") != "passed" or report.get("blocking_reasons"):
        reasons.append("referenced stage A audit did not pass")
    manifest = snapshot.read(gate["manifest_path"]) if gate.get("manifest_path") else None
    if (not gate.get("source_identity") or not manifest or not report or
            gate["source_identity"] != manifest.get("source_identity") or
            gate["source_identity"] != report.get("source_identity")):
        reasons.append("gate/source identity mismatch")
    return {"status": gate.get("status"), "binding_verified": not reasons,
            "gate_kind": gate.get("gate_kind"), "allowed_task_count": len(gate.get("allowed_task_ids", [])),
            "reasons": reasons, "scope": "saved gate and JSON artifact bindings; no checkpoint decoding or gate rerun"}


def exposure_category(phase):
    if phase in {"b", "b_repeat", "b_confirmation"}:
        return "formal"
    if phase in {"a1", "a2", "a3"}:
        return "tuning"
    return {"init": "initialization", "preflight": "preparation"}.get(phase, "other")


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def worker_interval(record, *, now):
    start, end = record.get("started_unix"), record.get("finished_unix")
    if not number(start):
        return None, False
    if number(end) and end >= start:
        return (start, end), False
    identity = record.get("process") or {"pid": record.get("pid"), "start_ticks": record.get("start_ticks")}
    if record.get("status") == "running" and process_observation(identity)["alive"] is True and now >= start:
        return (start, now), True
    return None, False


def worker_concurrency(intervals, *, missing=0):
    value = concurrency(intervals)
    return {"peak": value["peak"], "worker_seconds": value["device_seconds"],
            "worker_window_seconds": value["allocation_window_seconds"],
            "any_worker_running_seconds": value["any_device_allocated_seconds"],
            "mean_over_worker_window": value["mean_over_allocation_window"],
            "missing_timing_count": missing, "timing_complete": missing == 0,
            "scope": "worker process intervals; distinct from physical NPU lease concurrency"}


def shared_admission_interval(snapshot, directory, group, span):
    """Separate successful admission from the earlier conservative debit start.

    The group release timestamp is captured immediately after unlocking. The
    terminal queue log can be later while waiting to remove the FIFO entry, so
    it is not the physical lease end.
    """
    if span is None:
        return None, False
    path = directory / "stdout.log"
    try:
        events = []
        with path.open() as stream:
            for line in stream:
                if '"ascend_queue"' not in line:
                    continue
                event = json.loads(line)
                if event.get("event") == "ascend_queue" and event.get("status") == "running":
                    events.append(event)
        if not events:
            return None, False
        if len(events) != 1:
            raise ValueError("shared queue must have one successful admission")
        event = events[0]
        stamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        start = stamp.timestamp()
        if (stamp.tzinfo is None or not number(start) or not span[0] <= start <= span[1]
                or event.get("job_id") != group.get("job_id")
                or event.get("physical_ids") != [group.get("physical_id")]):
            raise ValueError("shared admission identity or interval mismatch")
        conservative = (event.get("accounting_scope") == "one device; includes complete successful admission attempt"
            and number(event.get("accounting_started_unix"))
            and math.isclose(event["accounting_started_unix"], span[0], abs_tol=1e-6, rel_tol=0))
        return (start, span[1]), conservative
    except FileNotFoundError:
        return None, False
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        snapshot.error(path, f"shared queue admission unavailable: {type(exc).__name__}: {exc}")
        return None, False


def shared_group_observations(snapshot, state, *, now):
    observations = {}
    for group_id, entry in state.get("shared_groups", {}).items():
        directory = snapshot.path(entry["path"])
        if directory is None:
            continue
        path = entry.get("manifest_path", directory / "manifest.json")
        manifest = snapshot.read(path, required=True) or {}
        group = snapshot.read(directory / "group.json", required=True) or {}
        reasons = []
        if not snapshot.bound(path, entry.get("manifest_sha256")):
            reasons.append("shared group manifest hash mismatch")
        if (not group or group.get("schema") != "pilot-r3-shared-group-state-v1"
                or group.get("job_id") != entry.get("job_id")
                or group.get("manifest_sha256") != entry.get("manifest_sha256")
                or group.get("backend") != manifest.get("backend")
                or group.get("backend") != entry.get("backend")):
            reasons.append("shared group identity mismatch")
        declared_workers = manifest.get("workers", [])
        acknowledged_workers = declared_workers
        admission = None
        if manifest.get("dynamic_admission"):
            try:
                from .shared_npu3 import read_admission_chain
                admission = read_admission_chain(manifest, Path(path), group)
                declared_workers = admission["published_workers"]
                acknowledged_workers = admission["acknowledged_workers"]
            except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
                reasons.append(f"dynamic admission evidence invalid: {error}")
        declared = [row.get("id") for row in declared_workers]
        acknowledged = [row.get("id") for row in acknowledged_workers]
        owner_alive = process_observation(group.get("owner", entry.get("process")))["alive"] is True
        updating = bool(manifest.get("dynamic_admission") and owner_alive
                        and state.get("status") not in TERMINAL | {"completed", "a_blocked_needs_repair"})
        if (len(set(declared)) != len(declared) or set(acknowledged) != set(group.get("workers", {}))
                or not updating and set(declared) != set(entry.get("task_ids", []))):
            reasons.append("shared group worker list mismatch")
        span, opened = None, False
        acquired, released = group.get("allocation_acquired_unix"), group.get("allocation_released_unix")
        if group.get("backend") == "npu":
            if number(acquired) and number(released) and released >= acquired:
                span = (acquired, released)
            elif number(acquired) and released is None and owner_alive:
                if now >= acquired:
                    span, opened = (acquired, now), True
            elif acquired is not None or released is not None or group.get("status") == "completed":
                if not (entry.get("dispatch_lease") and empty_dispatch_ticket_closed(
                        entry, group, snapshot.read(directory/"queue.json", required=True) or {})):
                    reasons.append("shared lease acquire has no verified release or live owner")
            if span and not opened and (not number(group.get("physical_device_seconds"))
                    or not math.isclose(group["physical_device_seconds"], span[1]-span[0], rel_tol=1e-6, abs_tol=.1)):
                reasons.append("shared physical accounting disagrees with acquire/release interval")
            if span and (type(group.get("physical_id")) is not int or group["physical_id"] < 0):
                reasons.append("shared physical accounting lacks a valid device id")
        elif acquired is not None or released is not None or group.get("physical_device_seconds", 0) != 0:
            reasons.append("CPU shared group has nonzero physical device accounting")
        for reason in reasons:
            snapshot.error(directory / "group.json", reason)
        admitted, conservative = shared_admission_interval(snapshot, directory, group, span if not reasons else None)
        observations[group_id] = {"id": group_id, "path": str(directory), "entry": entry, "manifest": manifest,
            "group": group, "interval": span if not reasons else None, "open_interval": opened,
            "admitted_interval": admitted, "conservative_admission": conservative, "owner_alive": owner_alive,
            "binding_verified": not reasons, "errors": reasons,
            "admission": admission, "updating": updating,
            "workers": group.get("workers", {}), "declared_workers": {row["id"]: row for row in declared_workers}}
    return observations


def build_report(round_path, *, now=None):
    now = time.time() if now is None else now
    snapshot = Snapshot(round_path)
    state = snapshot.read("scheduler.json", required=True) or {}
    if state and state.get("schema") != "pilot-r3-scheduler-v1":
        snapshot.error("scheduler.json", "unsupported scheduler schema")
    config = state.get("config", {})
    imports = {}
    if config.get("resume_import"):
        try:
            imported = snapshot.read_import_reference(config["resume_import"])
            snapshot.read_import_reference(imported["reconciliation"])
            if imported.get("schema") != "pilot-r3-resume-import-v1":
                raise ValueError("unsupported resume import")
            imports = imported["tasks"]
            for entry in imports.values():
                snapshot.external_files.add(Path(entry["task"]["path"]).resolve())
                if entry.get("attempt_path"):
                    snapshot.external_attempts.add(Path(entry["attempt_path"]).resolve())
                if entry.get("result"):
                    snapshot.external_attempts.add(Path(entry["result"]["path"]).resolve().parent)
                for previous in entry.get("attempt_evidence", []):
                    snapshot.external_attempts.add(Path(previous["attempt_path"]).resolve())
                    snapshot.external_files.add(Path(previous["task"]["path"]).resolve())
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            snapshot.error(config["resume_import"]["path"], f"resume import unavailable: {error}")
    manifest = snapshot.read("model-manifest.json") or {}
    selected = snapshot.read("selected.json") or {}
    final_summary = snapshot.read("report.json") or {}
    stage_a = snapshot.read("stage-a-report.json") or {}
    ledger = snapshot.read(config["budget_ledger"], inside=False) if config.get("budget_ledger") else None
    shared = shared_group_observations(snapshot, state, now=now)
    gates = {name: inspect_gate(snapshot, filename) for name, filename in (
        ("formal", "stage-b-gate.json"), ("repeat", "stage-b-repeat-gate.json"),
        ("confirmation", "stage-b-confirmation-gate.json"))}
    subjects = {item["id"] for item in manifest.get("subjects", []) if isinstance(item, dict) and "id" in item}
    groups = {item["id"]: item.get("group") for item in manifest.get("subjects", []) if isinstance(item, dict) and "id" in item}
    rows, exposure, physical = [], defaultdict(list), Counter()
    by_card, admitted_by_card, conservative_by_card = defaultdict(list), defaultdict(list), defaultdict(list)
    runq_intervals, runq_open = defaultdict(list), set()
    logical_intervals = {"cpu": [], "npu": []}
    logical_missing = Counter()
    backend_tokens, backend_seconds = Counter(), Counter()
    backend_timing_complete = {"cpu": True, "npu": True}
    open_allocations = 0
    attempted_npu = 0
    allocation_count = 0
    for group_id, observed in shared.items():
        group = observed["group"]
        if group.get("backend") == "npu":
            attempted_npu += 1
            if observed["interval"]:
                card = ("physical", group.get("physical_id"))
                by_card[card].append(observed["interval"])
                conservative_by_card[card].append(observed["conservative_admission"])
                if observed["admitted_interval"]:
                    admitted_by_card[card].append(observed["admitted_interval"])
                allocation_count += 1
                open_allocations += int(observed["open_interval"])
    seconds = npu_loss_tokens = 0
    timing_complete = True
    final_results = {}
    discovered_subjects = set(subjects)

    for task_id, record in state.get("tasks", {}).items():
        task_path = record.get("task_path", f"tasks/{task_id}/task.json")
        task = snapshot.read(task_path, required=bool(record.get("attempts"))) or {}
        phase = task.get("phase", "unknown")
        category = exposure_category(phase)
        model_id = task.get("model_id", record.get("model_id"))
        family = task.get("model", {}).get("family")
        if family == "tide" and model_id:
            discovered_subjects.add(model_id)
        requested_device = task.get("device", record.get("device", "npu:0"))
        is_npu = str(requested_device).startswith("npu")
        backend = "unbound" if requested_device == "unbound" else "npu" if is_npu else "cpu"
        latest_result = None
        latest_integrity = []
        latest_pending = False
        attempts = []
        attempt_records = [*record.get("prior_attempts", []), *record.get("attempts", [])]
        for index, attempt in enumerate(attempt_records):
            inherited = index < len(record.get("prior_attempts", []))
            directory = snapshot.path(attempt.get("path", f"tasks/{task_id}/attempt-{index}"))
            if directory is None:
                continue
            group_id = None if inherited else attempt.get("shared_group") or (record.get("shared_group") if index == len(attempt_records)-1 else None)
            grouped = shared.get(group_id) if group_id else None
            group_worker, group_binding, group_complete = {}, False, False
            import_row = imports.get(task_id, {}) if inherited else {}
            if inherited and import_row.get("attempt_evidence") is not None:
                try:
                    from .resume3 import imported_attempt
                    import_row, _, _ = imported_attempt(config, task_id, str(directory), read_bound=snapshot.read_import_reference)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    snapshot.error(directory, f"imported attempt binding invalid: {error}")
                    import_row = {}
            worker_complete = False
            admission_pending = False
            if inherited:
                # A separately bound terminal reconciliation replaces no
                # historical bytes. It establishes artifact provenance and
                # release for adoption, while original failed statuses remain.
                queue = {"status": "completed" if import_row.get("disposition") == "completed" else "failed",
                         "return_code": import_row.get("worker_return_code")}
            elif group_id:
                if grouped is None:
                    snapshot.error(directory, "shared worker references an unavailable group")
                else:
                    group_worker = grouped["workers"].get(task_id, {})
                    declared = grouped["declared_workers"].get(task_id, {})
                    admission_pending = bool(grouped.get("updating") and not group_worker
                        and record.get("status") == "running" and attempt.get("status") in {"admitting", "starting", "running"})
                    group_binding = (grouped["binding_verified"] and declared.get("output") == str(directory)
                        and (group_worker.get("output") == str(directory) or admission_pending)
                        and grouped["group"].get("backend") == backend
                        and (not attempt.get("command") or declared.get("argv") == attempt["command"]))
                    if not group_binding and not admission_pending:
                        snapshot.error(directory, "shared worker output/command/backend identity mismatch")
                    group_complete = (group_binding and grouped["group"].get("remaining_descendants") == []
                        and grouped["group"].get("status") in {"completed", "failed", "stopped_budget"}
                        and group_worker.get("status") == "completed" and group_worker.get("return_code") == 0
                        and (not is_npu or grouped["interval"] is not None and not grouped["open_interval"]))
                    worker_complete = group_complete
                    if grouped["manifest"].get("dynamic_admission"):
                        from .shared_npu3 import worker_spec_digest
                        worker_complete = (group_binding and group_worker.get("status") == "completed"
                            and group_worker.get("return_code") == 0
                            and group_worker.get("descendant_cleanup_verified") is True
                            and group_worker.get("remaining_descendants") == []
                            and group_worker.get("worker_spec_sha256") == worker_spec_digest(declared)
                            and process_observation(group_worker.get("process"))["alive"] is False)
                queue = {"status": (grouped or {}).get("group", {}).get("status"),
                         "return_code": group_worker.get("return_code")}
            else:
                queue = snapshot.read(directory/"queue.json", required=is_npu and attempt.get("status") == "completed") or {}
            if is_npu and not group_id and not inherited:
                attempted_npu += 1
                measured, opened, _ = queue_intervals(snapshot, directory, attempt, queue, now=now)
                ids = queue.get("physical_ids", [])
                card = ("physical", ids[0]) if len(ids) == 1 else ("single", task_id, index)
                if "runq_id" in attempt and measured:
                    key = (card, queue["lease_id"])
                    runq_intervals[key].extend(measured)
                    if opened:
                        runq_open.add(key)
                else:
                    by_card[card].extend(measured)
                    admitted_by_card[card].extend(measured)
                    conservative_by_card[card].extend(False for _ in measured)
                    allocation_count += len(measured)
                    open_allocations += int(opened)
            result_path = directory/"result.json"
            terminal = attempt.get("status") in TERMINAL
            never_started = import_row.get("disposition") == "not_started" if inherited else (group_binding and group_worker.get("status") == "not_started"
                and group_worker.get("pid") is None and group_worker.get("started_unix") is None
                and not any((directory / name).exists() for name in ("result.json", "heartbeat.json", "checkpoint.json", "failure.pt"))
                and not any((directory / "checkpoints").glob("*.pt"))) if group_id else (
                is_npu and cancelled_before_allocation(directory, queue))
            result = snapshot.read(result_path, required=terminal and not never_started)
            integrity = []
            publication_pending = (bool(result) and group_binding and grouped["owner_alive"]
                and grouped["group"].get("status") in {"running", "cleanup_blocked"}
                and group_worker.get("status") in {"pending", "starting", "running", "exited_waiting_cleanup"}
                and not group_worker.get("result_sha256")
                and record.get("status") != "completed" and not terminal) if group_id else False
            timing_record = group_worker if group_id else (result or {})
            span, span_open = worker_interval(timing_record, now=now)
            if span and (not group_id or group_binding):
                logical_intervals[backend].append(span)
            elif not never_started and (result or group_worker.get("started_unix") is not None):
                logical_missing[backend] += 1
            if result is not None:
                if inherited and (not import_row.get("result") or not snapshot.bound(result_path, import_row["result"]["sha256"])
                        or result.get("task_sha256") != import_row.get("task", {}).get("sha256")):
                    integrity.append("imported result or original task binding mismatch")
                if group_id and not publication_pending and (not group_binding or group_worker.get("result_sha256") != snapshot.hashes.get(result_path)):
                    integrity.append("shared worker result binding mismatch")
                if result.get("id") != task.get("id", task_id) or result.get("model_id") != model_id:
                    integrity.append("task/model identity mismatch")
                if not task.get("source_identity") or result.get("source_identity") != task.get("source_identity"):
                    integrity.append("source identity mismatch")
                if not snapshot.bound(task_path, result.get("task_sha256")):
                    integrity.append("worker result does not bind the task JSON")
                if index == len(attempt_records)-1 and record.get("status") == "completed":
                    if not snapshot.bound(result_path, record.get("result_sha256")):
                        integrity.append("scheduler result SHA256 mismatch")
                training = result.get("training", {})
                start, end, segment = training.get("from_tokens"), training.get("tokens"), training.get("segment_tokens")
                valid_tokens = (all(type(value) is int and value >= 0 for value in (start, end, segment))
                                and end >= start and segment == end-start)
                if result.get("status") in TERMINAL and not valid_tokens:
                    integrity.append("invalid cumulative/segment loss-token record")
                if not integrity and not publication_pending and valid_tokens:
                    recipe = {key: task.get(key) for key in RECIPE_FIELDS}
                    key = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    # Independent capacity probes do not form one continuation.
                    if category in {"preparation", "other"}:
                        key += ":" + task_id
                    exposure[(category, key)].append((start, end))
                    physical[category] += segment
                    duration = training.get("seconds")
                    if number(duration):
                        seconds += duration
                        backend_seconds[backend] += duration
                    if segment and (not number(duration) or duration <= 0):
                        timing_complete = False
                        backend_timing_complete[backend] = False
                    backend_tokens[backend] += segment
                    if is_npu and not inherited:
                        npu_loss_tokens += segment
                for reason in integrity:
                    snapshot.error(result_path, reason)
            attempts.append({"index": index, "status": attempt.get("status"),
                             "inherited_from_prior_round": inherited,
                             "return_code": import_row.get("worker_return_code") if inherited else group_worker.get("return_code") if group_id else attempt.get("return_code"), "queue_status": queue.get("status"),
                             "queue_return_code": queue.get("return_code"),
                             "device_seconds_recorded": attempt.get("device_seconds"),
                             "result_status": result.get("status") if result else None,
                             "shared_group": group_id, "group_completion_verified": group_complete,
                             "worker_completion_verified": worker_complete, "admission_pending": admission_pending,
                             "result_binding_pending": publication_pending,
                             "worker_interval": span, "live_open_worker_interval": span_open,
                             "never_started_verified": never_started,
                             "integrity_errors": integrity})
            latest_result, latest_integrity, latest_pending = result, integrity, publication_pending
        result = latest_result or {}
        training = result.get("training", {})
        gate_binding = True
        if category == "formal":
            gate_path = task.get("gate_file")
            gate = snapshot.read(gate_path) if gate_path else None
            gate_binding = bool(gate and gate.get("status") == "passed"
                and task_id in gate.get("allowed_task_ids", [])
                and snapshot.bound(task_path, gate.get("task_sha256", {}).get(task_id)))
            kind = {"b": "formal", "b_repeat": "repeat", "b_confirmation": "confirmation"}.get(phase)
            gate_binding = gate_binding and bool(kind and gates[kind]["binding_verified"]
                and gate.get("gate_kind") == kind and gate.get("source_identity") == task.get("source_identity")
                and inspect_gate(snapshot, gate_path)["binding_verified"])
        execution_complete = (record.get("status") == "completed" and result.get("status") == "completed"
            and bool(attempts) and not latest_integrity and not latest_pending and attempts[-1]["return_code"] in (0, None)
            and training.get("tokens") == task.get("target_tokens") and gate_binding
            and (attempts[-1]["worker_completion_verified"] if attempts[-1]["shared_group"] else
                 not is_npu or attempts[-1]["queue_status"] == "completed" and attempts[-1]["queue_return_code"] == 0))
        if record.get("status") == "completed" and not execution_complete:
            snapshot.error(task_path, "recorded completion is not established by the current task/result/gate evidence")
        health = result.get("health", {})
        recovery = result.get("recovery", {})
        resume = result.get("resume", {})
        continued = (resume.get("checkpoint_load", {}).get("status") == "exact"
                     and type(resume.get("from_tokens")) is int and resume["from_tokens"] > 0
                     and training.get("from_tokens") == resume["from_tokens"]
                     and type(training.get("segment_tokens")) is int and training["segment_tokens"] > 0)
        validation = result.get("validation_final", {})
        initial = result.get("validation_initial", {})
        confirmation = result.get("confirmation_final", {})
        heartbeat = record.get("heartbeat", {})
        if record.get("attempts"):
            last_dir = snapshot.path(record["attempts"][-1].get("path", f"tasks/{task_id}/attempt-{len(record['attempts'])-1}"))
            if last_dir is not None:
                heartbeat = snapshot.read(last_dir/"heartbeat.json") or heartbeat
        row = {"id": task_id, "model_id": model_id, "family": family, "group": groups.get(model_id),
            "backend": backend, "shared_group": record.get("shared_group"),
            "phase": phase, "stage": record.get("stage"), "seed": task.get("seed"),
            "scheduler_status": record.get("status"), "worker_status": result.get("status"),
            "execution_complete": execution_complete, "gate_binding_verified": gate_binding,
            "evidence_integrity_verified": bool(latest_result) and not latest_integrity and not latest_pending,
            "result_binding_pending": latest_pending,
            "reason": record.get("reason") or result.get("stop_reason"), "attempts": attempts,
            "target_loss_tokens": task.get("target_tokens"), "cumulative_loss_tokens": training.get("tokens"),
            "segment_loss_tokens": training.get("segment_tokens"), "from_loss_tokens": training.get("from_tokens"),
            "live_heartbeat": heartbeat, "training_seconds": training.get("seconds"),
            "tokens_per_second": training.get("tokens_per_second"),
            "health": health, "causality": result.get("causality"),
            "checkpoint_load": recovery.get("checkpoint_load", {"status": "not_recorded"}),
            "numeric_trajectory": recovery.get("numeric_trajectory", {"status": "not_recorded"}),
            "actual_training_continuation": continued,
            "resume": {key: resume.get(key) for key in ("from_tokens", "from_steps", "source_task_id", "state_reused", "runtime_override")},
            "selection_initial_nll": initial.get("nll"), "selection_final_nll": validation.get("nll"),
            "selection_evaluation_status": validation.get("status"),
            "confirmation_nll": confirmation.get("nll"), "confirmation_evaluation_status": confirmation.get("status"),
            "physical_visibility": result.get("physical_visibility"), "task_path": str(snapshot.path(task_path)),
            "resolved_initialization": result.get("initialization", {}).get("resolved_recipe"),
            "device_memory": result.get("device_memory")}
        if task.get("stage_a_migration"):
            from .source_migration3 import prior_evaluation_tokens
            try:
                receipt = snapshot.read_import_reference(task["stage_a_migration"])
                before = prior_evaluation_tokens(receipt, snapshot.read_import_reference)
                current = result.get("evaluation", {}).get("tokens")
                row["source_migration_evaluations"] = {"prior_lineage_tokens":before,
                    "current_attempt_tokens":current,
                    "cumulative_observed_tokens":before+current if type(current) is int else None}
            except (KeyError, OSError, TypeError, ValueError) as exc:
                snapshot.error(task_path, f"source migration evaluation history unavailable: {exc}")
        rows.append(row)
        final_results[task_id] = result

    unique = Counter()
    for (category, _), spans in exposure.items():
        unique[category] += union_length(spans)
    chosen = []
    for model_id, choice in sorted(selected.items()):
        entries = choice.get("confirmation", [])
        entry = entries[0] if entries else {}
        task = snapshot.read(entry["task_path"], required=True, inside=False) if entry.get("task_path") else None
        task = task or {}
        observed = (snapshot.read(entry["result_path"], required=True, inside=False) or {}) if entry.get("result_path") else {}
        if observed and (not task.get("source_identity") or observed.get("source_identity") != task["source_identity"]
                or not snapshot.bound(entry["task_path"], observed.get("task_sha256"), inside=False)):
            snapshot.error(entry["result_path"], "selected evidence does not bind its own task/source identity")
        chosen.append({"model_id": model_id, "recipe_id": choice.get("recipe_id"),
            **{field: task.get(field) for field in ("batch_size", "sequence_length", "cpu_threads", "lr", "warmup_tokens", "precision", "auxiliary_loss_weight")},
            "init_recipe": choice.get("init_recipe"),
            "resolved_initialization": observed.get("initialization", {}).get("resolved_recipe"),
            "activity": choice.get("activity"), "selection_nll": observed.get("validation_final", {}).get("nll"),
            "source_identity": task.get("source_identity"),
            "external_evidence": bool(entry.get("task_path") and not Path(entry["task_path"]).resolve().is_relative_to(snapshot.root)),
            "evidence_task_path": entries[0].get("task_path") if entries else None})

    formal = [row for row in rows if row["phase"] in {"b", "b_repeat"}]
    reached = {row["model_id"] for row in formal if type(row["cumulative_loss_tokens"]) is int
               and row["cumulative_loss_tokens"] >= FORMAL_TOKENS
               and not any(a["integrity_errors"] or a["result_binding_pending"] for a in row["attempts"])}
    completed = {row["model_id"] for row in formal if row["execution_complete"]
                 and row["target_loss_tokens"] == FORMAL_TOKENS and row["cumulative_loss_tokens"] == FORMAL_TOKENS}
    healthy = {row["model_id"] for row in formal if row["execution_complete"] and row["model_id"] in completed and row["health"].get("eligible") is True
               and not row["health"].get("blocking_reasons")}
    # Independent workers deliberately overlap inside the same physical lock.
    # Keep distinct leases separate for the cross-lease overlap checks below.
    runq_cards = {}
    for (card,lease), spans in runq_intervals.items():
        if runq_cards.setdefault(lease,card) != card:
            snapshot.error(str(lease),"one runq lease references different physical cards")
        merged = merge_intervals(spans)
        by_card[card].extend(merged)
        admitted_by_card[card].extend(merged)
        conservative_by_card[card].extend(False for _ in merged)
        allocation_count += len(merged)
    open_allocations += len(runq_open)
    intervals = [span for values in by_card.values() for span in merge_intervals(values)]
    accounting_overlap = 0
    for card, values in by_card.items():
        overlap = sum(end-start for start, end in values) - union_length(values)
        accounting_overlap += overlap
        if overlap > .1 and not all(conservative_by_card[card]):
            snapshot.error(str(card), "physical card has unexplained overlapping lease accounting intervals")
    for card, values in admitted_by_card.items():
        if sum(end-start for start, end in values) - union_length(values) > .1:
            snapshot.error(str(card), "physical card has overlapping admitted lease intervals")
    admitted_stats = concurrency([span for values in admitted_by_card.values() for span in merge_intervals(values)])
    admitted_stats.update(timing_complete=sum(map(len, admitted_by_card.values())) == allocation_count,
        scope="successful queue admission to physical lock release; excludes conservative admission prelude and post-release FIFO cleanup")
    queue_stats = concurrency(intervals)
    queue_stats.update(method="union by physical card of conservative shared accounting intervals and legacy queue intervals",
        attempted_npu_jobs=attempted_npu, verified_live_open_allocations=open_allocations,
        complete_intervals=allocation_count-open_allocations,
        conservative_accounting_overlap_seconds=accounting_overlap,
        recorded_group_device_seconds=sum(value["group"].get("physical_device_seconds", 0) for value in shared.values()
            if number(value["group"].get("physical_device_seconds"))),
        admitted_leases=admitted_stats,
        accounting_complete=all(value["binding_verified"] for value in shared.values()) and not any("queue" in e["reason"] or "acquire" in e["reason"] or "release" in e["reason"]
                                    or "accounting" in e["reason"] or "lease" in e["reason"] for e in snapshot.errors),
        ledger_device_seconds=state.get("device_seconds"),
        scheduler_polled_peak=state.get("peak_actual_npu_concurrency"))
    logical = worker_concurrency(logical_intervals["cpu"] + logical_intervals["npu"], missing=sum(logical_missing.values()))
    cpu_concurrency = worker_concurrency(logical_intervals["cpu"], missing=logical_missing["cpu"])
    npu_concurrency = worker_concurrency(logical_intervals["npu"], missing=logical_missing["npu"])
    if config.get("runq_root"):
        queue_stats["method"] = "union of independently recorded worker intervals per physical card"
        admitted_stats["scope"] = "observed worker occupancy within each identified physical runq lease"
        queue_stats["ledger_scope"] = "dedicated physical leases or union of campaign spans on mixed leases"
    queue_stats.update(logical_workers=logical, cpu_workers=cpu_concurrency, npu_workers=npu_concurrency,
                       shared_group_count=len(shared), scheduler_polled_npu_worker_peak=state.get("peak_npu_worker_concurrency"))
    stage_a_view = {"status": stage_a.get("status", "not_recorded"),
                    "blocking_reasons": stage_a.get("blocking_reasons", []),
                    "eligible_subject_count": stage_a.get("eligible_subject_count")}
    state_status = state.get("status", "unavailable")
    status = state_status
    if state_status == "completed" and snapshot.errors:
        status = "inconsistent_completed_record"
    elif state_status == "completed" and not config.get("preparation") and final_summary.get("schema") == "pilot-r3-final-summary-v1":
        if (stage_a.get("status") != "passed" or not gates["formal"]["binding_verified"]
                or len(completed & discovered_subjects) < 200 or any(not row["execution_complete"] for row in formal)):
            status = "incomplete_campaign_evidence"
    controller = process_observation(state.get("controller_identity"))
    observed_descendants = [process_observation(identity) for identity in state.get("all_seen_processes", {}).values()
                            if identity.get("pid") != state.get("controller_identity", {}).get("pid")]
    return {"schema": "pilot-r3-readonly-report-v1", "round": str(snapshot.root), "observed_unix": now,
        "status": status, "recorded_status": state_status, "stage": state.get("stage"),
        "blocking_reason": state.get("blocking_reason") or final_summary.get("blocking_reason"),
        "updated_unix": state.get("updated_unix"),
        "snapshot_age_seconds": max(0, now-state["updated_unix"]) if number(state.get("updated_unix")) else None,
        "gates": {"stage_a": stage_a_view, "stage_b_recorded": state.get("stage_b_gate", "not_recorded"), **gates},
        "execution": {"task_counts": dict(Counter(row["scheduler_status"] for row in rows)),
            "verified_completed_tasks": sum(row["execution_complete"] for row in rows),
            "pending_result_bindings": sum(row["result_binding_pending"] for row in rows),
            "declared_subject_models": len(subjects) if manifest else None,
            "observed_subject_models": len(discovered_subjects), "selected_models": len(chosen)},
        "exposure": {"unique_loss_tokens_by_category": dict(unique),
            "physically_computed_loss_tokens_by_category": dict(physical),
            "formal_target_per_trajectory": FORMAL_TOKENS,
            "distinct_formal_models_reached_target": len(reached),
            "distinct_formal_models_execution_completed": len(completed),
            "completed_subject_models": len(completed & discovered_subjects),
            "completed_healthy_subject_models": len(healthy & discovered_subjects),
            "completed_control_models": len(completed-discovered_subjects),
            "rule": "union of [from_tokens,tokens) by frozen recipe/model/seed identity; retries and confirmation do not duplicate exposure",
            "live_progress_policy": "heartbeats shown per task; not added to verified result exposure"},
        "concurrency": queue_stats,
        "resources": {"latest_scheduler_sample": state.get("resources"), "budget_ledger": ledger,
            "dispatch_queue": state.get("dispatch_queue"),
            "forecast": snapshot.read("forecast.json"), "model_capacity_choices": snapshot.read("model-resources.json"),
            "shared_groups": [{"id": key, "path": value["path"], "binding_verified": value["binding_verified"],
                "backend": value["group"].get("backend"), "status": value["group"].get("status"),
                "physical_id": value["group"].get("physical_id"),
                "running_workers": sum(row.get("status") == "running" for row in value["workers"].values()),
                "pending_workers": sum(row.get("status") == "pending" for row in value["workers"].values()),
                "admitted_workers": len(value["workers"]),
                "latest_resources": value["group"].get("resources"),
                "external_occupant_policy": value["manifest"].get("external_occupant_policy", "stop"),
                "external_contention_observed": value["group"].get("external_contention_observed"),
                "peak_external_processes": value["group"].get("peak_external_processes"),
                "device_observation_log": value["group"].get("device_observation_log"),
                "dynamic_admission": bool(value["manifest"].get("dynamic_admission")),
                "sharing_policy": value["group"].get("sharing_policy"),
                "sharing_policy_log": value["group"].get("sharing_policy_log"),
                "lifecycle_memory_observations": value["group"].get("lifecycle_memory_observations"),
                "physical_device_seconds": value["group"].get("physical_device_seconds"),
                "worker_wall_seconds": value["group"].get("worker_wall_seconds"), "errors": value["errors"]} for key, value in shared.items()]},
        "performance": {"sum_training_seconds": seconds,
            "external_contention_observed": any(value["group"].get("external_contention_observed") is True
                for value in shared.values()),
            "contention_scope": "sampled shared-group observations only; missing observations do not establish exclusivity",
            "training_timing_complete": timing_complete,
            "training_loss_tokens_per_second": sum(physical.values())/seconds if seconds and timing_complete else None,
            "cpu_loss_tokens": backend_tokens["cpu"], "npu_loss_tokens": backend_tokens["npu"],
            "cpu_training_seconds": backend_seconds["cpu"],
            "cpu_training_loss_tokens_per_second": backend_tokens["cpu"] / backend_seconds["cpu"]
                if backend_seconds["cpu"] and backend_timing_complete["cpu"] else None,
            "aggregate_cpu_loss_tokens_per_worker_window_second": backend_tokens["cpu"] / cpu_concurrency["worker_window_seconds"]
                if cpu_concurrency["worker_window_seconds"] and cpu_concurrency["timing_complete"] else None,
            "aggregate_npu_loss_tokens_per_allocation_window_second": npu_loss_tokens/queue_stats["allocation_window_seconds"]
                if queue_stats["allocation_window_seconds"] and queue_stats["accounting_complete"] else None,
            "training_scope": "worker transfer/forward/backward/update/synchronization and sampled diagnostics; excludes evaluation/checkpoint",
            "aggregate_scope": "new-round NPU loss-token computation over new-round first acquire to last release/current live observation; imported attempts excluded from this numerator and device lease intervals"},
        "numerical_health": {"eligibility_counts": dict(Counter("unverified_evidence" if not row["evidence_integrity_verified"] else "eligible" if row["health"].get("eligible") is True
            else "not_qualified" if row["health"] else "not_recorded" for row in rows)),
            "scope": "recorded sampled numerical/gradient/update health; no new diagnostic execution"},
        "state_loading": {"status_counts": dict(Counter(row["checkpoint_load"].get("status", "not_recorded") for row in rows)),
            "actual_training_continuations": sum(row["actual_training_continuation"] for row in rows),
            "scope": "saved state-digest/load observations and observed continued training; no checkpoint tensors loaded by this viewer"},
        "trajectory_reproduction": {"status_counts": dict(Counter(row["numeric_trajectory"].get("status", "not_recorded") for row in rows)),
            "policy": "state loading never implies trajectory tolerance passed; recorded thresholds/failures remain in task rows"},
        "learning": {"formal_trajectories": [{key: row[key] for key in ("model_id", "seed", "phase", "execution_complete",
            "cumulative_loss_tokens", "selection_initial_nll", "selection_final_nll", "selection_evaluation_status")} for row in formal],
            "confirmation_evaluations": [{key: row[key] for key in ("model_id", "seed", "confirmation_nll", "confirmation_evaluation_status")}
                for row in rows if row["phase"] == "b_confirmation"],
            "conclusion": "recorded NLL observations only; no automatic quality/equivalence claim"},
        "selected_recipes": chosen, "tasks": rows,
        "lifecycle": {"controller": controller, "recorded_descendants_currently_alive": [p for p in observed_descendants if p["alive"] is True],
            "saved_remaining_descendants": state.get("remaining_descendants"),
            "cleanup_observation_error": state.get("cleanup_observation_error"),
            "scope": "recorded PID/start identities only; not a fresh complete service/cgroup/queue release audit"},
        "observation_errors": snapshot.errors, "scientific_gate_reexecuted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write this explicit new JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    report = build_report(args.round)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")
    return 2 if report["status"] in {"unavailable", "inconsistent_completed_record", "incomplete_campaign_evidence"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
