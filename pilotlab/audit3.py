"""Read-only lifecycle and file-integrity audit for a round-three service.

No Torch/vendor import, queue cleanup, lock acquisition or service action is
performed. Optional report writes use the repository's review-only writer.
File integrity does not establish numerical replay or learning qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from .io import sha256

QUEUE_TERMINAL = {"completed", "failed", "cancelled", "timed_out"}
SCHEDULER_TERMINAL = {"completed", "failed", "a_blocked_needs_repair", "stopped_budget", "interrupted"}
PROPERTIES = "LoadState,ActiveState,SubState,MainPID,ControlGroup,Result,ExecMainStatus,ExecMainCode,Slice,InvocationID"
PROBLEM = {"missing", "invalid", "mismatch", "unknown", "changing"}


def json_observation(path: Path):
    observation = {"path": str(path)}
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        observation.update(status="passed", sha256=hashlib.sha256(raw).hexdigest())
        return value, observation
    except FileNotFoundError:
        observation.update(status="missing")
    except (ValueError, UnicodeError) as error:
        observation.update(status="invalid", error=str(error))
    except OSError as error:
        observation.update(status="unknown", error=str(error))
    return None, observation


def file_observation(path: Path, expected: str | None, size: int | None = None):
    observation = {"path": str(path), "expected_sha256": expected}
    try:
        before = path.stat()
        actual = sha256(path)
        after = path.stat()
        stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        observation.update(sha256=actual, size_bytes=after.st_size)
        if not stable:
            observation.update(status="changing")
        elif not expected:
            observation.update(status="unknown", reason="expected_digest_not_recorded")
        else:
            observation.update(status="passed" if actual == expected and (size is None or after.st_size == size) else "mismatch")
    except FileNotFoundError:
        observation.update(status="missing")
    except OSError as error:
        observation.update(status="unknown", error=str(error))
    return observation


def systemd_observation(unit: str):
    if not re.fullmatch(r"[A-Za-z0-9_.@\\-]+\.service", unit):
        return {"status": "invalid", "unit": unit, "reason": "invalid_service_unit"}
    try:
        result = subprocess.run(["systemctl", "--user", "show", unit, "--property=" + PROPERTIES],
                                capture_output=True, text=True, timeout=15, check=False)
        properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        observation = {"status": "observed" if result.returncode == 0 else "unknown", "unit": unit,
                       "properties": properties, "return_code": result.returncode}
        if result.returncode:
            observation["error"] = result.stderr.strip()[-1000:]
        if not properties.get("ControlGroup"):
            # Inactive services lose their ControlGroup property. Resolve its
            # location from the live user slice, rather than guessing a UID path.
            slice_name = properties.get("Slice") or "background.slice"
            result = subprocess.run(["systemctl", "--user", "show", slice_name, "--property=ControlGroup"],
                                    capture_output=True, text=True, timeout=15, check=False)
            fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            if result.returncode == 0 and fields.get("ControlGroup"):
                observation["expected_cgroup"] = fields["ControlGroup"].rstrip("/") + "/" + unit
                observation["expected_cgroup_basis"] = "observed user slice + exact service unit name"
        return observation
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unknown", "unit": unit, "error": str(error)}


def cgroup_observation(service: dict, *, root=Path("/sys/fs/cgroup")):
    group = service.get("properties", {}).get("ControlGroup") or service.get("expected_cgroup")
    observation = {"path": group, "released": None, "members": []}
    if not group or not str(group).startswith("/") or ".." in Path(group).parts:
        return dict(observation, status="unknown", reason="exact_cgroup_location_unavailable")
    try:
        # systemd exposes ControlGroup relative to its hierarchy. On v1
        # /sys/fs/cgroup itself is a tmpfs; hybrid hosts can expose both mounts.
        hierarchies = [root] if (root / "cgroup.procs").is_file() else [
            mount for mount in (root / "systemd", root / "unified") if (mount / "cgroup.procs").is_file()]
        if not hierarchies:
            return dict(observation, status="unknown", reason="systemd_cgroup_hierarchy_unavailable")
        members = set()
        events = {}
        found = False
        for mount in hierarchies:
            path = mount / str(group).lstrip("/")
            try:
                path.stat()
            except FileNotFoundError:
                continue
            found = True
            for member_file in [path / "cgroup.procs", *path.glob("**/*/cgroup.procs")]:
                members.update(int(line) for line in member_file.read_text().splitlines())
            event_path = path / "cgroup.events"
            if event_path.exists():  # This file belongs to the unified hierarchy.
                events = dict(line.split() for line in event_path.read_text().splitlines())
        released = not members and events.get("populated") != "1"
        if released and service.get("properties", {}).get("MainPID") not in (None, "", "0", 0):
            return dict(observation, status="changing", reason="systemd_main_pid_without_cgroup_members")
        if not found:
            return dict(observation, status="absent", released=True)
        return dict(observation, status="empty" if released else "populated", released=released,
                    members=sorted(members), events=events, location_basis=service.get("expected_cgroup_basis", "systemd ControlGroup"))
    except FileNotFoundError:
        # Membership changed while a previously visible group was being read.
        return dict(observation, status="changing", reason="cgroup_changed_during_read")
    except (OSError, ValueError) as error:
        return dict(observation, status="unknown", error=str(error))


def process_snapshot(root=Path("/proc")):
    table, errors = {}, []
    try:
        paths = list(root.iterdir())
    except OSError as error:
        return {}, [str(error)]
    for path in paths:
        if not path.name.isdecimal():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            text = (path / "stat").read_text()
            fields = text[text.rfind(")") + 2:].split()
            table[int(path.name)] = {"pid": int(path.name), "ppid": int(fields[1]),
                                    "state": fields[0], "start_ticks": int(fields[19])}
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError) as error:
            errors.append(f"{path}: {error}")
    return table, errors


def process_observation(state: dict, service: dict, cgroup: dict, table: dict, errors: list):
    identities = {}
    supplied = [state.get("controller_identity"), *state.get("all_seen_processes", {}).values()]
    for task in state.get("tasks", {}).values():
        supplied.append(task.get("process"))
        supplied.extend(attempt.get("process") for attempt in task.get("attempts", []))
        supplied.extend(attempt.get("worker_process") for attempt in task.get("attempts", []))
    for entry in state.get("shared_groups", {}).values():
        supplied.append(entry.get("process"))
        group, _ = json_observation(Path(entry["path"]) / "group.json")
        if group:
            supplied.append(group.get("owner"))
            supplied.extend(group.get("all_seen_processes", {}).values())
            supplied.extend(value.get("process") for value in group.get("workers", {}).values())
    malformed = []
    for value in supplied:
        if value is None:
            continue
        if not isinstance(value, dict) or type(value.get("pid")) is not int or type(value.get("start_ticks")) is not int:
            malformed.append(value)
        else:
            identities[(value["pid"], value["start_ticks"])] = value
    gone, reused, zombies, matching = 0, [], [], []
    for (pid, start), expected in identities.items():
        current = table.get(pid)
        if current is None:
            gone += 1
        elif current["start_ticks"] != start:
            reused.append({"pid": pid, "recorded_start_ticks": start, "current_start_ticks": current["start_ticks"]})
        elif current["state"] == "Z":
            zombies.append(current)
        else:
            matching.append(current)
    seeds = {row["pid"] for row in matching} | set(cgroup.get("members", []))
    raw_main = service.get("properties", {}).get("MainPID", "0")
    try:
        main_pid = int(raw_main)
    except (ValueError, TypeError):
        main_pid = 0
        errors = [*errors, "invalid systemd MainPID"]
    if main_pid in table:
        seeds.add(main_pid)
    descendants = set(seeds)
    while True:
        children = {pid for pid, row in table.items() if row["ppid"] in descendants}
        if children <= descendants:
            break
        descendants.update(children)
    live = [table[pid] for pid in sorted(descendants) if pid in table and table[pid]["state"] != "Z"]
    missing_members = sorted(pid for pid in cgroup.get("members", []) if pid not in table)
    unknown = bool(errors or malformed or missing_members or not state or not identities)
    return {"status": "live" if live else "unknown" if unknown else "released",
            "released": False if live else None if unknown else True, "recorded_identity_count": len(identities),
            "gone_count": gone, "reused_pids": reused, "zombies": zombies, "live_processes": live,
            "main_pid": main_pid, "errors": errors, "malformed_identities": malformed,
            "cgroup_members_unobservable_in_process_snapshot": missing_members}


def lock_snapshot(path=Path("/proc/locks")):
    try:
        locks = []
        for line in path.read_text().splitlines():
            fields = line.split()
            if "->" in fields or len(fields) < 8:
                continue
            major, minor, inode = fields[5].split(":")
            locks.append({"kind": fields[1], "pid": int(fields[4]),
                          "device": (int(major, 16), int(minor, 16)), "inode": int(inode)})
        return locks, None
    except (OSError, ValueError) as error:
        return [], str(error)


def lock_observation(path: Path, locks: list, error: str | None):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "status": "absent", "owners": []}
    except OSError as exc:
        return {"path": str(path), "status": "unknown", "error": str(exc)}
    if error:
        return {"path": str(path), "status": "unknown", "error": error}
    owners = [value["pid"] for value in locks if value["device"] == (os.major(stat.st_dev), os.minor(stat.st_dev))
              and value["inode"] == stat.st_ino]
    return {"path": str(path), "status": "held" if owners else "unlocked", "owners": owners,
            "method": "read /proc/locks by device/inode; never acquire a queue/device lock"}


def queue_observation(state: dict, processes: dict, *, runtime: Path, locks: list, lock_error: str | None):
    observations = []
    live_pids = {row["pid"] for row in processes["live_processes"]}
    queued = []
    groups = state.get("shared_groups", {})
    for group_id, entry in groups.items():
        if entry.get("backend") != "cpu":
            queued.append((group_id, {"path": entry["path"], "command": ["--job-id", entry["job_id"]],
                                      "shared_group": group_id}))
    for task_id, record in state.get("tasks", {}).items():
        for attempt in record.get("attempts", []):
            group_id = attempt.get("shared_group") or (record.get("shared_group") if len(record.get("attempts", [])) == 1 else None)
            if group_id:
                if group_id not in groups:
                    observations.append({"task_id": task_id, "status": "unknown", "released": None,
                                         "reason": "shared_group_queue_record_missing"})
                continue
            queued.append((task_id, attempt))
    for task_id, attempt in queued:
            command = attempt.get("command", [])
            if "--job-id" not in command:
                device_index = command.index("--device") if "--device" in command else -1
                if device_index < 0 or device_index + 1 >= len(command) or command[device_index + 1] != "cpu":
                    observations.append({"task_id": task_id, "attempt": attempt.get("path"),
                        "status": "unknown", "released": None, "reason": "queue_command_identity_missing"})
                continue
            index = command.index("--job-id")
            job_id = command[index + 1] if index + 1 < len(command) else ""
            row = {"task_id": task_id, "attempt": attempt.get("path"), "job_id": job_id,
                   "shared_group": attempt.get("shared_group")}
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", job_id):
                observations.append(dict(row, status="invalid", released=None, reason="invalid_recorded_job_id"))
                continue
            q, q_observation = json_observation(Path(attempt["path"]) / "queue.json")
            manifest, manifest_observation = json_observation(runtime / "ascend-job-queue/jobs" / (job_id + ".json"))
            job_lock = lock_observation(runtime / "ascend-job-queue/jobs" / (job_id + ".lock"), locks, lock_error)
            physical_ids = (q or {}).get("physical_ids", (manifest or {}).get("physical_ids", []))
            device_locks = []
            for physical in physical_ids:
                if type(physical) is not int or physical < 0:
                    device_locks.append({"status": "invalid", "reason": "invalid_physical_id"})
                    continue
                value = lock_observation(runtime / "ascend-device-locks" / f"physical-{physical}.lock", locks, lock_error)
                value["owned_by_this_run"] = any(pid in live_pids for pid in value.get("owners", []))
                device_locks.append(value)
            own_lock = any(value.get("owned_by_this_run") for value in device_locks)
            live = job_lock["status"] == "held" or own_lock
            known = (q is not None and q.get("job_id") == job_id and q.get("status") in QUEUE_TERMINAL
                     and manifest_observation["status"] == "missing" and job_lock["status"] in {"absent", "unlocked"}
                     and all(value["status"] not in PROBLEM for value in device_locks)
                     and processes["released"] is True)
            no_allocation_observed = (q_observation["status"] == "missing" and manifest_observation["status"] == "missing"
                and job_lock["status"] == "absent" and processes["released"] is True)
            observations.append(dict(row, status="active" if live else "released" if known else (
                    "not_allocated_observed" if no_allocation_observed else "unknown"),
                released=False if live else True if known or no_allocation_observed else None, queue_state=q_observation,
                recorded_status=(q or {}).get("status"), manifest=manifest_observation, job_lock=job_lock,
                device_locks=device_locks, scope="release of this job's cooperative ownership; another job may now hold the physical device"))
    if not observations:
        return {"status": "not_requested" if state else "unknown", "released": True if state else None,
                "runtime_directory": str(runtime), "attempts": []}
    status = "active" if any(row["status"] == "active" for row in observations) else (
        "released" if all(row["released"] is True for row in observations) else "unknown")
    return {"status": status, "released": status == "released" if status != "unknown" else None,
            "runtime_directory": str(runtime), "attempts": observations}


def shared_group_observations(state, output, cgroup, process_table=None, *, allow_inert_worker_estimates=False):
    def alive(identity):
        identity = identity or {}
        current = (process_table or {}).get(identity.get("pid"), {})
        return bool(current and current.get("start_ticks") == identity.get("start_ticks") and current.get("state") != "Z")

    checks, groups = [], {}
    scheduler_live = state.get("status") not in SCHEDULER_TERMINAL and alive(state.get("controller_identity"))
    for group_id, entry in state.get("shared_groups", {}).items():
        directory = Path(entry["path"]).resolve()
        manifest_path = Path(entry.get("manifest_path", directory / "manifest.json"))
        manifest, manifest_record = json_observation(manifest_path)
        group, group_record = json_observation(directory / "group.json")
        checks.extend([dict(manifest_record, kind="shared_manifest", group_id=group_id),
                       dict(group_record, kind="shared_group", group_id=group_id)])
        reasons, accounting_reasons = [], []
        if not directory.is_relative_to(output / "groups") or manifest_path.resolve().parent != directory:
            reasons.append("shared_group_paths_outside_round_groups")
        if manifest_record.get("sha256") != entry.get("manifest_sha256"):
            reasons.append("shared_manifest_sha256_mismatch")
        if manifest is None or group is None:
            continue
        if (manifest.get("schema") != "pilot-r3-shared-group-v1" or group.get("schema") != "pilot-r3-shared-group-state-v1"
                or manifest.get("job_id") != entry.get("job_id") or group.get("job_id") != entry.get("job_id")
                or group.get("manifest_sha256") != entry.get("manifest_sha256")
                or group.get("backend") != entry.get("backend") or group.get("backend") != manifest.get("backend")):
            reasons.append("shared_group_identity_mismatch")
        published = acknowledged = manifest.get("workers", [])
        admission = None
        dynamic = bool(manifest.get("dynamic_admission"))
        if dynamic:
            try:
                from .shared_npu3 import read_admission_chain
                admission = read_admission_chain(manifest, manifest_path, group)
                published, acknowledged = admission["published_workers"], admission["acknowledged_workers"]
                for command in admission["commands"]:
                    checks.append({"kind": "shared_admission_command", "group_id": group_id,
                        "path": command["path"], "sha256": command["sha256"], "status": "passed"})
                if group.get("status") == "completed" and not admission["acknowledged_closed"]:
                    reasons.append("dynamic_group_completed_without_close_acknowledgement")
            except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
                reasons.append(f"dynamic_admission_invalid:{error}")
        declared = {worker["id"]: worker for worker in published}
        ack_ids = {worker["id"] for worker in acknowledged}
        if (len(declared) != len(published) or ack_ids != set(group.get("workers", {}))
                or not (dynamic and scheduler_live) and set(declared) != set(entry.get("task_ids", []))):
            reasons.append("shared_group_worker_list_mismatch")
        for task_id, worker in declared.items():
            record = state.get("tasks", {}).get(task_id, {})
            attempts = [attempt for attempt in record.get("attempts", []) if attempt.get("shared_group") == group_id
                or len(record.get("attempts", [])) == 1 and record.get("shared_group") == group_id]
            observed = group.get("workers", {}).get(task_id, {})
            if dynamic and scheduler_live and (not observed or not record):
                checks.append({"kind": "shared_worker_admission", "group_id": group_id,
                    "task_id": task_id, "status": "pending", "reason": "admission_snapshot_publication_pending"})
                continue
            if (len(attempts) != 1 or worker.get("output") != attempts[0].get("path")
                    or observed.get("output") != worker.get("output")
                    or worker.get("argv") != attempts[0].get("command")
                    or Path(attempts[0].get("group_path", directory)).resolve() != directory):
                reasons.append("shared_worker_attempt_binding_mismatch:" + task_id)
            if record.get("device_seconds", 0) != 0 or any(a.get("device_seconds", 0) != 0 for a in attempts):
                # Old stop_all() wrote a fallback elapsed estimate into a
                # shared worker record. SharedSupervisor sums only its owner
                # debit; verify that exclusion explicitly, never erase it.
                estimate = record.get("device_seconds")
                owner_total = sum(g.get("device_seconds", 0) for g in state.get("shared_groups", {}).values())
                owner_total += sum(t.get("device_seconds", 0) for t in state.get("tasks", {}).values() if not t.get("shared_group"))
                inert = (allow_inert_worker_estimates and state.get("status") in SCHEDULER_TERMINAL
                    and cgroup.get("released") is True and not alive(state.get("controller_identity"))
                    and not alive(group.get("owner")) and group.get("remaining_descendants") == []
                    and record.get("status") == "failed" and record.get("shared_group") == group_id
                    and str(record.get("reason", "")).startswith("cleanup_evidence_error:")
                    and len(attempts) == len(record.get("attempts", [])) == 1
                    and attempts[0].get("accounting_estimated") is True
                    and attempts[0].get("accounting_reason") == "unreadable_current_attempt_accounting"
                    and isinstance(estimate, (int, float)) and not isinstance(estimate, bool)
                    and math.isfinite(estimate) and estimate > 0
                    and estimate == attempts[0].get("device_seconds")
                    and math.isclose(state.get("device_seconds", -1), owner_total, abs_tol=1e-6, rel_tol=1e-9))
                if inert:
                    checks.append({"kind": "inert_worker_accounting_estimate", "group_id": group_id,
                        "task_id": task_id, "status": "passed", "retained_estimate_seconds": estimate,
                        "scope": "original inconsistent worker field preserved; excluded from scheduler total; owner/segment/global ledger checked independently"})
                else:
                    accounting_reasons.append("shared_worker_double_charged_device_seconds:" + task_id)
            if observed.get("result_path") and observed["result_path"] != str(Path(worker["output"]) / "result.json"):
                reasons.append("shared_worker_result_path_mismatch:" + task_id)
            if dynamic and observed:
                from .shared_npu3 import worker_spec_digest
                if observed.get("worker_spec_sha256") != worker_spec_digest(worker):
                    reasons.append("dynamic_worker_spec_digest_mismatch:" + task_id)
                if observed.get("status") == "completed" and (observed.get("descendant_cleanup_verified") is not True
                        or observed.get("remaining_descendants") != [] or alive(observed.get("process"))):
                    reasons.append("dynamic_completed_worker_cleanup_unverified:" + task_id)
        if group.get("status") in {"completed", "failed", "stopped_budget"} and group.get("remaining_descendants") != []:
            reasons.append("shared_group_terminal_descendants_not_empty")
        owner_alive = alive(group.get("owner", entry.get("process")))
        if group.get("status") not in {"completed", "failed", "stopped_budget"}:
            checks.append({"kind": "shared_group_lifecycle", "group_id": group_id,
                "status": "pending" if owner_alive and state.get("status") not in SCHEDULER_TERMINAL else "unknown",
                "reason": "shared_group_not_terminal"})
        owner_cgroup = group.get("owner_cgroup")
        expected = cgroup.get("path")
        if not owner_cgroup:
            reasons.append("shared_owner_cgroup_not_recorded")
        elif expected:
            observed_paths = [line.split(":", 2)[-1] for line in owner_cgroup.splitlines()]
            if not any(path == expected or path.startswith(expected.rstrip("/") + "/") for path in observed_paths):
                reasons.append("shared_owner_was_outside_service_cgroup")
        for reason in reasons + accounting_reasons:
            checks.append({"kind": "shared_group_binding", "group_id": group_id, "path": str(directory),
                           "status": "mismatch", "reason": reason})
        groups[group_id] = {"entry": entry, "manifest": manifest, "group": group,
                           "binding_verified": not reasons, "owner_alive": owner_alive}
    for task_id, record in state.get("tasks", {}).items():
        for attempt in record.get("attempts", []):
            if attempt.get("shared_group") and attempt["shared_group"] not in groups:
                checks.append({"kind": "shared_group_binding", "status": "missing", "task_id": task_id,
                               "reason": "shared_attempt_group_unavailable"})
    return {"groups": groups, "checks": checks, "scheduler_live": scheduler_live}


def shared_accounting_observation(state, prepared, output, shared, *, allow_conservative_debits=False):
    if not state.get("shared_groups"):
        return {"status": "not_applicable", "checks": []}
    checks = []
    ledger_path = state.get("config", {}).get("budget_ledger") or prepared.get("budget_ledger")
    if not ledger_path:
        return {"status": "missing", "checks": [{"status": "missing", "reason": "budget_ledger_not_declared"}]}
    ledger, ledger_record = json_observation(Path(ledger_path))
    checks.append(ledger_record)
    if ledger is None:
        return {"status": ledger_record["status"], "checks": checks}
    conservative_excess = []
    def compare(name, observed, expected, *, tolerance=1e-6, pending=False, upper_bound=False):
        numeric = (not isinstance(observed, bool) and isinstance(observed, (int, float))
                   and math.isfinite(observed) and observed >= 0)
        valid = numeric and math.isclose(observed, expected, abs_tol=tolerance, rel_tol=1e-9)
        conservative = bool(upper_bound and allow_conservative_debits and numeric and observed >= expected and not valid)
        item = {"status": "passed" if valid or conservative else "pending" if pending and numeric else "mismatch",
                "kind": "physical_accounting", "reason": name, "observed": observed, "expected": expected}
        if conservative:
            item.update(comparison="retained_conservative_upper_bound", excess_seconds=observed-expected,
                        scope="budget debit retained in full; not an exact physical-duration claim")
            conservative_excess.append(dict(item))
        checks.append(item)
    total = preparation = scheduler_total = scheduler_preparation = 0.
    for group_id, entry in state["shared_groups"].items():
        value = shared["groups"].get(group_id)
        if value is None:
            checks.append({"status": "unknown", "reason": "missing_group_accounting", "group_id": group_id})
            continue
        group = value["group"]
        live_group = value.get("owner_alive") and state.get("status") not in SCHEDULER_TERMINAL
        seconds = group.get("physical_device_seconds")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
            checks.append({"status": "invalid", "reason": "invalid_group_device_seconds", "group_id": group_id})
            continue
        start, end = group.get("allocation_acquired_unix"), group.get("allocation_released_unix")
        if group.get("backend") == "cpu":
            compare("CPU group must not charge NPU time:" + group_id, seconds, 0)
            if start is not None or end is not None:
                checks.append({"status": "invalid", "reason": "CPU group has NPU allocation timestamps", "group_id": group_id})
        elif isinstance(start, (int, float)) and isinstance(end, (int, float)) and math.isfinite(start) and math.isfinite(end) and end >= start:
            compare("group acquire/release interval:" + group_id, seconds, end-start, tolerance=.1)
        elif start is None and end is None and seconds == 0:
            pass
        else:
            checks.append({"status": "pending" if live_group else "unknown",
                           "reason": "group allocation lacks terminal interval", "group_id": group_id})
        compare("scheduler group debit:" + group_id, entry.get("device_seconds"), seconds,
                pending=live_group or shared.get("scheduler_live") and entry.get("status") not in {"completed", "failed", "stopped_budget"},
                upper_bound=group.get("backend") == "npu")
        total += seconds
        debit = entry.get("device_seconds", 0)
        if not isinstance(debit, (int, float)) or isinstance(debit, bool) or not math.isfinite(debit) or debit < 0:
            continue
        scheduler_total += debit
        if entry.get("preparation"):
            preparation += seconds
            scheduler_preparation += debit
    for record in state.get("tasks", {}).values():
        if not record.get("shared_group"):
            total += record.get("device_seconds", 0)
            scheduler_total += record.get("device_seconds", 0)
            if state.get("config", {}).get("preparation") or record.get("preparation"):
                preparation += record.get("device_seconds", 0)
                scheduler_preparation += record.get("device_seconds", 0)
    segment = ledger.get("segments", {}).get(str(output), {})
    compare("scheduler total physical debit", state.get("device_seconds"), scheduler_total)
    compare("budget segment physical debit", segment.get("device_seconds"), scheduler_total, pending=shared.get("scheduler_live"))
    compare("budget segment preparation debit", segment.get("preparation_device_seconds"), scheduler_preparation, pending=shared.get("scheduler_live"))
    compare("global ledger sums its segments", ledger.get("device_seconds"),
            sum(value.get("device_seconds", 0) for value in ledger.get("segments", {}).values()))
    status = ("passed" if all(check["status"] == "passed" for check in checks) else
              "pending" if all(check["status"] in {"passed", "pending"} for check in checks) else "incomplete")
    return {"status": status, "physical_device_seconds": total, "scheduler_device_seconds": scheduler_total,
            "preparation_device_seconds": preparation, "checks": checks,
            "comparison_policy": "retained_conservative_upper_bound" if allow_conservative_debits else "exact_debit",
            "conservative_excess_seconds": sum(row["excess_seconds"] for row in conservative_excess),
            "conservative_debits": conservative_excess,
            "scope": "one debit per shared lease plus legacy ungrouped attempts; worker elapsed seconds excluded"}


def input_observations(prepared: dict):
    identity = prepared.get("source_identity", {})
    declared_sources = {**identity.get("source_files", {}),
                        **prepared.get("worker_source_identity", {}).get("source_files", {})}
    sources = [file_observation(Path(path), digest) for path, digest in declared_sources.items()]
    if not sources:
        sources.append({"status": "missing", "reason": "source_file_manifest_missing"})
    data = []
    if prepared.get("data"):
        directory = Path(prepared["data"])
        manifest, observation = json_observation(directory / "manifest.json")
        data.append(file_observation(directory / "manifest.json", identity.get("data_manifest_sha256")))
        if manifest:
            if manifest.get("schema_version") != "tide-pilot-data-v3":
                data.append({"path": str(directory / "manifest.json"), "status": "invalid",
                             "reason": "unexpected_data_manifest_schema"})
            source = Path(manifest["source_directory"])
            data.append(file_observation(source / "manifest.json", manifest.get("source_manifest_sha256")))
            for base, records in ((source, manifest.get("source_files", {})), (directory, manifest.get("files", {}))):
                for name, expected in records.items():
                    path = (base / name).resolve()
                    if not path.is_relative_to(base.resolve()):
                        data.append({"path": str(path), "status": "invalid", "reason": "data_asset_escapes_root"})
                    else:
                        data.append(file_observation(path, expected.get("sha256"), expected.get("size_bytes")))
        else:
            data.append(observation)
    else:
        data.append({"status": "missing", "reason": "data_directory_not_declared"})
    return {"source_files": sources, "data_files": data,
            "scope": "full byte hashes; no data preparation, import of training code, or device initialization"}


def artifact_observations(state: dict, prepared: dict, output: Path, shared=None):
    from .lifecycle3 import cancelled_before_allocation
    retirement, retirement_observation = json_observation(output / "checkpoint-retirement.json")
    if retirement is None and retirement_observation["status"] == "missing":
        retirement_observation["status"] = "not_present"
    retirements = {value.get("path"): value for value in (retirement or {}).get("files", [])}
    observations, hashes, checked_sources = [], {}, set()
    for task_id, record in state.get("tasks", {}).items():
        if record.get("imported_from"):
            try:
                from .resume3 import bound, imported_entry, verify_imported_completion
                imported, original_task, original_result = imported_entry(state["config"], task_id)
                if record.get("imported_completion"):
                    verify_imported_completion(state["config"], original_task, record)
                observations.append({"kind": "imported_evidence", "task_id": task_id, "status": "passed",
                    "disposition": imported["disposition"], "reference": state["config"]["resume_import"],
                    "scope": "bound original result/task and supplementary release review; old failed attempts retained"})
                if original_result:
                    cp = original_result.get("checkpoint")
                    if cp:
                        cp_observation = file_observation(Path(cp["path"]), cp["sha256"], cp.get("size_bytes"))
                        if cp_observation["status"] == "missing":
                            import_manifest = bound(state["config"]["resume_import"])
                            from .resume3 import retirement_references
                            retired = any(item.get("path") == cp["path"] and item.get("sha256") == cp["sha256"]
                                and item.get("task_id") == task_id
                                for reference in retirement_references(import_manifest)
                                for item in bound(reference).get("files", []))
                            if retired and original_result.get("status") == "completed" and original_task.get("phase") in {"preflight", "init", "a1", "a2"}:
                                cp_observation.update(status="retired", reason="bound original successful-checkpoint retirement")
                        observations.append(dict(cp_observation, kind="imported_checkpoint", task_id=task_id))
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                observations.append({"kind": "imported_evidence", "task_id": task_id, "status": "mismatch", "reason": str(error)})
        attempts = record.get("attempts", [])
        task_path = Path(record.get("task_path", output / "tasks" / task_id / "task.json"))
        task, task_observation = json_observation(task_path)
        task_observation.update(kind="task", task_id=task_id)
        if task is not None and (task.get("id") != task_id or not task.get("source_identity")):
            task_observation.update(status="mismatch", reason="task_or_source_identity_mismatch")
        if task is not None and str(task.get("phase", "")).startswith("b") and task.get("source_identity") != prepared.get("worker_source_identity", prepared.get("source_identity")):
            task_observation.update(status="mismatch", reason="formal_task_is_not_bound_to_current_frozen_identity")
        if task and task.get("source_identity") != prepared.get("source_identity"):
            identity = task.get("source_identity") or {}
            if not identity.get("source_files"):
                observations.append({"status": "missing", "kind": "task_source", "task_id": task_id,
                                     "reason": "distinct_task_source_identity_has_no_file_hashes"})
            for path, digest in identity.get("source_files", {}).items():
                if (path, digest) not in checked_sources:
                    observations.append(dict(file_observation(Path(path), digest), kind="task_source", task_id=task_id))
                    checked_sources.add((path, digest))
        observations.append(task_observation)
        if not attempts:
            continue
        for attempt in attempts:
            directory = Path(attempt["path"]).resolve()
            if not directory.is_relative_to(output.resolve()):
                observations.append({"kind": "attempt", "status": "invalid", "path": str(directory),
                                     "reason": "attempt_outside_new_round"})
                continue
            result_path = directory / "result.json"
            result, result_observation = json_observation(result_path)
            result_observation.update(kind="result", task_id=task_id)
            group_id = attempt.get("shared_group") or (record.get("shared_group") if len(attempts) == 1 else None)
            group_info = (shared or {}).get("groups", {}).get(group_id, {})
            worker = group_info.get("group", {}).get("workers", {}).get(task_id, {})
            group_not_started = (group_info.get("binding_verified") and worker.get("status") == "not_started"
                and worker.get("pid") is None and worker.get("started_unix") is None
                and not any((directory / name).exists() for name in ("result.json", "heartbeat.json", "checkpoint.json", "failure.pt"))
                and not any((directory / "checkpoints").glob("*.pt")))
            if result is None and attempt.get("status") == "running" and result_observation["status"] == "missing":
                result_observation["status"] = "pending"
            elif result is None and result_observation["status"] == "missing" and group_not_started:
                result_observation.update(status="not_started", reason="shared_group_worker_not_started")
            elif (result is None and result_observation["status"] == "missing"
                  and cancelled_before_allocation(directory, json_observation(directory / "queue.json")[0] or {})):
                result_observation.update(status="not_started", reason="cancelled_before_allocation")
            elif result is not None:
                if str(result_path) == record.get("result_path") and result_observation["sha256"] != record.get("result_sha256"):
                    result_observation.update(status="mismatch", reason="scheduler_result_digest_mismatch")
                if (task is None or result.get("task_sha256") != task_observation.get("sha256")
                        or result.get("source_identity") != task.get("source_identity")):
                    result_observation.update(status="mismatch", reason="result_task_or_source_identity_mismatch")
                if group_id and (not group_info.get("binding_verified") or worker.get("result_sha256") != result_observation.get("sha256")):
                    pending = (group_info.get("binding_verified") and group_info.get("owner_alive")
                        and group_info.get("group", {}).get("status") in {"running", "cleanup_blocked"}
                        and worker.get("status") in {"pending", "starting", "running", "exited_waiting_cleanup"}
                        and not worker.get("result_sha256")
                        and record.get("status") == "running" and attempt.get("status") in {"admitting", "starting", "running"}
                        and state.get("status") not in SCHEDULER_TERMINAL)
                    if pending and result_observation["status"] == "passed":
                        result_observation.update(status="pending", reason="shared_worker_result_publication_pending")
                    elif not pending:
                        result_observation.update(status="mismatch", reason="shared_worker_result_binding_mismatch")
            observations.append(result_observation)
            result = result or {}
            index, index_observation = json_observation(directory / "checkpoint.json")
            references = []
            if result.get("checkpoint"):
                references.append(result["checkpoint"])
            elif result.get("status") == "completed":
                observations.append({"kind": "checkpoint", "task_id": task_id, "status": "missing",
                                     "reason": "completed_result_has_no_checkpoint"})
            if index:
                if index.get("schema") != "pilot-r3-checkpoint-generations-v1":
                    index_observation.update(status="invalid", reason="unexpected_checkpoint_index_schema")
                references.extend(index[key] for key in ("current", "previous") if index.get(key))
            elif not references and index_observation["status"] == "missing":
                index_observation["status"] = "not_created"
            index_observation.update(kind="checkpoint_index", task_id=task_id)
            observations.append(index_observation)
            seen = set()
            for reference in references:
                path = Path(reference["path"]).resolve()
                digest = reference.get("sha256")
                if path in seen:
                    if hashes.get(str(path)) != digest:
                        observations.append({"path": str(path), "kind": "checkpoint", "status": "mismatch",
                                             "reason": "conflicting_checkpoint_digests"})
                    continue
                seen.add(path)
                hashes[str(path)] = digest
                value = file_observation(path, digest, reference.get("size_bytes"))
                value.update(kind="checkpoint", task_id=task_id)
                if not path.is_relative_to(directory / "checkpoints"):
                    value.update(status="invalid", reason="checkpoint_outside_own_attempt")
                retired = retirements.get(str(path))
                is_formal_final = task and task.get("phase") in {"b", "b_repeat"} and str(path) == result.get("checkpoint", {}).get("path")
                legal = (retired and retirement.get("schema") == "pilot-r3-retirement-v1"
                    and record.get("status") == attempt.get("status") == result.get("status") == "completed"
                    and result_observation["status"] == task_observation["status"] == "passed"
                    and task and bool(task.get("source_identity"))
                    and path.is_relative_to(directory / "checkpoints") and not is_formal_final
                    and retired.get("task_id") == task_id and retired.get("sha256") == digest
                    and retired.get("bytes") == reference.get("size_bytes") and bool(retired.get("reason")))
                if value["status"] == "missing" and legal:
                    value.update(status="retired", retirement=retired)
                elif retired and not legal:
                    value.update(status="invalid", reason="retirement_not_authorized_for_this_artifact")
                observations.append(value)
            for path in (directory / "checkpoints").glob("*.pt"):
                if path.resolve() not in seen:
                    observations.append({"path": str(path), "kind": "checkpoint", "status": "unknown",
                        "reason": "checkpoint_file_has_no_recorded_generation_digest"})
            capture = result.get("failure_capture")
            if capture:
                path = Path(capture["path"]).resolve()
                value = file_observation(path, capture.get("sha256"))
                value.update(kind="failure_capture", task_id=task_id)
                if path != directory / "failure.pt":
                    value.update(status="invalid", reason="failure_payload_outside_own_attempt")
                observations.append(value)
            elif (directory / "failure.pt").exists():
                observations.append({"path": str(directory / "failure.pt"), "kind": "failure_capture",
                    "status": "unknown", "reason": "failure_payload_has_no_recorded_digest"})
            if result.get("failure_capture_error"):
                observations.append({"kind": "failure_capture", "task_id": task_id, "status": "missing",
                                     "reason": result["failure_capture_error"]})
    return {"retirement_manifest": retirement_observation, "files": observations,
            "scope": "full SHA256/size against recorded generation/result identities; payload decoding and numerical replay are not performed"}


def selected_observations(output):
    selected, record = json_observation(output / "selected.json")
    checks, seen = [], set()
    if selected is None:
        return {"checks": [] if record["status"] == "missing" else [record], "external_reference_count": 0}
    external = 0
    for model_id, choice in selected.items():
        for field in ("diagnostics", "selection_a1", "selection_a2", "confirmation"):
            for entry in choice.get(field, []):
                task_path, result_path = Path(entry["task_path"]).resolve(), Path(entry["result_path"]).resolve()
                pair = (task_path, result_path)
                if pair in seen:
                    continue
                seen.add(pair)
                is_external = not task_path.is_relative_to(output) or not result_path.is_relative_to(output)
                external += int(is_external)
                task, task_record = json_observation(task_path)
                result, result_record = json_observation(result_path)
                for item in (task_record, result_record):
                    item.update(kind="selected_reference", model_id=model_id, external=is_external)
                    checks.append(item)
                if task and result and (task.get("model_id") != model_id or not task.get("source_identity")
                        or result.get("source_identity") != task["source_identity"]
                        or result.get("task_sha256") != task_record.get("sha256")):
                    result_record.update(status="mismatch", reason="selected_reference_task_or_source_binding_mismatch")
                if entry.get("result_sha256") and result_record.get("sha256") != entry["result_sha256"]:
                    result_record.update(status="mismatch", reason="selected_reference_result_digest_mismatch")
    return {"checks": checks, "external_reference_count": external,
            "scope": "only explicit selected.json task/result references; external source identity remains original, outside this round's process/accounting/checkpoint ownership"}


def audit(prepared_path: Path, unit: str, *, round_path: Path | None = None,
          proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup"), runtime: Path | None = None,
          allow_conservative_debits=False, allow_inert_worker_estimates=False):
    started = time.time()
    observation_errors = []

    def observe(scope, function, fallback):
        try:
            return function()
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            observation_errors.append({"scope": scope, "status": "invalid",
                                       "reason": f"{type(error).__name__}: {error}"})
            return fallback

    prepared, prepared_observation = json_observation(prepared_path)
    prepared = prepared or {}
    if prepared and prepared.get("schema") != "pilot-r3-prepared-inputs-v1":
        prepared_observation.update(status="invalid", reason="unexpected_prepared_schema")
    launch_checks = []
    launch_candidates = [prepared_path.parent / name for name in
                         ("submission.json", "campaign-launch-draft.json", "launch-draft.json")]
    launch_path = next((path for path in launch_candidates if path.exists()), launch_candidates[-1])
    draft, draft_observation = json_observation(launch_path)
    if draft is not None:
        if draft.get("unit") and draft["unit"] != unit:
            launch_checks.append({"status": "invalid", "reason": "requested_unit_differs_from_frozen_launch",
                                  "requested": unit, "expected": draft["unit"]})
        launch_checks.append(file_observation(prepared_path, draft.get("prepared_sha256")))
        if draft.get("launcher"):
            launch_checks.append(file_observation(Path(draft["launcher"]), draft.get("launcher_sha256")))
        for field in ("plan", "preflight_report"):
            if draft.get(field+"_path"):
                launch_checks.append(file_observation(Path(draft[field+"_path"]), draft.get(field+"_sha256")))
    elif draft_observation["status"] != "missing":
        launch_checks.append(draft_observation)
    output = Path(round_path or prepared.get("output", prepared_path.parent / "round")).resolve()
    state, scheduler_observation = json_observation(output / "scheduler.json")
    state = state or {}
    if state and state.get("schema") != "pilot-r3-scheduler-v1":
        scheduler_observation.update(status="invalid", reason="unexpected_scheduler_schema")
    service = systemd_observation(unit)
    group = cgroup_observation(service, root=cgroup_root)
    table, errors = process_snapshot(proc_root)
    shared = observe("shared_groups", lambda: shared_group_observations(state, output, group, table,
                     allow_inert_worker_estimates=allow_inert_worker_estimates), {"groups": {}, "checks": []})
    processes = observe("processes", lambda: process_observation(state, service, group, table, errors),
                        {"status": "unknown", "released": None, "live_processes": [], "main_pid": 0})
    locks, lock_error = lock_snapshot(proc_root / "locks")
    runtime = runtime or Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    queue = observe("queue", lambda: queue_observation(state, processes, runtime=runtime, locks=locks, lock_error=lock_error),
                    {"status": "unknown", "released": None, "attempts": []})
    inputs = observe("inputs", lambda: input_observations(prepared), {"source_files": [], "data_files": []})
    artifacts = observe("artifacts", lambda: artifact_observations(state, prepared, output, shared),
                        {"retirement_manifest": {"status": "unknown"}, "files": []})
    selected = observe("selected", lambda: selected_observations(output), {"checks": []})
    accounting = observe("physical_accounting", lambda: shared_accounting_observation(state, prepared, output, shared,
                         allow_conservative_debits=allow_conservative_debits),
                         {"status": "unknown", "checks": []})
    final_state, final_scheduler_observation = json_observation(output / "scheduler.json")
    if scheduler_observation.get("sha256") != final_scheduler_observation.get("sha256"):
        scheduler_observation.update(status="changing", reason="scheduler_changed_during_audit")
    active = service.get("properties", {}).get("ActiveState")
    running = active in {"active", "activating", "deactivating", "reloading"} or processes["released"] is False or group["released"] is False
    terminal = (service["status"] == "observed" and active in {"inactive", "failed"}
                and processes["main_pid"] == 0 and processes["released"] is True and group["released"] is True)
    lifecycle = "running" if running else "terminal" if terminal else "unknown"
    checks = [prepared_observation, scheduler_observation, *launch_checks, *inputs["source_files"], *inputs["data_files"],
              artifacts["retirement_manifest"], *artifacts["files"], *shared["checks"], *selected["checks"], *accounting["checks"]]
    findings = observation_errors + [value for value in checks if value["status"] in PROBLEM]
    findings.extend(dict(row["queue_state"], scope="queue", task_id=row["task_id"])
                    for row in queue["attempts"] if row.get("queue_state", {}).get("status") in PROBLEM)
    for name, value in (("systemd", service), ("cgroup", group), ("processes", processes), ("queue", queue)):
        if value["status"] in PROBLEM:
            findings.append({"scope": name, "status": value["status"], "reason": value.get("reason", value.get("error"))})
    if lifecycle == "terminal" and state.get("status") not in SCHEDULER_TERMINAL:
        findings.append({"scope": "scheduler", "status": "unknown", "reason": "service_terminal_but_scheduler_not_terminal"})
    status = "running" if lifecycle == "running" else "passed" if lifecycle == "terminal" and queue["released"] is True and not findings else "incomplete"
    return {"schema": "pilot-r3-terminal-audit-v1", "status": status, "started_unix": started,
        "finished_unix": time.time(), "unit": unit, "round": str(output), "prepared": prepared_observation,
        "launch_binding": launch_checks,
        "lifecycle": {"status": lifecycle, "systemd": service, "cgroup": group, "processes": processes, "queue": queue},
        "execution": {"scheduler": scheduler_observation, "status": state.get("status"), "stage": state.get("stage"),
                      "blocking_reason": state.get("blocking_reason"), "counts": state.get("counts"),
                      "stage_b_gate": state.get("stage_b_gate", "not_recorded")},
        "inputs": inputs, "artifacts": artifacts, "shared_groups": shared, "selected_references": selected,
        "physical_accounting": accounting, "findings": findings,
        "finding_counts": dict(Counter(value["status"] for value in findings)),
        "interpretation": "passed means observed terminal release and recorded file integrity only; execution failure, numerical/discrete validity, recovery trajectory and learning remain separate conclusions"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--round", type=Path)
    parser.add_argument("--output", type=Path, help="optional docs/evidence/*.json review report")
    parser.add_argument("--allow-conservative-debits", action="store_true",
                        help="accept retained budget overcharges, record every excess, still reject any undercharge")
    parser.add_argument("--allow-inert-worker-estimates", action="store_true",
                        help="reconcile stopped shared-worker fallback fields only when proved excluded from all budget sums; retain the fields")
    args = parser.parse_args(argv)
    report = audit(args.prepared, args.unit, round_path=args.round,
                   allow_conservative_debits=args.allow_conservative_debits,
                   allow_inert_worker_estimates=args.allow_inert_worker_estimates)
    if args.output:
        from scripts.review_io import write_review_json
        write_review_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 2 if report["status"] == "running" else 1


if __name__ == "__main__":
    raise SystemExit(main())
