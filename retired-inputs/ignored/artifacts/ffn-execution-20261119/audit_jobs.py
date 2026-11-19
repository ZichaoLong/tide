#!/usr/bin/env python3
"""Read saved FFN jobs and exact process identities; atomically write audits only.

No models, device API, process signalling, external argv/environment inspection,
or artifact removal. Active jobs are included in provisional resource totals but
their source/terminal audit is skipped. Run again when the owner requests it.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


SCHEMA = "tide-ffn-job-terminal-audit-v3"
TERMINAL_STATES = {"passed", "failed", "terminated", "cancelled", "stopped"}
UNIT_FIELDS = ("Id", "LoadState", "MainPID", "ExecMainPID", "ControlGroup",
               "ActiveState", "SubState", "Result", "ExecMainCode",
               "ExecMainStatus", "Slice", "CPUUsageNSec", "MemoryCurrent")


def now():
    return datetime.now(timezone.utc)


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def atomic(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    fd, stage = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".staged", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(stage):
            os.unlink(stage)


def fingerprint(path, root):
    data = path.read_bytes()
    return dict(path=str(path.relative_to(root)), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def read_json(path):
    return json.loads(path.read_text())


def properties(text):
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def unit_query(unit):
    if not re.fullmatch(r"tide-ffn-[a-z0-9-]+\.service", unit):
        raise ValueError("Refusing a unit outside this FFN audit scope")
    command = ["systemctl", "--user", "show", unit, "--property=" + ",".join(UNIT_FIELDS)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    return dict(command=command, exit_code=result.returncode,
                properties=properties(result.stdout), stderr=result.stderr.strip())


def process_stat(pid):
    """Read only an explicitly recorded PID's stat, never cmdline/environ."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                    start_ticks=int(fields[19]), rss_bytes=max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE"))
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity_check(identity):
    actual = process_stat(identity["pid"])
    answer = dict(pid=identity["pid"], start_ticks=identity["start_ticks"], observed_process=actual)
    if actual is None:
        answer.update(terminal=True, reason="recorded_pid_absent")
    elif actual["start_ticks"] != identity["start_ticks"]:
        answer.update(terminal=True, reason="pid_reused_original_birth_identity_absent")
    elif actual["state"] in {"Z", "X", "x"}:
        answer.update(terminal=True, reason="recorded_identity_exited_or_zombie")
    else:
        answer.update(terminal=False, reason="recorded_identity_still_live")
    return answer


def cgroup_mounts():
    result = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, extra = left.split(), right.split()
        if extra[0] not in {"cgroup", "cgroup2"}:
            continue
        decode = lambda value: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)
        result.append(dict(type=extra[0], root=decode(fields[3]),
                           mountpoint=decode(fields[4]), options=extra[2]))
    return result


def check_cgroups(original, current, mounts):
    systemd_mounts = [m for m in mounts if m["type"] == "cgroup2" or "name=systemd" in m["options"].split(",")]
    paths = sorted({p for p in (original, current) if p})
    checks = []
    for path in paths:
        for mount in systemd_mounts:
            relative = Path(path).relative_to(mount["root"])
            location = Path(mount["mountpoint"]) / relative
            row = dict(hierarchy_path=path, filesystem_path=str(location),
                       mount=mount, exists=location.exists(), process_files={}, errors=[])
            if location.exists():
                def error(exc):
                    row["errors"].append(str(exc))
                for base, dirs, names in os.walk(location, followlinks=False, onerror=error):
                    if "cgroup.procs" in names:
                        file = Path(base) / "cgroup.procs"
                        try:
                            row["process_files"][str(file)] = sorted({int(p) for p in file.read_text().split()})
                        except FileNotFoundError:
                            row["process_files"][str(file)] = []
                        except OSError as exc:
                            row["errors"].append(str(exc))
            pids = sorted({p for values in row["process_files"].values() for p in values})
            row["processes"] = [process_stat(p) or dict(pid=p, state="exited_during_read") for p in pids]
            row["empty_or_absent"] = not pids and not row["errors"]
            checks.append(row)
    return dict(state="empty_or_absent" if checks and all(c["empty_or_absent"] for c in checks) else "unresolved",
                mounts=systemd_mounts, checks=checks,
                scope="Systemd hierarchy, recursively including nested cgroup.procs; legacy controller ancestor groups are not per-job groups.")


def git_tree(root, revision, cache):
    if revision not in cache:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Expected a fixed full source commit")
        result = subprocess.run(["git", "ls-tree", "-r", "-z", "--full-tree", revision],
                                cwd=root, check=True, capture_output=True, timeout=30)
        entries = []
        for value in result.stdout.split(b"\0"):
            if not value:
                continue
            identity, path = value.split(b"\t", 1)
            mode, kind, blob = identity.decode().split()
            entries.append(dict(path=os.fsdecode(path), mode=mode, type=kind, blob=blob))
        cache[revision] = entries
    return cache[revision]


def source_check(root, run, launch, tree_cache):
    source = run / "source"
    if Path(launch["source_path"]).resolve() != source.resolve():
        raise ValueError("Frozen source is not the job-owned source directory")
    entries = git_tree(root, launch["source"], tree_cache)
    files, paths = [], set()
    for entry in entries:
        relative = entry["path"]
        paths.add(relative)
        file = source / relative
        record = dict(entry)
        try:
            info = file.lstat()
            if entry["type"] != "blob":
                raise ValueError("Non-blob Git entry needs separate verification")
            if stat.S_ISLNK(info.st_mode):
                content = os.fsencode(os.readlink(file))
                actual_mode = "120000"
            elif stat.S_ISREG(info.st_mode) and file.resolve().is_relative_to(source.resolve()):
                content = file.read_bytes()
                actual_mode = "100755" if info.st_mode & 0o111 else "100644"
            else:
                raise ValueError("Not a regular in-tree file or tracked symlink")
            digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            record.update(actual_blob=digest, actual_mode=actual_mode, bytes=len(content),
                          matches=digest == entry["blob"] and actual_mode == entry["mode"])
        except (OSError, ValueError) as exc:
            record.update(matches=False, error=str(exc))
        files.append(record)
    extras = []
    trace_requested = "--trace" in launch.get("arguments", {}).get("command", [])
    for base, dirs, names in os.walk(source, followlinks=False):
        for name in names:
            file = Path(base) / name
            relative = str(file.relative_to(source))
            if relative not in paths:
                if "__pycache__" in file.parts:
                    kind = "python_bytecode_cache"
                elif ".pytest_cache" in file.parts:
                    kind = "pytest_runtime_cache"
                elif (trace_requested and relative.startswith("export_only_prof_dir/")
                      and file.suffix not in {".py", ".pyc", ".so", ".sh", ".a", ".c", ".cpp", ".h"}):
                    kind = "requested_torch_profiler_export"
                else:
                    kind = "unexpected_untracked_file"
                extras.append(dict(path=relative, kind=kind,
                                   generated_runtime_output=kind != "unexpected_untracked_file"))
    mismatches = [v["path"] for v in files if not v["matches"]]
    unexpected = [v["path"] for v in extras if not v["generated_runtime_output"]]
    content_digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(state="matches_source_commit" if not mismatches and not unexpected else "mismatch_or_unexpected_files",
                source_commit=launch["source"], path=str(source.relative_to(root)),
                tracked_files=len(files), tracked_bytes=sum(v.get("bytes", 0) for v in files),
                per_file_verification_sha256=content_digest, mismatches=mismatches,
                unexpected_extra_files=unexpected, generated_extras=extras, files=files,
                method="Each tracked file's Git blob hash and executable/symlink mode compared with git ls-tree of the fixed commit; Python/pytest caches and explicitly requested profiler export outputs are individually listed separately.")


def audit_terminal(root, run, launch, status, query, mounts, cache):
    original = properties(launch.get("unit_observation", ""))
    current = query["properties"]
    descendants = [identity_check(p) for p in status.get("descendant_identities", [])]
    supervisor = process_stat(status["pid"])
    supervisor_terminal = supervisor is None or supervisor["state"] in {"Z", "X", "x"}
    cgroups = check_cgroups(original.get("ControlGroup"), current.get("ControlGroup"), mounts)
    source = source_check(root, run, launch, cache)
    inputs = {name: fingerprint(run / (name + ".json"), root) for name in ("launch", "status")}
    complete_identity_list = bool(descendants) and status.get("child_pid") in {p["pid"] for p in descendants}
    background = original.get("Slice") == "background.slice" and "/background.slice/" in original.get("ControlGroup", "")
    unit_terminal = query["exit_code"] == 0 and current.get("MainPID") == "0" and current.get("ActiveState") in {"inactive", "failed"}
    identity_terminal = complete_identity_list and all(p["terminal"] for p in descendants)
    terminal = bool(unit_terminal and supervisor_terminal and identity_terminal and cgroups["state"] == "empty_or_absent")
    passed = terminal and background and source["state"] == "matches_source_commit" and status.get("source") == launch["source"]
    audit = dict(schema=SCHEMA, state="terminal" if terminal else "terminal_unconfirmed", passed=passed,
                 observed_at=now().isoformat(), unit=current.get("Id"), unit_query=query,
                 outer_state=status["state"], outer_exit_code=status.get("exit_code"),
                 source_inputs=inputs, original_unit_observation=original,
                 original_background_slice_verified=background,
                 supervisor=dict(pid=status["pid"], current=supervisor, terminal=supervisor_terminal,
                                 birth_identity_recorded=False,
                                 note="Supervisor start_ticks were not saved; a still-present non-zombie PID is treated as unresolved, never presumed reused."),
                 complete_recorded_descendant_list=complete_identity_list,
                 descendants=descendants, cgroup=cgroups, frozen_source=source,
                 interpretation="Terminal verifies lifecycle, including failed jobs; it does not qualify an experiment. A garbage-collected unit's Result=success alone is not execution evidence.")
    destination = run / "terminal-audit.json"
    disposition = "written"
    if destination.exists():
        prior = read_json(destination)
        same = (prior.get("schema") == SCHEMA and prior.get("passed") is True and passed and
                prior.get("source_inputs") == inputs and
                prior.get("frozen_source", {}).get("per_file_verification_sha256") == source["per_file_verification_sha256"])
        if same:
            disposition = "reused_after_fresh_unit_identity_cgroup_and_source_recheck"
        else:
            audit["prior_terminal_audit"] = dict(file=fingerprint(destination, root), record=prior)
    if disposition == "written":
        atomic(destination, audit)
    return dict(disposition=disposition, file=fingerprint(destination, root),
                terminal=terminal, audit_passed=passed, recorded_descendants=len(descendants),
                source_state=source["state"], tracked_files=source["tracked_files"],
                generated_extra_files=len(source["generated_extras"]),
                mismatches=source["mismatches"], unexpected_extra_files=source["unexpected_extra_files"],
                checked_at=audit["observed_at"], unit_state=current.get("ActiveState"),
                cgroup_state=cgroups["state"], supervisor_terminal=supervisor_terminal,
                all_recorded_descendants_terminal=identity_terminal)


def resource_row(root, run, launch, status, observed):
    arguments = launch["arguments"]
    terminal = status.get("state") in TERMINAL_STATES and bool(status.get("finished"))
    elapsed = float(status.get("elapsed", 0))
    if not terminal and status.get("started"):
        elapsed = max(elapsed, (observed - stamp(status["started"])).total_seconds())
    row = dict(name=run.name, path=str(run.relative_to(root)), source=launch["source"],
               outer_state=status.get("state"), outer_exit_code=status.get("exit_code"),
               started=status.get("started"), finished=status.get("finished"),
               npu_count=arguments.get("npu_count", 0), threads=arguments.get("threads"),
               worker_requested_affinity=arguments.get("affinity"),
               worker_affinity_scope="Original launch record; not modified or re-probed by this audit.",
               service_elapsed_seconds_including_waits=elapsed,
               service_elapsed_provisional=not terminal,
               peak_process_tree_rss_bytes=status.get("peak_rss_bytes"),
               sampled_rss_bound_bytes=status.get("max_rss_bytes"),
               rss_sample_count=status.get("rss_sample_count"),
               rss_scope="Sampled sum across owned workload processes; shared pages can be counted more than once; not cgroup MemoryPeak or a hard memory limit.")
    row["sampled_rss_within_recorded_bound"] = (status.get("peak_rss_bytes", 0) <= status.get("max_rss_bytes", 0))
    queue_path = run / "queue.json"
    if queue_path.exists():
        queue = read_json(queue_path)
        queue_terminal = queue.get("status") in {"completed", "failed", "cancelled", "timeout", "timed_out", "terminated"}
        queue_elapsed = float(queue.get("elapsed_seconds", 0))
        if not queue_terminal and queue.get("timestamp"):
            queue_elapsed += max(0, (observed - stamp(queue["timestamp"])).total_seconds())
        row["queue"] = dict(status=queue.get("status"), return_code=queue.get("return_code"),
                            physical_ids=queue.get("physical_ids", []),
                            elapsed_seconds_including_admission_wait=queue_elapsed,
                            elapsed_provisional=not queue_terminal, file=fingerprint(queue_path, root))
    else:
        row["queue"] = None
    dependency_path = run / "dependency.json"
    wait = 0.0
    if dependency_path.exists():
        dependency = read_json(dependency_path)
        wait = float(dependency.get("elapsed_seconds", 0))
        wait_scope = "Recorded released dependency-wait duration."
        if dependency.get("state") != "released" and dependency.get("started"):
            boundary = stamp(status["finished"]) if terminal else observed
            wait = max(wait, (boundary - stamp(dependency["started"])).total_seconds())
            wait_scope = "Wait record remained waiting; conservative estimate through outer-service finish/current snapshot, including shutdown lag. Original wait record unchanged."
        row["dependency_wait"] = dict(file=fingerprint(dependency_path, root), state=dependency.get("state"),
                                       elapsed_seconds=wait, scope=wait_scope)
    else:
        row["dependency_wait"] = None
    row["post_dependency_service_seconds_upper_bound"] = max(0.0, elapsed - wait)
    row["post_dependency_service_scope"] = "Service minus recorded/estimated predecessor wait; includes runtime startup and NPU admission wait, so this is not pure worker compute time."
    reason_path = run / "resubmission-reason.json"
    if reason_path.exists():
        reason = read_json(reason_path)
        row["resubmission"] = dict(file=fingerprint(reason_path, root), record=reason,
                                    queue_record_exists=queue_path.exists())
    row["worker_and_allocator"] = saved_worker_and_allocator(root, run, bool(row["npu_count"]))
    affinity = run / "supervisor-affinity.json"
    if affinity.exists():
        original = read_json(affinity)
        row["supervisor_affinity_adjustment"] = dict(file=fingerprint(affinity, root),
            record={k:original.get(k) for k in ("observed", "pid", "old_affinity", "new_affinity", "scope")})
    return row


def saved_worker_and_allocator(root, run, is_npu):
    """Read saved result JSON only; retain timings and allocator boundaries separately."""
    excluded = {"launch.json", "status.json", "queue.json", "dependency.json", "environment.json",
                "terminal-audit.json", "supervisor-affinity.json", "resubmission-reason.json"}
    files, programme, timed = [], [], []
    maxima, witnesses = {}, {}
    allocator_count = 0
    violations = []
    def inspect(value, pointer, file, budgets):
        nonlocal allocator_count
        if isinstance(value, dict):
            keys = ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")
            fields = {k:v for k,v in value.items() if k in keys and isinstance(v, (int, float))}
            # Storage/parameter counts are not allocator measurements. Recovery
            # producers also save allocator payloads under process-memory paths:
            # require their exact NPU device key and all four standard fields.
            named_allocator = any("allocator" in str(p) for p in pointer)
            npu_device_allocator = (is_npu and bool(pointer)
                and isinstance(pointer[-1], str)
                and re.fullmatch(r"npu:\d+", pointer[-1]) is not None
                and set(fields) == set(keys))
            if fields and (named_allocator or npu_device_allocator):
                allocator_count += 1
                for key, amount in fields.items():
                    if key not in maxima or amount > maxima[key]:
                        maxima[key] = amount
                        witnesses[key] = dict(file=file, json_pointer="/" + "/".join(map(str, pointer)) + "/" + key)
                    kind = "reserved" if "reserved" in key else "allocated"
                    if is_npu and amount > budgets[kind]:
                        violations.append(dict(file=file, json_pointer="/" + "/".join(map(str, pointer)),
                                               field=key, observed_bytes=amount, budget_bytes=budgets[kind]))
            for key, item in value.items():
                inspect(item, (*pointer, key), file, budgets)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect(item, (*pointer, index), file, budgets)
    for base, dirs, names in os.walk(run, followlinks=False):
        dirs[:] = [name for name in dirs if name not in {"source", "__pycache__", ".pytest_cache"}]
        for name in sorted(names):
            if not name.endswith(".json") or name in excluded:
                continue
            path = Path(base) / name
            data = read_json(path)
            if not isinstance(data, dict):
                continue
            fp = fingerprint(path, root)
            arguments = data.get("arguments", {})
            budgets = {kind:int((arguments.get(kind + "_budget_gib") or 56) * 2**30) for kind in ("allocated", "reserved")}
            before = allocator_count
            inspect(data, (), fp, budgets)
            ordinary, warm = [], []
            normal = data.get("normal")
            if isinstance(normal, list):
                ordinary = normal
                warm = data.get("warmup", [])
            elif isinstance(normal, dict):
                ordinary = normal.get("observations", [])
                warm = normal.get("warmup", [])
            normal_values = [v["complete_seconds"] for v in ordinary if isinstance(v, dict) and "complete_seconds" in v]
            warm_values = [v["complete_seconds"] for v in warm if isinstance(v, dict) and "complete_seconds" in v]
            if normal_values or warm or ordinary:
                timed.append(dict(file=fp, result_state=data.get("state"), normal_samples_with_recorded_seconds=len(normal_values),
                    normal_complete_seconds_sum=sum(normal_values), warmup_samples_recorded=len(warm),
                    warmup_samples_with_recorded_seconds=len(warm_values),
                    warmup_complete_seconds_sum=sum(warm_values) if warm_values else None,
                    normal_pass_seconds_including_excluded_work=normal.get("pass_elapsed_seconds_including_excluded_work") if isinstance(normal, dict) else None))
            if path == run / "results/suite.json":
                children = [r for r in data.get("rows", []) if "elapsed_seconds" in r]
                programme.append(dict(file=fp, state=data.get("state"), elapsed_seconds=data.get("elapsed_seconds"),
                    child_count_with_recorded_elapsed=len(children),
                    child_elapsed_seconds_sum=sum(r["elapsed_seconds"] for r in children),
                    scope="Recorded serial worker programme and subprocess wall intervals include model construction, warmup, normal, checks, diagnostics and IO. They exclude predecessor waits and initial device admission; not pure compute time."))
            if allocator_count > before or normal_values or warm_values or path.name == "suite.json":
                files.append(fp)
    return dict(evidence_files=files, worker_programmes=programme, timed_sections=timed,
        recorded_worker_programme_elapsed_seconds=sum(p["elapsed_seconds"] for p in programme if p["elapsed_seconds"] is not None) if programme else None,
        recorded_serial_worker_child_elapsed_seconds=sum(p["child_elapsed_seconds_sum"] for p in programme) if programme else None,
        normal_complete_seconds_recorded_sum=sum(t["normal_complete_seconds_sum"] for t in timed),
        warmup_complete_seconds_recorded_sum=sum(t["warmup_complete_seconds_sum"] or 0 for t in timed),
        worker_elapsed_coverage="Serial suite intervals only; standalone tests/probes/qualification often do not record full worker wall time. Missing duration is unavailable, not zero. Timed normal/warmup sections are separately scoped, incomplete lower bounds of worker time and are not added to programme elapsed.",
        allocator_observations=allocator_count, device_allocator_maxima=maxima if is_npu and allocator_count else None,
        max_witnesses=witnesses, allocator_budget_violations=violations,
        observed_allocator_within_budget=not violations if is_npu and allocator_count else None,
        allocator_scope="Observed per-process NPU allocator values compared with each result's allocated/reserved limits (56 GiB plan fallback). Not a hard cap or simultaneous parent+child aggregate device occupancy. CPU empty allocator is unavailable; host RSS is separate.")


def disk_inventory(root, roots, output):
    seen, rows, total_apparent, total_allocated, errors = set(), [], 0, 0, []
    for directory in roots:
        apparent = allocated = count = duplicates = 0
        for base, dirs, names in os.walk(directory, followlinks=False):
            for path in [Path(base)] + [Path(base) / n for n in names] + [Path(base) / n for n in dirs if (Path(base) / n).is_symlink()]:
                if path == output:
                    continue
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    errors.append(dict(path=str(path.relative_to(root)), reason="disappeared_during_inventory"))
                    continue
                identity = (info.st_dev, info.st_ino)
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)
                count += 1
                if not stat.S_ISDIR(info.st_mode):
                    apparent += info.st_size
                allocated += info.st_blocks * 512
        total_apparent += apparent
        total_allocated += allocated
        rows.append(dict(path=str(directory.relative_to(root)), apparent_file_bytes=apparent,
                         allocated_bytes=allocated, unique_inodes=count, duplicate_hardlinks_skipped=duplicates))
    return dict(observed_at=now().isoformat(), scope=[str(p.relative_to(root)) for p in roots],
                apparent_file_bytes_excluding_this_resource_report=total_apparent,
                allocated_bytes_excluding_this_resource_report=total_allocated,
                per_directory=rows, races=errors,
                method="Current lstat inventory; no symlink traversal; hardlinks deduplicated by device/inode; allocated bytes include directories. Active files can grow after observation. This is current artifact scope size, not a whole-filesystem historical net delta.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / "artifacts/ffn-execution-20261119/resource-audit.json"
    plan = read_json(destination.parent / "plan.json")
    observed, mounts, cache = now(), cgroup_mounts(), {}
    roots = sorted(p for p in (root / "artifacts").glob("ffn-*") if p.is_dir() and not p.is_symlink())
    extra = root / "artifacts/grouped-ffn-development"
    if extra.is_dir() and not extra.is_symlink():
        roots.append(extra)
    jobs, other = [], []
    for run in roots:
        launch_path, status_path = run / "launch.json", run / "status.json"
        if not launch_path.exists() or not status_path.exists():
            other.append(str(run.relative_to(root)))
            continue
        launch, status = read_json(launch_path), read_json(status_path)
        row = resource_row(root, run, launch, status, observed)
        unit = launch["unit"] if launch["unit"].endswith(".service") else launch["unit"] + ".service"
        query = unit_query(unit)
        row["unit_observation"] = query
        claimed_terminal = status.get("state") in TERMINAL_STATES and bool(status.get("finished"))
        active = query["properties"].get("ActiveState") in {"active", "activating", "deactivating", "reloading"}
        if not claimed_terminal or active:
            row["terminal_audit"] = dict(disposition="skipped_active_or_not_yet_terminal",
                                         frozen_source_checked=False)
        else:
            row["terminal_audit"] = audit_terminal(root, run, launch, status, query, mounts, cache)
        jobs.append(row)
    cpu = sum(j["service_elapsed_seconds_including_waits"] for j in jobs if not j["npu_count"])
    all_services = sum(j["service_elapsed_seconds_including_waits"] for j in jobs)
    queue = sum(j["queue"]["elapsed_seconds_including_admission_wait"] * j["npu_count"] for j in jobs if j["queue"])
    npu_services = sum(j["service_elapsed_seconds_including_waits"] * j["npu_count"] for j in jobs)
    waits = sum((j["dependency_wait"] or {}).get("elapsed_seconds", 0) for j in jobs)
    workers = [j["worker_and_allocator"] for j in jobs]
    npu_observed = [j for j in jobs if j["npu_count"] and j["worker_and_allocator"]["allocator_observations"]]
    disk = disk_inventory(root, roots, destination)
    skipped = [j["name"] for j in jobs if j["terminal_audit"]["disposition"].startswith("skipped")]
    failed = [j["name"] for j in jobs if j["terminal_audit"].get("audit_passed") is False]
    report = dict(schema="tide-ffn-resource-and-lifecycle-audit-v1",
        state="partial_active_jobs_skipped" if skipped else "completed_with_audit_findings" if failed else "completed_read_audit",
        observed_at=observed.isoformat(), finished_at=now().isoformat(),
        audit_script=fingerprint(Path(__file__).resolve(), root),
        plan=fingerprint(destination.parent / "plan.json", root), budgets=plan["budget"],
        requested_job_count=29, discovered_job_count=len(jobs),
        requested_job_count_matches=len(jobs) == 29,
        totals=dict(cpu_only_service_elapsed_seconds_including_waits=cpu,
                    cpu_only_service_elapsed_hours_including_waits=cpu / 3600,
                    all_service_elapsed_seconds_including_waits=all_services,
                    npu_queue_device_seconds_including_admission_wait=queue,
                    npu_queue_device_hours_including_admission_wait=queue / 3600,
                    npu_service_device_seconds_including_dependency_and_admission_wait=npu_services,
                    predecessor_wait_seconds_all_jobs=waits,
                    cpu_predecessor_wait_seconds=sum((j["dependency_wait"] or {}).get("elapsed_seconds", 0) for j in jobs if not j["npu_count"]),
                    npu_predecessor_wait_seconds_before_admission_not_device_use=sum((j["dependency_wait"] or {}).get("elapsed_seconds", 0) for j in jobs if j["npu_count"]),
                    cpu_post_dependency_service_seconds_upper_bound=sum(j["post_dependency_service_seconds_upper_bound"] for j in jobs if not j["npu_count"]),
                    npu_post_dependency_service_device_seconds_upper_bound=sum(j["post_dependency_service_seconds_upper_bound"] * j["npu_count"] for j in jobs),
                    cpu_recorded_worker_programme_seconds=sum(j["worker_and_allocator"]["recorded_worker_programme_elapsed_seconds"] or 0 for j in jobs if not j["npu_count"]),
                    npu_recorded_worker_programme_device_seconds=sum((j["worker_and_allocator"]["recorded_worker_programme_elapsed_seconds"] or 0) * j["npu_count"] for j in jobs),
                    recorded_normal_timed_seconds_all_jobs=sum(w["normal_complete_seconds_recorded_sum"] for w in workers),
                    recorded_warmup_timed_seconds_all_jobs=sum(w["warmup_complete_seconds_recorded_sum"] for w in workers),
                    jobs_without_full_worker_programme_duration=[j["name"] for j in jobs if j["worker_and_allocator"]["recorded_worker_programme_elapsed_seconds"] is None],
                    npu_jobs_with_saved_allocator_observations=len(npu_observed),
                    npu_jobs_without_saved_allocator_observations=[j["name"] for j in jobs if j["npu_count"] and not j["worker_and_allocator"]["allocator_observations"]],
                    all_observed_npu_allocator_values_within_recorded_limits=all(j["worker_and_allocator"]["observed_allocator_within_budget"] for j in npu_observed),
                    max_observed_npu_peak_allocated_bytes=max((j["worker_and_allocator"]["device_allocator_maxima"].get("peak_allocated_bytes", 0) for j in npu_observed), default=None),
                    max_observed_npu_peak_reserved_bytes=max((j["worker_and_allocator"]["device_allocator_maxima"].get("peak_reserved_bytes", 0) for j in npu_observed), default=None),
                    max_per_job_sampled_process_tree_rss_bytes=max((j["peak_process_tree_rss_bytes"] or 0 for j in jobs), default=0),
                    cpu_elapsed_within_budget=cpu <= plan["budget"]["cpu_job_hours"] * 3600,
                    npu_queue_elapsed_within_budget=queue <= plan["budget"]["npu_device_hours"] * 3600,
                    all_jobs_sampled_rss_within_recorded_bound=all(j["sampled_rss_within_recorded_bound"] for j in jobs),
                    all_jobs_requested_threads_within_plan=all((j["threads"] or 0) <= plan["budget"]["max_cpu_threads"] for j in jobs),
                    all_jobs_recorded_rss_bound_within_plan=all((j["sampled_rss_bound_bytes"] or 0) <= plan["budget"]["max_rss_gib"] * 2**30 for j in jobs)),
        skipped_active_jobs=skipped, audit_findings_jobs=failed, jobs=jobs, disk=disk,
        non_job_artifact_directories=other,
        scope=["Read-only checks except this script's atomic terminal-audit.json and resource-audit.json outputs.",
               "Terminal failure is valid lifecycle evidence; supervisor/queue success does not replace inner numerical or performance results.",
               "CPU duration sums CPU-only service wall elapsed, conservatively including dependency waits; not CPU core-hours or utilization.",
               "NPU queue duration includes admission waits and setup; multiplied by requested device count, not claimed occupied device compute time. NPU service elapsed additionally includes predecessor waits.",
               "Dependency wait, conservative post-dependency service, recorded serial worker-programme wall intervals and timed normal/warmup sections have separate boundaries and must not be added together. Unrecorded standalone-worker durations remain unavailable.",
               "Cancelled v1 attempts remain included in service/wait accounting; absence of a queue record before cancellation does not invent a device workload. Replacement v2 jobs retain separate identities.",
               "Allocator limits apply only to recorded per-process device observations; they are neither hard caps nor a claim about unrecorded concurrent parent/child occupancy. CPU allocator absence is not zero HBM.",
               "Active jobs contribute provisional elapsed/RSS/disk observations but receive no terminal/source verification.",
               "Supervisor affinity changes are separate saved records; no worker affinity or input is changed.",
               "No external full argv, sensitive environment, model/device execution, process signals, or artifact cleanup."])
    disk["budget_bytes"] = int(plan["budget"]["new_artifacts_gib"] * 2**30)
    block = os.statvfs(destination.parent).f_frsize
    for _ in range(6):
        size = len((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
        rounded = ((size + block - 1) // block) * block
        disk.update(resource_report_serialized_bytes=size,
                    apparent_file_bytes_including_this_report=disk["apparent_file_bytes_excluding_this_resource_report"] + size,
                    conservative_bytes_including_report=max(disk["apparent_file_bytes_excluding_this_resource_report"], disk["allocated_bytes_excluding_this_resource_report"]) + rounded)
        disk["within_64_gib_budget"] = disk["conservative_bytes_including_report"] <= disk["budget_bytes"]
    atomic(destination, report)
    actual_size = destination.stat().st_size
    if actual_size != disk["resource_report_serialized_bytes"]:
        raise RuntimeError("Report-size accounting did not reach a fixed point")
    print(json.dumps(dict(path=str(destination.relative_to(root)), state=report["state"],
        jobs=len(jobs), terminal_audits=sum(j["terminal_audit"].get("audit_passed", False) for j in jobs),
        skipped_active_jobs=skipped, audit_findings_jobs=failed, totals=report["totals"],
        disk_conservative_gib=disk["conservative_bytes_including_report"] / 2**30,
        disk_within_budget=disk["within_64_gib_budget"], file=fingerprint(destination, root)), ensure_ascii=False))


if __name__ == "__main__":
    main()
