#!/usr/bin/env python3
"""Freeze, submit, inspect, or stop finite tide-learning-lab user services.

Resume is explicit: submit a NEW job/output with the training command's
--resume CHECKPOINT. This launcher never guesses application recovery state.
Resolved private paths are stored only in the chosen (normally ignored) output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.supervisor import (atomic_json, cgroup_processes, file_hash, identity_alive,
                                 now, read_json, source_inventory, verify_source)


def checked(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def freeze(args):
    repository = args.source.resolve()
    if checked(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=normal"]):
        raise ValueError("commit relevant source changes before freezing; dirty sources are not silently omitted")
    revision = checked(["git", "-C", str(repository), "rev-parse", f"{args.revision}^{{commit}}"])
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    archive = subprocess.Popen(["git", "-C", str(repository), "archive", "--format=tar", revision], stdout=subprocess.PIPE)
    try:
        subprocess.run(["tar", "-xf", "-", "-C", str(destination)], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait() != 0:
            raise RuntimeError("git archive failed")
        identity = {"schema": "tide-frozen-source-v1", "commit": revision,
                    "created_at": now(), "content": source_inventory(destination)}
        atomic_json(destination / ".tide-source.json", identity)
        # Preserve executability while making accidental writes fail. Job
        # outputs and caches belong outside this snapshot.
        for path in sorted(destination.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if not path.is_symlink():
                path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
        destination.chmod(0o555)
        print(json.dumps({"source": str(destination), "commit": revision,
                          "tree_sha256": identity["content"]["tree_sha256"]}, indent=2))
    except BaseException:
        archive.wait()
        # Preserve partial output for diagnosis; it cannot pass verify_source.
        raise


def systemd_quote(value):
    """Quote one systemd Exec* argv field (no shell is involved)."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def unit_info(unit):
    fields = ["Id", "LoadState", "ActiveState", "SubState", "Result", "ExecMainStatus", "MainPID",
              "Slice", "ControlGroup", "Transient", "CPUQuotaPerSecUSec", "MemoryMax", "RuntimeMaxUSec"]
    reply = subprocess.run(["systemctl", "--user", "show", unit, *[f"--property={field}" for field in fields]],
                           capture_output=True, text=True, check=False)
    values = dict(line.split("=", 1) for line in reply.stdout.splitlines() if "=" in line)
    values["inspection_exit_code"] = reply.returncode
    if reply.returncode:
        values["inspection_error"] = reply.stderr.strip()
    return values


def make_spec(args):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,95}", args.name):
        raise ValueError("name must be 1–96 lowercase letters, digits, dots, underscores or hyphens")
    if args.timeout <= 0 or args.threads <= 0 or args.memory_gib <= 0:
        raise ValueError("timeout, threads, and memory-gib must be positive finite budgets")
    if args.wait_seconds < 0 or args.wait_seconds >= args.timeout:
        raise ValueError("queue wait must be nonnegative and below the total timeout")
    if args.device == "npu" and (not args.queue_helper or args.wait_seconds <= 0):
        raise ValueError("NPU jobs require --queue-helper and a positive finite --wait-seconds")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("supply a workload argv after --")
    source = args.source.resolve()
    identity = verify_source(source)
    supervisor = source / "tide_lab/supervisor.py"
    if not supervisor.is_file():
        raise ValueError("frozen source does not contain tide_lab/supervisor.py")
    output = args.output.resolve()
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("job output must be outside frozen source")
    environment = {key: str(args.threads) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source))
    for assignment in args.env:
        key, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("--env requires NAME=VALUE")
        if any(word in key.upper() for word in ("TOKEN", "PASSWORD", "SECRET", "CREDENTIAL")):
            raise ValueError("--env is recorded; load credentials privately from the environment script instead")
        environment[key] = value
    def fixed_file(path):
        path = path.resolve(strict=True)
        return {"path": str(path), "sha256": file_hash(path)}
    return {"schema": "tide-job-v1", "job_id": args.name, "project": "tide-learning-lab",
            "unit": f"tide-lab-{args.name}.service", "created_at": now(),
            "source": {"path": str(source), "commit": identity["commit"],
                       "tree_sha256": identity["content"]["tree_sha256"]},
            "environment_script": fixed_file(args.env_script), "environment": environment,
            "queue_helper": fixed_file(args.queue_helper) if args.device == "npu" else None,
            "inputs": [fixed_file(path) for path in args.input], "device": args.device,
            "priority": args.priority, "command": command,
            "supervisor_python": str(Path(sys.executable).resolve()),
            "resources": {"threads": args.threads, "cpu_quota_percent": args.threads * 100,
                          "memory_bytes": args.memory_gib * 1024 ** 3,
                          "timeout_seconds": args.timeout, "wait_seconds": args.wait_seconds,
                          "stop_grace_seconds": 30, "npu_count": 1 if args.device == "npu" else 0},
            "paths": {"output": str(output), "log": str(output / "job.log"),
                      "state": str(output / "state.json"), "queue_state": str(output / "queue-state.json"),
                      "worker_state": str(output / "worker-state.json")},
            "resume_contract": "new job and output; pass the application --resume checkpoint explicitly"}


def submit(args):
    spec = make_spec(args)
    existing = unit_info(spec["unit"])
    if existing.get("LoadState") != "not-found":
        raise RuntimeError("unit already exists or the user manager is unavailable; choose a new job ID")
    output = Path(spec["paths"]["output"])
    output.mkdir(parents=True, exist_ok=False)
    record = output / "job.json"
    atomic_json(record, spec)
    atomic_json(output / "state.json", {"schema": "tide-job-lifecycle-v1", "job_id": spec["job_id"],
                                      "state": "submitted", "submitted_at": now(), "termination_verified": False})
    python = spec["supervisor_python"]
    supervisor = str(Path(spec["source"]["path"]) / "tide_lab/supervisor.py")
    finalizer = " ".join(systemd_quote(part) for part in [python, supervisor, "finalize", "--record", str(record)])
    command = ["systemd-run", "--user", f"--unit={spec['unit']}", "--service-type=exec",
               "--slice=background.slice", f"--working-directory={spec['source']['path']}",
               "--property=KillMode=control-group", "--property=TimeoutStopSec=30",
               "--property=SendSIGKILL=yes", "--property=OOMPolicy=stop",
               f"--property=RuntimeMaxSec={args.timeout}", f"--property=CPUQuota={args.threads * 100}%",
               f"--property=MemoryMax={spec['resources']['memory_bytes']}", "--property=Nice=10",
               f"--property=StandardOutput=append:{spec['paths']['log']}",
               f"--property=StandardError=append:{spec['paths']['log']}",
               f"--property=ExecStopPost={finalizer}", python, supervisor, "run", "--record", str(record)]
    atomic_json(output / "submission.json", {"argv": command, "time": now()})
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        atomic_json(output / "state.json", {"schema": "tide-job-lifecycle-v1", "job_id": spec["job_id"],
                   "state": "failed", "stage": "submission", "exit_code": completed.returncode,
                   "error": completed.stderr.strip(), "ended_at": now(), "termination_verified": True})
        raise RuntimeError(completed.stderr.strip())
    info = unit_info(spec["unit"])
    atomic_json(output / "submission-observation.json", {"time": now(), "systemd": info})
    if info.get("Slice") != "background.slice":
        subprocess.run(["systemctl", "--user", "stop", spec["unit"]], check=False)
        raise RuntimeError("service did not enter background.slice; stopped it")
    print(json.dumps({"job_id": spec["job_id"], "record": str(record), "systemd": info,
                      "state": read_json(output / "state.json")}, indent=2))


def locate(path):
    return path.resolve() / "job.json" if path.is_dir() else path.resolve()


def inspect(record):
    spec = read_json(record)
    if not spec or spec.get("schema") != "tide-job-v1":
        raise ValueError("expected a tide-job-v1 job.json or its directory")
    state = read_json(record.parent / "state.json", {})
    if state.get("stage") == "submission":
        return {"job_id": spec["job_id"], "state": state, "systemd": None,
                "termination_verified_now": True, "note": "No service was accepted for this submission."}
    info = unit_info(spec["unit"])
    group = info.get("ControlGroup") or state.get("control_group")
    pids = cgroup_processes(group)
    alive = [item for item in state.get("descendants", []) if identity_alive(item)]
    return {"job_id": spec["job_id"], "systemd": info, "state": state,
            "queue": read_json(record.parent / "queue-state.json"),
            "worker": read_json(record.parent / "worker-state.json"),
            "current_cgroup_processes": pids, "known_descendants_alive": alive,
            "termination_verified_now": info.get("MainPID", "0") == "0" and not pids and not alive}


def stop(args):
    record = locate(args.record)
    spec = read_json(record)
    if read_json(record.parent / "state.json", {}).get("stage") == "submission":
        raise ValueError("this submission was rejected; it owns no service to stop")
    atomic_json(record.parent / "stop-request.json", {"requested_at": now(), "method": "systemctl --user stop"})
    completed = subprocess.run(["systemctl", "--user", "stop", spec["unit"]], capture_output=True, text=True)
    observation = inspect(record)
    observation.update(stop_command_exit_code=completed.returncode, stopped_at=now())
    atomic_json(record.parent / "stop-observation.json", observation)
    print(json.dumps(observation, indent=2))
    if completed.returncode or not observation["termination_verified_now"]:
        raise RuntimeError("stop did not verify MainPID=0 and absence of all recorded/cgroup descendants")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    frozen = commands.add_parser("freeze", help="archive a clean committed tree; never includes uncommitted work")
    frozen.add_argument("--source", type=Path, required=True)
    frozen.add_argument("--destination", type=Path, required=True)
    frozen.add_argument("--revision", default="HEAD")
    launch = commands.add_parser("submit", help="submit one bounded user service")
    launch.add_argument("--name", required=True)
    launch.add_argument("--source", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--env-script", type=Path, required=True)
    launch.add_argument("--env", action="append", default=[], help="recorded nonsecret NAME=VALUE; repeatable")
    launch.add_argument("--device", choices=("cpu", "npu"), required=True)
    launch.add_argument("--timeout", type=int, required=True, help="total runtime including waiting; at most 30s stop grace")
    launch.add_argument("--wait-seconds", type=int, default=0)
    launch.add_argument("--threads", type=int, required=True)
    launch.add_argument("--memory-gib", type=int, required=True)
    launch.add_argument("--queue-helper", type=Path)
    launch.add_argument("--priority", type=int, default=0)
    launch.add_argument("--input", type=Path, action="append", default=[], help="additional immutable file to hash and verify")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("status", "logs", "stop"):
        item = commands.add_parser(name)
        item.add_argument("record", type=Path, help="job.json or job directory")
        if name == "logs":
            item.add_argument("--lines", type=int, default=80)
            item.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "freeze":
            freeze(args)
        elif args.action == "submit":
            submit(args)
        elif args.action == "status":
            print(json.dumps(inspect(locate(args.record)), indent=2))
        elif args.action == "stop":
            stop(args)
        elif args.action == "logs":
            spec = read_json(locate(args.record))
            return subprocess.call(["tail", "-n", str(args.lines), *(["-f"] if args.follow else []), spec["paths"]["log"]])
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
