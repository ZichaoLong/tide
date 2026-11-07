"""Small, stdlib-only lifecycle supervisor for frozen research jobs.

This records execution, never scientific acceptance. The systemd finalizer
also records failures that cannot be caught by Python (OOM or SIGKILL).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


TERMINAL = {"completed", "failed", "timed_out", "stopped", "interrupted"}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(directory):
    """Include every source file; caches and the identity record are excluded."""
    directory = Path(directory).resolve()
    files = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if "__pycache__" in relative.parts or path.suffix == ".pyc" or str(relative) == ".tide-source.json":
            continue
        if path.is_symlink():
            # An external mutable target would defeat the frozen-source claim.
            try:
                path.resolve(strict=True).relative_to(directory)
            except (ValueError, FileNotFoundError):
                raise ValueError(f"frozen source has an external or broken symlink: {relative}")
            digest = hashlib.sha256(os.readlink(path).encode()).hexdigest()
            kind = "symlink"
        elif path.is_file():
            digest, kind = file_hash(path), "file"
        else:
            continue
        files.append({"path": str(relative), "sha256": digest, "kind": kind,
                      "executable": bool(path.stat().st_mode & 0o111)})
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"tree_sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def verify_source(directory):
    directory = Path(directory).resolve()
    identity = read_json(directory / ".tide-source.json")
    if not identity or identity.get("schema") != "tide-frozen-source-v1":
        raise ValueError("source needs .tide-source.json; create it with scripts/job.py freeze")
    current = source_inventory(directory)
    if current != identity.get("content"):
        raise ValueError("frozen source differs from its recorded identity")
    return identity


def process_identity(pid):
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return {"pid": int(pid), "start_ticks": int(fields[19]), "state": fields[0],
                "parent_pid": int(fields[1]), "process_group": int(fields[2])}
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        return None


def identity_alive(identity):
    actual = process_identity(identity["pid"])
    return bool(actual and actual["start_ticks"] == identity["start_ticks"] and actual["state"] != "Z")


def group_processes(group):
    if not group:
        return []
    result = []
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            item = process_identity(int(directory.name))
            if item and item["process_group"] == group and item["state"] != "Z":
                result.append(item)
    return sorted(result, key=lambda item: item["pid"])


def current_cgroup():
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, value = line.split(":", 2)
        if hierarchy == "0" or "name=systemd" in controllers:
            return value
    return None


def cgroup_processes(group):
    if not group:
        return []
    # This host uses cgroup v2. A missing retired group is empty, not a running job.
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    pids = set()
    try:
        paths = [root / "cgroup.procs", *root.rglob("cgroup.procs")]
        for path in paths:
            try:
                pids.update(int(value) for value in path.read_text().split())
            except FileNotFoundError:
                pass
    except PermissionError:
        raise RuntimeError(f"cannot inspect job control group: {group}")
    return [item for pid in sorted(pids) if (item := process_identity(pid)) and item["state"] != "Z"]


def event(directory, kind, **fields):
    with (Path(directory) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), "event": kind, **fields}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_inputs(spec):
    identity = verify_source(spec["source"]["path"])
    if (identity["commit"] != spec["source"]["commit"] or
            identity["content"]["tree_sha256"] != spec["source"]["tree_sha256"]):
        raise ValueError("frozen source identity changed after submission")
    for item in [spec["environment_script"], *spec.get("inputs", [])]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError(f"fixed input changed: {item['path']}")
    if spec.get("queue_helper") and file_hash(spec["queue_helper"]["path"]) != spec["queue_helper"]["sha256"]:
        raise ValueError("queue helper changed since submission")


def normalized_code(code):
    return code if code >= 0 else 128 - code


def worker(record):
    """Run beneath the queue so allocation and workload exit are separate facts."""
    spec = read_json(record)
    output = Path(record).parent
    started = time.monotonic()
    state = {"schema": "tide-job-worker-v1", "job_id": spec["job_id"], "state": "starting",
             "started_at": now(), "worker": process_identity(os.getpid()),
             "physical_visibility": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
             "logical_device": "npu:0" if spec["device"] == "npu" else "cpu"}
    child = None
    received = None

    def request_stop(signum, _frame):
        nonlocal received
        if received is None:
            received = signum
            if child is not None and child.poll() is None:
                child.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    try:
        atomic_json(output / "worker-state.json", state)
        child = subprocess.Popen(spec["command"], cwd=spec["source"]["path"])
        state.update(state="running", process=process_identity(child.pid))
        atomic_json(output / "worker-state.json", state)
        code = child.wait()
        state.update(state="interrupted" if received else ("completed" if code == 0 else "failed"),
                     exit_code=code, signal=received)
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}", exit_code=None)
        code = 1
    finally:
        state.update(ended_at=now(), elapsed_seconds=time.monotonic() - started)
        atomic_json(output / "worker-state.json", state)
    return 128 + received if received else normalized_code(code)


def run(record):
    spec = read_json(record)
    output = Path(record).parent
    started = time.monotonic()
    state = {"schema": "tide-job-lifecycle-v1", "job_id": spec["job_id"], "state": "starting",
             "started_at": now(), "supervisor": process_identity(os.getpid()),
             "control_group": current_cgroup(), "descendants": [], "termination_verified": False}
    child = None
    received = None
    timed_out = False
    signal_time = None
    result = 1

    def request_stop(signum, _frame=None):
        nonlocal received, signal_time
        if received is None:
            received, signal_time = signum, time.monotonic()
            if child is not None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    try:
        atomic_json(output / "state.json", state)
        validate_inputs(spec)
        if received is not None:
            state.update(state="stopped" if (output / "stop-request.json").exists() else "interrupted",
                         signal=received, termination_verified=True)
            result = 128 + received
            return result
        program = str(Path(spec["source"]["path"]) / "tide_lab/supervisor.py")
        command = ["python", "-u", program, "worker", "--record", str(record)]
        if spec["device"] == "npu":
            command = ["python", spec["queue_helper"]["path"], "--job-id", spec["job_id"],
                       "--project", "tide-learning-lab", "--priority", str(spec["priority"]),
                       "--npu-count", "1", "--max-wait-seconds", str(spec["resources"]["wait_seconds"]),
                       "--state-file", str(output / "queue-state.json"), "--", *command]
        environment = os.environ.copy()
        environment.update(spec["environment"])
        # Each command is an argv element, never shell interpolation.
        shell = ["/bin/bash", "-c", 'set -e; source "$1"; shift; exec "$@"',
                 "tide-job-env", spec["environment_script"]["path"], *command]
        child = subprocess.Popen(shell, cwd=spec["source"]["path"], env=environment, start_new_session=True)
        state.update(child_process_group=child.pid, launcher=process_identity(child.pid))
        event(output, "launched", pid=child.pid)
        last_write = 0.0
        while child.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= spec["resources"]["timeout_seconds"] and received is None:
                timed_out = True
                request_stop(signal.SIGTERM)
            if received and time.monotonic() - signal_time >= 20:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            queue = read_json(output / "queue-state.json", {})
            current_worker = read_json(output / "worker-state.json", {})
            phase = "stopping" if received else ("running" if current_worker else queue.get("status", "starting"))
            if phase != state["state"] or elapsed - last_write >= 10:
                known = { (item["pid"], item["start_ticks"]): item for item in state["descendants"] }
                known.update({(item["pid"], item["start_ticks"]): item for item in group_processes(child.pid)})
                state.update(state=phase, updated_at=now(), elapsed_seconds=elapsed,
                             descendants=list(known.values()))
                atomic_json(output / "state.json", state)
                last_write = elapsed
            time.sleep(.2)
        launcher_code = child.wait()
        remaining = group_processes(child.pid)
        leaked = bool(remaining)
        if remaining:
            os.killpg(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while group_processes(child.pid) and time.monotonic() < deadline:
                time.sleep(.1)
            if group_processes(child.pid):
                os.killpg(child.pid, signal.SIGKILL)
                time.sleep(.1)
        actual_worker = read_json(output / "worker-state.json", {})
        queue = read_json(output / "queue-state.json", {})
        exit_code = actual_worker.get("exit_code")
        if timed_out or queue.get("status") == "timed_out":
            terminal = "timed_out"
        elif received:
            terminal = "stopped" if (output / "stop-request.json").exists() else "interrupted"
        else:
            terminal = "completed" if launcher_code == 0 and exit_code == 0 and not leaked else "failed"
        state.update(state=terminal, launcher_exit_code=launcher_code, workload_exit_code=exit_code,
                     queue_state=queue or None, signal=received, leaked_descendants=leaked,
                     remaining_descendants=group_processes(child.pid),
                     termination_verified=not group_processes(child.pid))
        result = 0 if terminal == "completed" else (124 if terminal == "timed_out" else normalized_code(launcher_code) or 1)
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}")
        print(state["error"], file=sys.stderr, flush=True)
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                state["termination_error"] = "launcher did not exit after SIGKILL"
        state["termination_verified"] = not child or not group_processes(child.pid)
        result = 1
    finally:
        state.update(ended_at=now(), elapsed_seconds=time.monotonic() - started)
        atomic_json(output / "state.json", state)
        event(output, "supervisor_terminal", state=state["state"], exit_code=result)
    return result


def finalize(record):
    """systemd calls this after termination, including a killed supervisor."""
    spec = read_json(record)
    output = Path(record).parent
    state = read_json(output / "state.json", {"schema": "tide-job-lifecycle-v1", "job_id": spec["job_id"]})
    service = {key: os.environ.get(key) for key in ("SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS")}
    result = service["SERVICE_RESULT"]
    if result == "timeout":
        state["state"] = "timed_out"
    elif (output / "stop-request.json").exists() and state.get("state") != "completed":
        state["state"] = "stopped"
    elif state.get("state") not in TERMINAL or (result not in (None, "success") and state.get("state") == "completed"):
        state["state"] = "failed"
    group = state.get("control_group") or current_cgroup()
    remaining = [item for item in cgroup_processes(group) if item["pid"] != os.getpid()]
    known_alive = [item for item in state.get("descendants", []) if identity_alive(item)]
    state.update(systemd=service, finalized_at=now(), ended_at=state.get("ended_at", now()),
                 remaining_descendants=remaining, known_descendants_alive=known_alive,
                 termination_verified=not remaining and not known_alive)
    if remaining or known_alive:
        state["state"] = "failed"
        state["termination_error"] = "live descendants remain after service termination"
    atomic_json(output / "state.json", state)
    event(output, "systemd_finalized", state=state["state"], termination_verified=state["termination_verified"])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "finalize"))
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    return {"run": run, "worker": worker, "finalize": finalize}[args.action](args.record.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
