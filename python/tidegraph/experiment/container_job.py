"""Bounded Linux subreaper jobs; no service manager or SSH connection required.

This is the installed implementation of the project's container job mechanism.
It drains all owned descendants, including children that create new sessions.
It makes no promise across host reboot or a killed supervisor.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .records import now, read_json, write_json


TERMINAL = {"passed", "failed", "cancelled"}


def process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]), start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def descendants(parent):
    processes = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            info = process(int(path.name))
            if info:
                processes[int(path.name)] = info
    owned = {parent}
    while True:
        more = {pid for pid, info in processes.items() if info["ppid"] in owned}-owned
        if not more:
            return {pid: processes[pid] for pid in owned-{parent} if processes[pid]["state"] not in {"Z", "X"}}
        owned.update(more)


def signal_workload(sig):
    # Signalling only descendants also avoids touching an unrelated process
    # group that an owned child might have joined. Start ticks guard PID reuse.
    for pid, observed in descendants(os.getpid()).items():
        current = process(pid)
        if current and current["start_ticks"] == observed["start_ticks"]:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def drain(child, grace):
    signal_workload(signal.SIGTERM)
    deadline = time.monotonic()+grace
    while descendants(os.getpid()) and time.monotonic() < deadline:
        child.poll()
        time.sleep(.05)
    deadline = time.monotonic()+10
    while descendants(os.getpid()) and time.monotonic() < deadline:
        signal_workload(signal.SIGKILL)
        time.sleep(.05)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass  # Preserve live PID evidence; never claim an unkillable child stopped.
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    return sorted(descendants(os.getpid()))


def supervise(out):
    out = Path(out)
    record = read_json(out/"status.json")
    cancelled = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, frame: cancelled.append(number))
    record.update(supervisor_pid=os.getpid(), supervisor_start_ticks=process(os.getpid())["start_ticks"])
    child = None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "cannot become a child subreaper")
        current_nice = os.getpriority(os.PRIO_PROCESS, 0)
        if current_nice < 10:
            os.nice(10-current_nice)
        with (out/"task.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(record["command"], cwd=record["cwd"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            info = process(child.pid)
            record.update(state="running", workload_pid=child.pid, pgid=child.pid,
                          workload_start_ticks=info["start_ticks"] if info else None)
            write_json(out/"status.json", record)
            deadline = time.monotonic()+record["timeout_seconds"]
            while child.poll() is None and not cancelled and time.monotonic() < deadline:
                time.sleep(.1)
            code = child.poll()
            reason = ("signal" if cancelled else "timeout" if code is None else
                      "descendants_after_exit" if descendants(os.getpid()) else "exit")
            remaining = drain(child, record["grace_seconds"])
            code = child.returncode
            exit_code = (128+cancelled[0] if cancelled else 124 if reason == "timeout"
                         else 1 if remaining or reason == "descendants_after_exit" else code)
            record.update(state="cancelled" if cancelled else "passed" if exit_code == 0 else "failed",
                exit_code=exit_code, workload_exit_code=code, termination_reason=reason, remaining_live_pids=remaining)
    except BaseException as error:
        remaining = drain(child, record["grace_seconds"]) if child is not None else []
        record.update(state="failed", exit_code=1, error=repr(error), remaining_live_pids=remaining)
    finally:
        record["finished"] = now()
        write_json(out/"status.json", record)
    return record["exit_code"]


def prepare(out, *, command, cwd, timeout, grace=5, metadata=None):
    if not command or not (0 < timeout < float("inf") and 0 < grace < float("inf")):
        raise ValueError("job command, finite positive timeout and grace are required")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="tide-durable-job-v1", state="starting", started=now(),
                  cwd=str(Path(cwd).resolve()), command=list(command), timeout_seconds=timeout,
                  grace_seconds=grace, metadata=metadata or {},
                  scope="Linux subreaper and all owned descendants; no reboot recovery")
    write_json(out/"status.json", record, exclusive=True)
    return out


def detach(out, *, environment=None):
    """Launch a prepared job. The caller binds/fixes its executable and inputs."""
    out = Path(out).resolve()
    with (out/"supervisor.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, "-m", "tidegraph.experiment.container_job", "_supervise", str(out)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env=environment)
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        record = inspect(out)
        if record["state"] != "starting":
            return record
        if child.poll() is not None:
            record.update(state="failed", finished=now(), exit_code=child.returncode,
                          termination_reason="supervisor_startup_failure")
            write_json(out/"status.json", record)
            raise RuntimeError("supervisor exited before running; inspect supervisor.log")
        time.sleep(.05)
    raise RuntimeError("supervisor startup unconfirmed; inspect status before retrying")


def start_command(out, *, command, cwd, timeout, grace=5, metadata=None, environment=None):
    prepare(out, command=command, cwd=cwd, timeout=timeout, grace=grace, metadata=metadata)
    return detach(out, environment=environment)


def inspect(out):
    record = read_json(Path(out)/"status.json")
    current = process(record.get("supervisor_pid", -1))
    record["observed_supervisor_alive"] = bool(current and current["state"] not in {"Z", "X"}
        and current["start_ticks"] == record.get("supervisor_start_ticks"))
    record["observed_live_workload_pids"] = sorted(descendants(record["supervisor_pid"])) if record["observed_supervisor_alive"] else []
    if not record["observed_supervisor_alive"]:
        child = process(record.get("workload_pid", -1))
        if child and child["state"] not in {"Z", "X"} and child["start_ticks"] == record.get("workload_start_ticks"):
            record["observed_live_workload_pids"] = [record["workload_pid"], *sorted(descendants(record["workload_pid"]))]
        if record["state"] not in TERMINAL or record["observed_live_workload_pids"]:
            record["observation"] = "supervisor absent; descendant cleanup and completion unconfirmed"
    return record


def stop(out):
    record = inspect(out)
    if record["state"] not in TERMINAL:
        if not record["observed_supervisor_alive"]:
            raise RuntimeError("supervisor identity missing; inspect workload before manual recovery")
        os.kill(record["supervisor_pid"], signal.SIGTERM)
        deadline = time.monotonic()+record["grace_seconds"]+15
        while time.monotonic() < deadline:
            record = inspect(out)
            if record["state"] in TERMINAL:
                break
            time.sleep(.1)
    if record["state"] not in TERMINAL or record["observed_live_workload_pids"] or record.get("remaining_live_pids"):
        raise RuntimeError("job stop did not establish a terminal state with no live descendants")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("_supervise", "status", "stop"))
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    if args.action == "_supervise":
        return supervise(args.output_dir)
    record = (stop if args.action == "stop" else inspect)(args.output_dir)
    print(json.dumps(record))
    return 0 if record["state"] in TERMINAL or record["observed_supervisor_alive"] else 1


if __name__ == "__main__":
    sys.exit(main())
