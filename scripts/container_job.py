#!/usr/bin/env python3
"""Bounded detached jobs for Linux containers without a user service manager.

Linux subreaping retains descendants even when tools create their own groups.
This is not a reboot-recovery service or a replacement for systemd on hosts.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from durable_records import write_json
from source_identity import source_state


TERMINAL = {"passed", "failed", "cancelled"}


def now():
    return datetime.now(timezone.utc).isoformat()


def process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]), start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
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
        more = {pid for pid, info in processes.items() if info["ppid"] in owned} - owned
        if not more:
            return {pid: processes[pid] for pid in owned - {parent}
                    if processes[pid]["state"] not in {"Z", "X"}}
        owned.update(more)


def signal_workload(sig):
    # Signal only owned PIDs, including leaders/members of Ninja-created groups.
    # Never signal a shared external group to which a descendant might attach.
    for pid, observed in descendants(os.getpid()).items():
        current = process(pid)
        if current and current["start_ticks"] == observed["start_ticks"]:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def drain(child, grace):
    """Return only after termination and checking all non-zombie group members."""
    signal_workload(signal.SIGTERM)
    deadline = time.monotonic() + grace
    while descendants(os.getpid()) and time.monotonic() < deadline:
        child.poll()
        time.sleep(.05)
    deadline = time.monotonic() + 10
    while descendants(os.getpid()) and time.monotonic() < deadline:
        signal_workload(signal.SIGKILL)
        time.sleep(.05)
    child.wait(timeout=10)
    # Reap orphaned grandchildren; Popen retains its own child's real exit code.
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    return sorted(descendants(os.getpid()))


def supervise(out):
    record = json.loads((out / "status.json").read_text())
    cancelled = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, frame: cancelled.append(number))
    record.update(supervisor_pid=os.getpid(),
                  supervisor_start_ticks=process(os.getpid())["start_ticks"])
    child = None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "cannot become a child subreaper")
        os.nice(10)
        with (out / "task.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(record["command"], cwd=record["cwd"],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True)
            record.update(state="running", workload_pid=child.pid, pgid=child.pid,
                          workload_start_ticks=process(child.pid)["start_ticks"])
            write_json(out / "status.json", record)
            deadline = time.monotonic() + record["timeout_seconds"]
            while child.poll() is None and not cancelled and time.monotonic() < deadline:
                time.sleep(.1)
            code = child.poll()
            reason = ("signal" if cancelled else "timeout" if code is None else
                      "descendants_after_exit" if descendants(os.getpid()) else "exit")
            remaining = drain(child, record["grace_seconds"])
            code = child.returncode
            exit_code = (128 + cancelled[0] if cancelled else 124 if reason == "timeout"
                         else 1 if remaining or reason == "descendants_after_exit" else code)
            record.update(state="cancelled" if cancelled else "passed" if exit_code == 0 else "failed",
                          exit_code=exit_code, workload_exit_code=code,
                          termination_reason=reason, remaining_live_pids=remaining)
    except BaseException as error:
        remaining = drain(child, record["grace_seconds"]) if child is not None else []
        record.update(state="failed", exit_code=1, error=repr(error), remaining_live_pids=remaining)
    finally:
        record["finished"] = now()
        write_json(out / "status.json", record)
    return record["exit_code"]


def start(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("command required after --")
    if not (0 < args.timeout < float("inf") and 0 < args.grace < float("inf")):
        raise ValueError("timeout and grace must be positive and finite")
    cwd = Path(args.cwd).resolve()
    source, dirty = source_state(cwd)
    if dirty:
        raise ValueError("freeze a clean source checkout before submission")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(state="starting", source=source, dirty=dirty, started=now(),
                  cwd=str(cwd), command=command, timeout_seconds=args.timeout,
                  grace_seconds=args.grace, launcher=str(Path(__file__).resolve()),
                  scope="Linux subreaper and all descendant process groups; no reboot recovery")
    write_json(out / "status.json", record)
    with (out / "supervisor.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                  "_supervise", "--output-dir", str(out)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = json.loads((out / "status.json").read_text())
        if record["state"] != "starting":
            print(json.dumps(record))
            return 0 if record["state"] in {"running", "passed"} else 1
        if child.poll() is not None:
            raise RuntimeError("supervisor exited before publishing running state; inspect supervisor.log")
        time.sleep(.05)
    raise RuntimeError("supervisor startup unconfirmed; inspect status before retrying")


def inspect(out):
    record = json.loads((out / "status.json").read_text())
    current = process(record.get("supervisor_pid", -1))
    record["observed_supervisor_alive"] = bool(current and current["state"] not in {"Z", "X"}
        and current["start_ticks"] == record.get("supervisor_start_ticks"))
    record["observed_live_workload_pids"] = sorted(descendants(record["supervisor_pid"])) if record["observed_supervisor_alive"] else []
    return record


def stop(out):
    record = inspect(out)
    if record["state"] not in TERMINAL:
        if not record["observed_supervisor_alive"]:
            raise RuntimeError("supervisor identity missing; inspect workload before manual recovery")
        os.kill(record["supervisor_pid"], signal.SIGTERM)
        deadline = time.monotonic() + record["grace_seconds"] + 15
        while time.monotonic() < deadline:
            record = inspect(out)
            if record["state"] in TERMINAL:
                break
            time.sleep(.1)
    print(json.dumps(record))
    return 0 if record["state"] in TERMINAL and not record["observed_live_workload_pids"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("start")
    launch.add_argument("--cwd", required=True)
    launch.add_argument("--output-dir", required=True)
    launch.add_argument("--timeout", required=True, type=float)
    launch.add_argument("--grace", type=float, default=5)
    launch.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("status", "stop", "_supervise"):
        sub.add_parser(name).add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.action == "start":
        return start(args)
    out = Path(args.output_dir).resolve()
    if args.action == "_supervise":
        return supervise(out)
    if args.action == "stop":
        return stop(out)
    record = inspect(out)
    print(json.dumps(record))
    return 0 if record["state"] in TERMINAL or record["observed_supervisor_alive"] else 1


if __name__ == "__main__":
    sys.exit(main())
