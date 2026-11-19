"""Wait inside a bounded service before competing with specified user units.

This is resource sequencing, not success propagation. Terminal failed units also
release resources. No service is stopped or modified, and device admission runs
only after this wrapper replaces itself with the requested command.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time

from job import atomic


def inspect(unit):
    result = subprocess.run(["systemctl", "--user", "show", unit, "--no-pager",
                             "--property=LoadState,ActiveState,SubState,MainPID,Result"],
                            text=True, capture_output=True, timeout=10)
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode and properties.get("LoadState") != "not-found":
        raise RuntimeError("cannot inspect "+unit+": "+result.stderr.strip())
    return properties


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", action="append", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not command:
        parser.error("a finite positive bound and a command are required")
    started = time.monotonic()
    record = dict(state="waiting", started=datetime.now(timezone.utc).isoformat(),
                  units=args.unit, command=command, boundary="resource sequencing only; prior success not required")
    previous = None
    while True:
        observations = {unit: inspect(unit) for unit in args.unit}
        busy = [unit for unit, value in observations.items()
                if value.get("ActiveState") in {"active", "activating", "deactivating", "reloading"}
                or int(value.get("MainPID", 0)) != 0]
        record.update(observations=observations, busy=busy, elapsed_seconds=time.monotonic()-started)
        if not busy:
            record.update(state="released", finished=datetime.now(timezone.utc).isoformat())
            atomic(args.status, record)
            print(json.dumps(record), flush=True)
            os.execvpe(command[0], command, os.environ)
        if record["elapsed_seconds"] >= args.seconds:
            record["state"] = "timed_out"
            atomic(args.status, record)
            return 124
        if observations != previous:
            atomic(args.status, record)
            previous = observations
        time.sleep(min(5, max(0.01, args.seconds-record["elapsed_seconds"])))


if __name__ == "__main__":
    raise SystemExit(main())
