#!/usr/bin/env python3
"""Checkout adapter to the installed Linux descendant-safe durable supervisor."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"python"))
from tidegraph.experiment.container_job import prepare, detach, inspect, stop as stop_job, supervise
from tidegraph.experiment.records import read_json, write_json
from source_identity import source_state


def start(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    cwd = Path(args.cwd).resolve()
    source, dirty = source_state(cwd)
    if dirty:
        raise ValueError("freeze a clean source checkout before submission")
    out = prepare(args.output_dir, command=command, cwd=cwd, timeout=args.timeout, grace=args.grace)
    record = read_json(out/"status.json")
    record.update(source=source, dirty=dirty, launcher=str(Path(__file__).resolve()))
    write_json(out/"status.json", record)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]/"python")
    record = detach(out, environment=environment)
    print(json.dumps(record))
    return 0 if record["state"] in {"running", "passed"} else 1


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
    if args.action == "_supervise":
        return supervise(args.output_dir)
    record = stop_job(args.output_dir) if args.action == "stop" else inspect(args.output_dir)
    print(json.dumps(record))
    return 0 if record["state"] in {"passed", "failed", "cancelled"} or record["observed_supervisor_alive"] else 1


if __name__ == "__main__":
    sys.exit(main())
