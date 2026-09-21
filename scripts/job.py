#!/usr/bin/env python3
"""Persistent lifecycle for a command run by an external durable launcher."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    parser.error("command required after --")
out = Path(args.output_dir).resolve()
out.mkdir(parents=True, exist_ok=False)
record = {"command": command, "cwd": str(Path.cwd()), "state": "running",
          "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(),
          "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
    temp = out / "status.tmp"
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(out / "status.json")
save()
try:
    with (out / "task.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    record.update(state="passed" if result.returncode == 0 else "failed", exit_code=result.returncode)
except BaseException as error:
    record.update(state="failed", error=repr(error), exit_code=1)
finally:
    record["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
sys.exit(record["exit_code"])
