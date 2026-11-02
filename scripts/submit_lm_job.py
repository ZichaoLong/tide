#!/usr/bin/env python3
"""Submit an immutable-source command using the existing subreaper in background.slice."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tidegraph.experiment.container_job import prepare
from tidegraph.experiment.records import read_json, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--unit", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seconds", type=int, required=True)
parser.add_argument("--memory-gib", type=int, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
if not re.fullmatch(r"gef-lm-[a-z0-9-]+", args.unit):
    parser.error("unit must start with gef-lm- and contain lowercase letters/digits/hyphens")
if args.seconds <= 0 or not 0 < args.memory_gib <= 384:
    parser.error("positive timeout and memory limit <=384 GiB required")
source = Path(__file__).resolve().parents[1]
dirty = subprocess.check_output(["git","status","--porcelain"],cwd=source,text=True).strip()
if dirty:
    parser.error("qualification source must be a clean fixed checkout")
commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
out = prepare(args.output,command=command,cwd=source,timeout=args.seconds,grace=10,
              metadata=dict(unit=args.unit+".service", source=commit, source_directory=str(source)))
record = read_json(out/"status.json")
record.update(source=commit,dirty="")
write_json(out/"status.json",record)
launch = ["systemd-run","--user","--unit="+args.unit,"--service-type=exec","--slice=background.slice",
          "--working-directory="+str(source),"--property=KillMode=control-group","--property=Nice=10",
          f"--property=MemoryMax={args.memory_gib}G",f"--property=RuntimeMaxSec={args.seconds+60}",
          "--property=TimeoutStopSec=30","--property=StandardOutput=append:"+str(out/"supervisor.log"),
          "--property=StandardError=append:"+str(out/"supervisor.log"),
          "--setenv=PYTHONDONTWRITEBYTECODE=1","--setenv=PYTHONPATH="+str(source/"python"),
          sys.executable,"-B","-m","tidegraph.experiment.container_job","_supervise",str(out)]
result = subprocess.run(launch,capture_output=True,text=True)
if result.returncode:
    record.update(state="failed",exit_code=result.returncode,error=result.stderr)
    write_json(out/"status.json",record)
    raise SystemExit(result.returncode)
print(json.dumps(dict(unit=args.unit+".service",source=commit,output=str(out))))
