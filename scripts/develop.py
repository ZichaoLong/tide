#!/usr/bin/env python3
"""Freeze dirty development source, build native code and run selected tests.

Run under scripts/job.py in a durable service. Keep checkout/build caches frozen
until termination. Clean qualification still uses scripts/verify.py.
"""
import argparse
import hashlib
from pathlib import Path
import os
import subprocess
import sys
from durable_records import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--jobs", type=int, default=2)
parser.add_argument("tests", nargs="+", help="pytest paths or node IDs")
args = parser.parse_args()
out = Path(args.output_dir).resolve()
if not out.is_dir() or (out/"source.tar.gz").exists() or args.jobs < 1:
    parser.error("existing new job directory and positive build jobs required")
root = Path(__file__).resolve().parents[1]


def files():
    return subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)


def identity():
    digest = hashlib.sha256()
    for name in sorted(set(files().split(b"\0")) - {b""}):
        path = root/os.fsdecode(name)
        digest.update(name+b"\0"+(path.read_bytes() if path.is_file() else b"deleted")+b"\0")
    return digest.hexdigest()


before = identity()
archive = out/"source.tar.gz"
subprocess.run(["tar", "--null", "-T", "-", "-czf", str(archive)], input=files(), cwd=root, check=True)
(out/"source.sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+"  source.tar.gz\n")
env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           PYTHONPATH=os.pathsep.join((str(root/"python"), str(root/"build"))))
commands = [[sys.executable, "scripts/build.py", "--jobs", str(args.jobs)],
            [sys.executable, "-m", "pytest", *args.tests, "-q", "--dtype", "both"]]
record = {"tree_sha256": before, "commands": commands, "stages": [], "state": "running"}
code = 1
try:
    for command in commands:
        result = subprocess.run(command, cwd=root, env=env)
        record["stages"].append({"command": command, "exit_code": result.returncode})
        code = result.returncode
        if code:
            break
    if identity() != before:
        raise RuntimeError("source changed during development run; restore frozen inputs and rerun")
except BaseException as error:
    code = 1
    record["error"] = repr(error)
    raise
finally:
    record.update(state="passed" if code == 0 and identity() == before else "failed", exit_code=code)
    write_json(out/"development.json", record)
sys.exit(code)
