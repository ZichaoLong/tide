#!/usr/bin/env python3
"""Freeze development inputs and check both original-LH assertion variants.

Run under scripts/job.py in a durable service; its output directory must exist.
The checkout and shared build caches must remain frozen until termination.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--lh-snapshot", required=True)
parser.add_argument("--component", required=True,
                    choices=("selector", "add", "full", "attention", "pronounce", "iocortex", "all"))
parser.add_argument("--jobs", type=int, default=2)
parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
args = parser.parse_args()
out = Path(args.output_dir).resolve()
if not out.is_dir() or (out/"source.tar.gz").exists() or args.jobs < 1:
    parser.error("existing new job directory and positive build jobs required")
root = Path(__file__).resolve().parents[1]
files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
archive = out/"source.tar.gz"
subprocess.run(["tar", "--null", "-T", "-", "-czf", str(archive)], input=files, cwd=root, check=True)
(out/"source.sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+"  source.tar.gz\n")
subprocess.run([sys.executable, str(root/"scripts/build.py"), "--jobs", str(args.jobs)], cwd=root, check=True)
for assertions, directory, cache in (("on", "oracle", "build/lh-oracle"),
                                     ("off", "oracle-release", "build/lh-oracle-release")):
    subprocess.run([sys.executable, str(root/"scripts/check_lh_selector.py"), "--device", "cpu", "--dtype", args.dtype,
                    "--snapshot", args.lh_snapshot, "--component", args.component, "--jobs", str(args.jobs),
                    "--output-dir", str(out/directory), "--runtime-assertions", assertions,
                    "--oracle-build-dir", cache], cwd=root, check=True)
if args.component in ("iocortex", "all"):
    subprocess.run([sys.executable, str(root/"scripts/check_lh_iocortex_python.py"), "--device", "cpu",
                    "--oracle-result", str(out/"oracle/result.json"), "--output-dir", str(out/"python")], cwd=root, check=True)
