#!/usr/bin/env python3
"""Build then verify a frozen source checkout; suitable for scripts/job.py."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--jobs", type=int, default=2)
parser.add_argument("--lh-snapshot", help="also qualify original C++ LH selector from this immutable snapshot")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(root / "scripts/build.py"), "--jobs", str(args.jobs)], check=True)
subprocess.run([sys.executable, str(root / "scripts/verify.py"), "--device", "cpu", "--dtype", "both",
                "--output-dir", str(Path(args.output_dir).resolve() / "verification")], check=True)
if args.lh_snapshot:
    subprocess.run([sys.executable, str(root / "scripts/check_lh_selector.py"), "--device", "cpu", "--dtype", "both",
                    "--snapshot", args.lh_snapshot, "--output-dir", str(Path(args.output_dir).resolve() / "oracle"),
                    "--jobs", str(args.jobs)], check=True)
