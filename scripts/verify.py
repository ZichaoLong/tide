#!/usr/bin/env python3
"""Run CPU qualification, retaining source identity and terminal evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", required=True, choices=("auto", "cpu", "cuda", "npu"))
parser.add_argument("--device-index", type=int)
parser.add_argument("--dtype", default="both", choices=("float32", "float64", "both"))
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--build-dir", default="build")
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
if args.device not in {"cpu", "auto"}:
    parser.error(f"{args.device} is unsupported by this CPU-only qualification target")
if args.device_index is not None and (args.device != "cpu" or args.device_index != 0):
    parser.error("only explicit cpu --device-index 0 is supported")
if args.seed < 0:
    parser.error("seed must be nonnegative")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch

root = Path(__file__).resolve().parents[1]
build = Path(args.build_dir).resolve()
sys.path[:0] = [str(root / "python"), str(build)]
import _tide_native

out = Path(args.output_dir).resolve()
out.mkdir(parents=True, exist_ok=False)
binary = Path(_tide_native.__file__)
command = [sys.executable, "-m", "pytest", "tests", "-q", "--dtype", args.dtype]
manifest = {
    "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
    "command": command, "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "environment": {"architecture": platform.machine(), "python": platform.python_version(),
                    "torch": torch.__version__, "torch_git": torch.version.git_version,
                    "cxx11_abi": torch.compiled_with_cxx11_abi(), "threads": 1},
    "device": "cpu", "resolution_reason": "explicit:cpu" if args.device == "cpu" else "auto:only-compiled-cpu",
    "dtype": args.dtype, "seed": args.seed,
    "native_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "state": "running",
}
def save():
    temp = out / "result.tmp"
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(out / "result.json")
save()
env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root / "python"), str(build))), TIDE_TEST_SEED=str(args.seed))
with (out / "tests.log").open("w") as log:
    result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
manifest.update(exit_code=result.returncode, state="passed" if result.returncode == 0 else "failed",
                finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()
print((out / "tests.log").read_text())
print(f"Evidence: {out / 'result.json'}")
sys.exit(result.returncode)
