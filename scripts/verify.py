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
from build_identity import source_hash
from durable_records import write_json

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

build_manifest = json.loads((build / "build-manifest.json").read_text())
if build_manifest["cpp_source_sha256"] != source_hash(root):
    parser.error("native build does not match current C++ sources; run scripts/build.py")
for name, digest in build_manifest["binary_sha256"].items():
    if hashlib.sha256((build / name).read_bytes()).hexdigest() != digest:
        parser.error(f"native build artifact changed: {name}")
if Path(_tide_native.__file__).parent.resolve() != build:
    parser.error("loaded native module is outside the requested build directory")

out = Path(args.output_dir).resolve()
out.mkdir(parents=True, exist_ok=False)
binary = Path(_tide_native.__file__)
command = [sys.executable, "-m", "pytest", "tests", "-q", "--dtype", args.dtype,
           "--basetemp", str(out / "test-tmp")]
manifest = {
    "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
    "command": command, "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "environment": {"architecture": platform.machine(), "python": platform.python_version(),
                    "torch": torch.__version__, "torch_git": torch.version.git_version,
                    "cxx11_abi": torch.compiled_with_cxx11_abi(), "threads": 1},
    "device": "cpu", "resolution_reason": "explicit:cpu" if args.device == "cpu" else "auto:only-compiled-cpu",
    "dtype": args.dtype, "seed": args.seed, "build": build_manifest,
    "native_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "state": "running",
}
def save():
    write_json(out / "result.json", manifest)
save()
env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root / "python"), str(build))),
           TIDE_TEST_SEED=str(args.seed), TIDE_BUILD_DIR=str(build))
with (out / "tests.log").open("w") as log:
    result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
manifest.update(exit_code=result.returncode, state="passed" if result.returncode == 0 else "failed",
                finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()
print((out / "tests.log").read_text())
print(f"Evidence: {out / 'result.json'}")
sys.exit(result.returncode)
