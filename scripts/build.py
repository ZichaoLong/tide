#!/usr/bin/env python3
"""Build against the exact Torch imported by this Python; no site paths."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-dir", default="build")
parser.add_argument("--jobs", type=int, default=2)
args = parser.parse_args()
if args.jobs < 1:
    parser.error("--jobs must be positive")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"  # This target is CPU-only.
import torch

root = Path(__file__).resolve().parents[1]
target = Path(args.build_dir).resolve()
subprocess.run(["cmake", "-S", str(root), "-B", str(target), "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
                f"-DPython3_EXECUTABLE={sys.executable}"], check=True)
subprocess.run(["cmake", "--build", str(target), "--parallel", str(args.jobs)], check=True)
