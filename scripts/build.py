#!/usr/bin/env python3
"""Build against the exact Torch imported by this Python; no site paths."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import json
import platform
from build_identity import revision, source_hash

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
before = source_hash(root)
subprocess.run(["cmake", "-S", str(root), "-B", str(target), "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
                f"-DPython3_EXECUTABLE={sys.executable}"], check=True)
subprocess.run(["cmake", "--build", str(target), "--parallel", str(args.jobs)], check=True)
if source_hash(root) != before:
    raise RuntimeError("C++ source changed during build; freeze the source and rebuild")
manifest = {"source": revision(root), "cpp_source_sha256": before, "torch": torch.__version__,
            "architecture": platform.machine(), "python": platform.python_version(),
            "cxx11_abi": torch.compiled_with_cxx11_abi(),
            "binary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (target / "libtidegraph.a", target / "_tide_native.so", target / "tidegraph-smoke", target / "tidegraph-kernel-check",
                                        target / "tidegraph-full-check", target / "tidegraph-aggregate-check")}}
(target / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
