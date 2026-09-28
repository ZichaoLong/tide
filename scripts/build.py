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
from durable_records import write_json
from build_environment import check_reuse, environment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-dir", default="build")
parser.add_argument("--jobs", type=int, default=2)
parser.add_argument("--backend", choices=("cpu", "cuda", "npu"), default="cpu")
parser.add_argument("--npu-runtime", choices=("python", "standalone"), default="python")
args = parser.parse_args()
if args.jobs < 1:
    parser.error("--jobs must be positive")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

root = Path(__file__).resolve().parents[1]
target = Path(args.build_dir).resolve()
check_reuse(target, args.backend, args.npu_runtime, torch)
before = source_hash(root)
if args.backend == "cuda" and not torch.version.cuda:
    parser.error("CUDA build requires a CUDA-enabled Torch distribution")
bindings = not (args.backend == "npu" and args.npu_runtime == "standalone")
clients = not (args.backend == "npu" and args.npu_runtime == "python")
vendor = None
if args.backend == "npu" and bindings:
    import torch_npu
    vendor = torch_npu.__version__
prefixes = [torch.utils.cmake_prefix_path, *os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)]
subprocess.run(["cmake", "-S", str(root), "-B", str(target), "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={';'.join(p for p in prefixes if p)}",
                f"-DTorch_DIR={Path(torch.__file__).parent / 'share/cmake/Torch'}",
                f"-DTIDE_BACKEND={args.backend.upper()}", f"-DTIDE_NPU_RUNTIME={args.npu_runtime}",
                f"-DTIDE_PYTHON_BINDINGS={'ON' if bindings else 'OFF'}",
                f"-DTIDE_BUILD_CLIENTS={'ON' if clients else 'OFF'}",
                f"-DPython3_EXECUTABLE={sys.executable}"], check=True)
subprocess.run(["cmake", "--build", str(target), "--parallel", str(args.jobs)], check=True)
if source_hash(root) != before:
    raise RuntimeError("C++ source changed during build; freeze the source and rebuild")
manifest = {"source": revision(root), "cpp_source_sha256": before, "torch": torch.__version__,
            "environment": environment(target),
            "backend": args.backend, "npu_runtime": args.npu_runtime if args.backend == "npu" else None,
            "torch_npu": vendor, "cuda": torch.version.cuda,
            "architecture": platform.machine(), "python": platform.python_version(),
            "cxx11_abi": torch.compiled_with_cxx11_abi(),
            "binary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (target / "libtidegraph.a", target / "libtidegraph_runtime.a", target / "_tide_native.so", target / "tidegraph-smoke", target / "tidegraph-kernel-check",
                                        target / "tidegraph-full-check", target / "tidegraph-aggregate-check",
                                        target / "tidegraph-optimizer-check", target / "tidegraph-checkpoint-check",
                                        target / "tidegraph-lh-scope-check", target / "tidegraph-streaming-bench",
                                        target / "tidegraph-scale-bench", target / "tidegraph-profile-check",
                                        target / "tidegraph-settle-check", target / "tidegraph-accelerator-check") if p.is_file()}}
write_json(target / "build-manifest.json", manifest)
