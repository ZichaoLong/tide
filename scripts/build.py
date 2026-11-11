#!/usr/bin/env python3
"""Build against the exact Torch imported by this Python; no site paths."""
import argparse
from build_resources import add_build_jobs
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
add_build_jobs(parser)
parser.add_argument("--backend", choices=("cpu", "cuda", "npu"), default="cpu")
parser.add_argument("--npu-runtime", choices=("python", "standalone"), default="python")
parser.add_argument('--libtorch-prefix', type=Path, help='standalone NPU SDK: use this LibTorch instead of Python Torch')
parser.add_argument('--cxx11-abi', type=int, choices=(0, 1), help='declared SDK ABI; required with --libtorch-prefix')
args = parser.parse_args()
if args.jobs < 1:
    parser.error("--jobs must be positive")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
root = Path(__file__).resolve().parents[1]
target = Path(args.build_dir).resolve()
if args.libtorch_prefix:
    if args.backend != 'npu' or args.npu_runtime != 'standalone' or args.cxx11_abi is None:
        parser.error('--libtorch-prefix requires --backend npu --npu-runtime standalone --cxx11-abi 0|1')
    from libtorch_distribution import standalone
    distribution = standalone(args.libtorch_prefix, args.cxx11_abi)
    extra = [f'-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI={args.cxx11_abi}']
else:
    if args.cxx11_abi is not None:
        parser.error('--cxx11-abi requires --libtorch-prefix; Python Torch declares its own ABI')
    import torch
    distribution = dict(torch=str(torch.__version__), torch_dir=str(Path(torch.__file__).parent / 'share/cmake/Torch'),
                        prefix=torch.utils.cmake_prefix_path, cuda=torch.version.cuda,
                        cxx11_abi=torch.compiled_with_cxx11_abi(), discovery='python-torch')
    extra = [f'-DPython3_EXECUTABLE={sys.executable}']
check_reuse(target, args.backend, args.npu_runtime, torch_dir=distribution['torch_dir'], torch_version=distribution['torch'])
before = source_hash(root)
if args.backend == "cuda" and not distribution['cuda']:
    parser.error("CUDA build requires a CUDA-enabled Torch distribution")
bindings = not (args.backend == "npu" and args.npu_runtime == "standalone")
clients = not (args.backend == "npu" and args.npu_runtime == "python")
vendor = None
if args.backend == "npu" and bindings:
    import torch_npu
    vendor = torch_npu.__version__
prefixes = [distribution['prefix'], *os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)]
subprocess.run(["cmake", "-S", str(root), "-B", str(target), "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={';'.join(p for p in prefixes if p)}",
                f"-DTorch_DIR={distribution['torch_dir']}",
                f"-DTIDE_BACKEND={args.backend.upper()}", f"-DTIDE_NPU_RUNTIME={args.npu_runtime}",
                f"-DTIDE_PYTHON_BINDINGS={'ON' if bindings else 'OFF'}",
                f"-DTIDE_BUILD_CLIENTS={'ON' if clients else 'OFF'}",
                *extra], check=True)
subprocess.run(["cmake", "--build", str(target), "--parallel", str(args.jobs)], check=True)
if source_hash(root) != before:
    raise RuntimeError("C++ source changed during build; freeze the source and rebuild")
manifest = {"source": revision(root), "cpp_source_sha256": before, "torch": distribution['torch'],
            "distribution": distribution,
            "environment": environment(target), "build_jobs": args.jobs,
            "backend": args.backend, "npu_runtime": args.npu_runtime if args.backend == "npu" else None,
            "torch_npu": vendor, "cuda": distribution['cuda'],
            "architecture": platform.machine(), "python": platform.python_version(),
            "cxx11_abi": distribution['cxx11_abi'],
            "binary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (target / "libtidegraph.a", target / "libtidegraph_runtime.a", target / "_tide_native.so", target / "tidegraph-smoke", target / "tidegraph-kernel-check",
                                        target / "tidegraph-full-check", target / "tidegraph-aggregate-check",
                                        target / "tidegraph-optimizer-check", target / "tidegraph-checkpoint-check", target / "tidegraph-weighted-training-check",
                                        target / "tidegraph-lh-scope-check", target / "tidegraph-streaming-bench",
                                        target / "tidegraph-scale-bench", target / "tidegraph-profile-check",
                                        target / "tidegraph-settle-check", target / "tidegraph-greedy-check",
                                        target / "tidegraph-accelerator-check", target / "tidegraph-runtime-check",
                                        target / "tidegraph-read-precision-check", target / "tidegraph-placement-check",
                                        target / "tidegraph-payload-ownership-check",
                                        target / "tidegraph-weighted-binding-module-check", target / "tidegraph-column-check",
                                        target / "libtidegraph-weighted-binding-module.so") if p.is_file()}}
write_json(target / "build-manifest.json", manifest)
