#!/usr/bin/env python3
"""Build the optional resident backend against a matching core/runtime owner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from build_environment import cache_values
from build_identity import source_hash
from device_component_checks import CHECKS
from durable_records import write_json
from source_identity import source_state, digest


def component_hash(root):
    result = hashlib.sha256()
    paths = [root / "scripts/build_device_control.py", *sorted((root / "tools/device_online").rglob("*"))]
    for path in paths:
        if path.is_file():
            result.update(str(path.relative_to(root)).encode()+b"\0"+path.read_bytes()+b"\0")
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-build", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--ascendc-soc", help="Explicit target SoC; enable Ascend C closure")
    parser.add_argument("--runtime", choices=("standalone", "python"), help="CUDA runtime owner; NPU must match its core")
    parser.add_argument("--cuda-architectures", help="Explicit CMake CUDA architecture list, CC80 or newer")
    parser.add_argument("--checks", nargs="+", choices=tuple(CHECKS),
                        help="Build only named standalone components; omitted builds the complete backend")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    core, build = args.core_build.resolve(), args.build_dir.resolve()
    manifest = json.loads((core / "build-manifest.json").read_text())
    backend = manifest.get("backend")
    runtime = args.runtime or (manifest.get("npu_runtime") if backend == "npu" else "standalone")
    if backend not in {"cuda", "npu"} or runtime not in {"standalone", "python"}:
        parser.error("resident component requires a matching CUDA/NPU core")
    if backend == "npu" and (runtime != manifest.get("npu_runtime") or args.cuda_architectures):
        parser.error("NPU runtime must match its core and cannot request CUDA architectures")
    if backend == "cuda" and (args.ascendc_soc or not args.cuda_architectures):
        parser.error("CUDA requires --cuda-architectures and no Ascend SoC")
    if backend == "npu" and runtime == "python" and not args.ascendc_soc:
        parser.error("Python resident backend requires --ascendc-soc")
    if args.checks and runtime != "standalone":
        parser.error("component subsets require a standalone runtime owner")
    if backend == "npu" and args.checks and not args.ascendc_soc and any(check not in {"peer", "control", "failure", "numerical", "queue"} for check in args.checks):
        parser.error("selected component requires --ascendc-soc")
    if manifest["cpp_source_sha256"] != source_hash(root):
        parser.error("core source differs; rebuild the native core")
    for name, expected in manifest["binary_sha256"].items():
        if digest(core / name) != expected:
            parser.error("core binary changed: " + name)
    before, identity = source_state(root), component_hash(root)
    build.mkdir(parents=True, exist_ok=False)
    prefix = build / "core-prefix"
    subprocess.run(["cmake", "--install", str(core), "--prefix", str(prefix)], check=True)
    package, = prefix.glob("lib*/cmake/TideGraph")
    prefixes = [str(prefix), *os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)]
    targets = [CHECKS[check][0] for check in dict.fromkeys(args.checks or [])]
    subprocess.run(["cmake", "-S", str(root / "tools/device_online"), "-B", str(build),
                    # CANN9 legacy host-stub extraction uses literal object
                    # paths; Ninja emits /./ paths inconsistent with its JSON.
                    "-G", "Unix Makefiles" if args.ascendc_soc else "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", "-DTideGraph_DIR="+str(package),
                    "-DTorch_DIR="+cache_values(core)["Torch_DIR"],
                    "-DTIDE_DEVICE_RUNTIME="+runtime,
                    "-DTIDE_DEVICE_BACKEND="+backend.upper(),
                    *(["-DCMAKE_CUDA_ARCHITECTURES="+args.cuda_architectures] if backend == "cuda" else []),
                    *(["-DTIDE_DEVICE_CHECK_TARGETS="+";".join(targets)] if targets else []),
                    *(["-DPython3_EXECUTABLE="+sys.executable] if runtime == "python" else []),
                    "-DCMAKE_PREFIX_PATH="+";".join(p for p in prefixes if p),
                    *(["-DTIDE_DEVICE_ASCENDC=ON", "-DSOC_VERSION="+args.ascendc_soc]
                      if args.ascendc_soc else [])], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(args.jobs),
                    *(["--target", "tide-device-selected-checks"] if targets else [])], check=True)
    if runtime == "standalone":
        subset_tests = {"peer": "device-peer-help", "control": "device-control-help", "numerical": "device-numerical-help",
                        "queue": "packed-queue-cpu-fp32|packed-queue-cpu-fp64",
                        "full-training": "resident-control-comparison"}
        tests = [subset_tests[check] for check in args.checks or [] if check in subset_tests]
        if not targets or tests:
            subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error",
                            *(["-R", "^(" + "|".join(tests) + ")$"] if targets else [])],
                           check=True, timeout=120)
    binaries, loaders = {}, {}
    names = [CHECKS[c][0] for c in ("peer", "sequence", "control", "failure", "numerical", "queue")]
    if args.ascendc_soc or backend == "cuda":
        names = list(dict.fromkeys(name for name, _ in CHECKS.values())) + ["libtide-resident.so"]
    if runtime == "python":
        names = ["_tide_resident.so", "libtide-resident.so"]
    if targets:
        names = targets
    for name in names:
        binary = build / name
        closure = subprocess.check_output(["ldd", str(binary)], text=True)
        forbidden = ("not found", "/stubs/", "/stub/", "/simulator/")
        if runtime == "standalone":
            forbidden += ("libtorch_python", "libpython")
        else:
            # NPU registration is provided by the wheel already loaded by the
            # client. Neither optional library may carry a second NPU owner.
            forbidden += ("libtorch_npu",)
        if any(item in closure.lower() for item in forbidden):
            raise RuntimeError("device loader unresolved or contains a conflicting runtime/stub")
        (build / (name + "-loader.txt")).write_text(closure)
        binaries[name] = digest(binary)
        loaders[name] = digest(build / (name + "-loader.txt"))
    if source_state(root) != before or component_hash(root) != identity:
        raise RuntimeError("control source changed during build")
    write_json(build / "control-build.json", dict(schema="tide-device-control-build-v1",
        source=before[0], dirty=before[1], component_sha256=identity, core=manifest,
        backend=backend, runtime=runtime, npu_runtime=runtime if backend == "npu" else None,
        cuda_architectures=args.cuda_architectures,
        ascendc_soc=args.ascendc_soc,
        binary_sha256=binaries, loader_sha256=loaders,
        requested_checks=args.checks,
        scope="build/loader/help only; device control check required separately"))


if __name__ == "__main__":
    main()
