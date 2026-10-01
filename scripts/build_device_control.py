#!/usr/bin/env python3
"""Build the optional device backend against a matching NPU core/runtime owner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from build_environment import cache_values
from build_identity import source_hash
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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    core, build = args.core_build.resolve(), args.build_dir.resolve()
    manifest = json.loads((core / "build-manifest.json").read_text())
    runtime = manifest.get("npu_runtime")
    if manifest.get("backend") != "npu" or runtime not in {"standalone", "python"}:
        parser.error("control component requires a matching NPU core")
    if runtime == "python" and not args.ascendc_soc:
        parser.error("Python resident backend requires --ascendc-soc")
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
    subprocess.run(["cmake", "-S", str(root / "tools/device_online"), "-B", str(build),
                    # CANN9 legacy host-stub extraction uses literal object
                    # paths; Ninja emits /./ paths inconsistent with its JSON.
                    "-G", "Unix Makefiles" if args.ascendc_soc else "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", "-DTideGraph_DIR="+str(package),
                    "-DTorch_DIR="+cache_values(core)["Torch_DIR"],
                    "-DTIDE_DEVICE_RUNTIME="+runtime,
                    *(["-DPython3_EXECUTABLE="+sys.executable] if runtime == "python" else []),
                    "-DCMAKE_PREFIX_PATH="+";".join(p for p in prefixes if p),
                    *(["-DTIDE_DEVICE_ASCENDC=ON", "-DSOC_VERSION="+args.ascendc_soc]
                      if args.ascendc_soc else [])], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(args.jobs)], check=True)
    if runtime == "standalone":
        subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error"],
                       check=True, timeout=120)
    binaries, loaders = {}, {}
    names = ["tide-device-control-check", "tide-device-failure-check", "tide-device-numerical-check", "tide-packed-queue-check"]
    if args.ascendc_soc:
        names.extend(("tide-device-closure-check", "tide-device-queue-check",
                      "tide-device-broadcast-check", "tide-device-ready-check", "tide-device-selector-check",
                      "tide-content-flow-check", "tide-content-window-check", "tide-packed-full-check", "tide-packed-sum-check",
                      "tide-device-add-check", "tide-device-state-vjp-check", "tide-device-full-vjp-check", "tide-device-extra-full-vjp-check", "tide-device-reverse-links-check", "tide-device-graph-vjp-check", "tide-device-parameter-vjp-check", "tide-device-optimizer-check", "tide-device-training-step-check", "tide-device-retained-check", "tide-device-clock-check", "tide-device-norm-check", "tide-device-lh-full-check",
                      "tide-device-origin-check", "tide-device-emission-check", "tide-device-swiglu-check", "tide-device-fiber-check",
                      "tide-device-fiber-pool-check", "tide-device-event-attention-check", "tide-device-attention-tile-check", "tide-device-memory-check", "tide-device-event-batch-check", "tide-device-fiber-batch-check", "tide-device-aggregate-check"))
        names.append("libtide-resident.so")
        names.append("tide-resident-check")
        names.append("tide-resident-training-check")
        names.append("tide-resident-full-training-check")
        names.extend(("tide-device-aggregate-vjp-check", "tide-resident-aggregate-training-check"))
        names.extend(("tide-device-control-vjp-check", "tide-resident-control-training-check"))
    if runtime == "python":
        names = ["_tide_resident.so", "libtide-resident.so"]
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
        npu_runtime=runtime,
        ascendc_soc=args.ascendc_soc,
        binary_sha256=binaries, loader_sha256=loaders,
        scope="build/loader/help only; device control check required separately"))


if __name__ == "__main__":
    main()
