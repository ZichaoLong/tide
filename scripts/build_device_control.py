#!/usr/bin/env python3
"""Build the experimental control component against a matching standalone core."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
    if manifest.get("backend") != "npu" or manifest.get("npu_runtime") != "standalone":
        parser.error("control component requires a standalone NPU core/SDK")
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
                    "-DCMAKE_PREFIX_PATH="+";".join(p for p in prefixes if p),
                    *(["-DTIDE_DEVICE_ASCENDC=ON", "-DSOC_VERSION="+args.ascendc_soc]
                      if args.ascendc_soc else [])], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(args.jobs)], check=True)
    subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error"],
                   check=True, timeout=120)
    binaries, loaders = {}, {}
    names = ["tide-device-control-check", "tide-device-numerical-check", "tide-packed-queue-check"]
    if args.ascendc_soc:
        names.extend(("tide-device-closure-check", "tide-device-queue-check",
                      "tide-device-broadcast-check", "tide-device-ready-check", "tide-device-selector-check",
                      "tide-content-flow-check", "tide-packed-full-check"))
    for name in names:
        binary = build / name
        closure = subprocess.check_output(["ldd", str(binary)], text=True)
        if any(item in closure.lower() for item in ("not found", "libtorch_python", "libpython", "/stubs/", "/stub/", "/simulator/")):
            raise RuntimeError("standalone control loader unresolved or depends on Python/stubs")
        (build / (name + "-loader.txt")).write_text(closure)
        binaries[name] = digest(binary)
        loaders[name] = digest(build / (name + "-loader.txt"))
    if source_state(root) != before or component_hash(root) != identity:
        raise RuntimeError("control source changed during build")
    write_json(build / "control-build.json", dict(schema="tide-device-control-build-v1",
        source=before[0], dirty=before[1], component_sha256=identity, core=manifest,
        ascendc_soc=args.ascendc_soc,
        binary_sha256=binaries, loader_sha256=loaders,
        scope="build/loader/help only; device control check required separately"))


if __name__ == "__main__":
    main()
