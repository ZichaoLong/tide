#!/usr/bin/env python3
"""Build a standalone public-API benchmark against verified installed packages."""
import argparse
from build_resources import add_build_jobs
import json
import os
from pathlib import Path
import subprocess
from build_environment import cache_values
from build_identity import source_hash
from durable_records import write_json
from source_identity import digest, source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-build", type=Path, required=True)
    parser.add_argument("--resident-build", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true", help="frozen development snapshot only, not qualification")
    add_build_jobs(parser)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, dirty = source_state(root)
    if dirty and not args.allow_dirty:
        parser.error("standalone benchmark build requires clean immutable source")
    core, out = args.core_build.resolve(), args.output_dir.resolve()
    core_record = json.loads((core/"build-manifest.json").read_text())
    if core_record["cpp_source_sha256"] != source_hash(root):
        parser.error("core source differs from the benchmark source")
    builds = [(core, core_record)]
    backend = None
    if args.resident_build:
        backend = args.resident_build.resolve()
        record = json.loads((backend/"control-build.json").read_text())
        from build_device_control import component_hash
        if (record["runtime"] != "standalone" or record["core"] != core_record
                or record["component_sha256"] != component_hash(root)):
            parser.error("resident source/runtime/core mismatch")
        builds.append((backend, record))
    for build, record in builds:
        for name, expected in record["binary_sha256"].items():
            if digest(build/name) != expected:
                parser.error("changed build artifact: "+name)
    out.mkdir(parents=True, exist_ok=False)
    prefix, build = out/"installed", out/"client"
    files = list((root/"tools/weighted_benchmark").iterdir()) + [root/"cpp/bench/compare.cpp", root/"cpp/bench/streaming.h",
             root/"tools/lh_repro/vendor/nlohmann/json.hpp", root/"tools/device_online/weighted_precision_reference.cpp",
             root/"tools/device_online/weighted_precision_reference.h"]
    report = dict(schema="tide-weighted-benchmark-build-v1", state="running", source=source, dirty=dirty,
                  core=core_record, resident=builds[-1][1] if backend else None, commands=[],
                  sources={str(p.relative_to(root)): digest(p) for p in files if p.is_file()},
                  scope="standalone build/loader/help only; numerical execution separately required")

    def run(command):
        report["commands"].append(list(map(str, command))); write_json(out/"result.json", report)
        subprocess.run(report["commands"][-1], check=True)

    try:
        run(["cmake", "--install", core, "--prefix", prefix])
        if backend:
            for component in ("weighted", "weighted-training"):
                run(["cmake", "--install", backend, "--prefix", prefix, "--component", component])
        prefixes = [str(prefix), *os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)]
        run(["cmake", "-S", root/"tools/weighted_benchmark", "-B", build, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DTorch_DIR="+cache_values(core)["Torch_DIR"],
             "-DCMAKE_PREFIX_PATH="+";".join(p for p in prefixes if p),
             "-DTIDE_WEIGHTED_RESIDENT="+("ON" if backend else "OFF")])
        run(["cmake", "--build", build, "--parallel", str(args.jobs)])
        binary = build/"weighted-benchmark"
        loader = subprocess.check_output(["ldd", str(binary)], text=True)
        if any(x in loader.lower() for x in ("not found", "libtorch_python", "libpython", "/stub/", "/stubs/", "/simulator/")):
            raise RuntimeError("standalone benchmark loader closure failed")
        (out/"loader.txt").write_text(loader)
        help_text = subprocess.check_output([str(binary), "--help"], text=True)
        (out/"help.txt").write_text(help_text)
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during standalone benchmark build")
        report.update(state="passed", binary_sha256=digest(binary), loader_sha256=digest(out/"loader.txt"),
                      help_sha256=digest(out/"help.txt"))
    except BaseException as error:
        report.update(state="failed", error=repr(error)); raise
    finally:
        write_json(out/"result.json", report)


if __name__ == "__main__":
    main()
