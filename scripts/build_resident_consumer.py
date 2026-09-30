#!/usr/bin/env python3
"""Install separate core/backend packages and compile a public-header-only client."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from build_environment import cache_values
from durable_records import write_json
from source_identity import digest, source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-build", type=Path, required=True)
    parser.add_argument("--resident-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    core, backend, out = (p.resolve() for p in (args.core_build, args.resident_build, args.output_dir))
    core_record = json.loads((core / "build-manifest.json").read_text())
    record = json.loads((backend / "control-build.json").read_text())
    if record.get("npu_runtime") != "standalone" or record["core"] != core_record:
        parser.error("installed client requires a matching standalone core and backend")
    for build, identity in ((core, core_record), (backend, record)):
        for name, expected in identity["binary_sha256"].items():
            if digest(build / name) != expected:
                parser.error("changed build artifact: " + name)
    out.mkdir(parents=True, exist_ok=False)
    prefix, build = out / "installed prefix", out / "consumer"
    source, dirty = source_state(root)
    report = dict(schema="tide-resident-consumer-build-v1", source=source, dirty=dirty,
                  core=core_record, backend=record, state="running", commands=[],
                  consumer_sources={p.name:digest(p) for p in (root / "tools/resident_consumer").iterdir() if p.is_file()},
                  scope="installed-package build/loader only; live device execution required separately")
    def run(command):
        report["commands"].append(list(map(str, command)))
        write_json(out / "result.json", report)
        subprocess.run(report["commands"][-1], check=True)
    try:
        for directory in (core, backend):
            run(["cmake", "--install", directory, "--prefix", prefix])
        package, = prefix.glob("lib*/cmake/TideResident")
        prefixes = [str(prefix), *os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)]
        run(["cmake", "-S", root / "tools/resident_consumer", "-B", build, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DTideResident_DIR="+str(package),
             "-DTorch_DIR="+cache_values(core)["Torch_DIR"], "-DCMAKE_PREFIX_PATH="+";".join(p for p in prefixes if p)])
        run(["cmake", "--build", build, "--parallel", "2"])
        binary = build / "resident-consumer"
        loader = subprocess.check_output(["ldd", str(binary)], text=True)
        if any(x in loader.lower() for x in ("not found", "libtorch_python", "libpython", "/stub/", "/stubs/", "/simulator/")):
            raise RuntimeError("installed resident loader closure failed")
        (out / "loader.txt").write_text(loader)
        report.update(state="passed", binary_sha256=digest(binary), loader_sha256=digest(out / "loader.txt"))
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", report)


if __name__ == "__main__":
    main()
