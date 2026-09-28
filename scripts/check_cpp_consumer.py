#!/usr/bin/env python3
"""Install and relocate the CMake consumer; Python only orchestrates the build."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from build_environment import cache_values, digest
from durable_records import write_json
from source_identity import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build, out = Path(args.build_dir).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    manifest = json.loads((build / "build-manifest.json").read_text())
    record = dict(schema="tide-cpp-consumer-v1", source=source, dirty=dirty, state="running",
                  device=args.device, dtype=args.dtype, build=manifest, commands=[])
    write_json(out / "result.json", record)
    def run(command):
        command = [str(x) for x in command]
        record["commands"].append(command)
        with (out / "commands.log").open("a") as log:
            log.write(json.dumps(command)+"\n"); log.flush()
            return subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        if manifest["backend"] == "npu" and manifest["npu_runtime"] != "standalone":
            raise ValueError("standalone consumption needs the standalone NPU build")
        run(["cmake", "--install", build, "--prefix", out / "prefix"])
        shutil.copytree(root / "examples/consumer_cpp", out / "application")
        prefixes = [str(out / "prefix"), *os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)]
        torch_dir = cache_values(build)["Torch_DIR"]
        run(["cmake", "-S", out / "application", "-B", out / "build", "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_PREFIX_PATH="+";".join(p for p in prefixes if p),
             "-DTorch_DIR="+torch_dir])
        run(["cmake", "--build", out / "build", "--parallel", "2"])
        binary = out / "build/consumer"
        closure = subprocess.check_output(["ldd", str(binary)], text=True)
        (out / "loader.txt").write_text(closure)
        if any(term in closure.lower() for term in ("not found", "/stubs/", "/stub/", "libtorch_python", "libpython")):
            raise RuntimeError("standalone loader closure is incomplete or includes Python/stubs")
        run([binary, "--device", args.device, "--dtype", args.dtype])
        if source_state(root) != (source, dirty):
            raise RuntimeError("consumer source changed during verification")
        record.update(state="passed", binary_sha256=digest(binary), loader_sha256=digest(out / "loader.txt"))
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", record)
    print(json.dumps(dict(state=record["state"], device=args.device)))


if __name__ == "__main__":
    main()
