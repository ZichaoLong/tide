#!/usr/bin/env python3
"""Immutable full CPU + complex configuration + installed consumer acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from build_identity import source_hash
from durable_records import write_json
from source_identity import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--reuse-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, dirty = source_state(root)
    if dirty:
        parser.error("library qualification requires a clean immutable source")
    out, build = Path(args.output_dir).resolve(), Path(args.build_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="tide-library-release-v1", source=source, state="running", stages=[])
    write_json(out / "result.json", record)
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "python"))
    commands = []
    if not args.reuse_build:
        commands.append(("build", [sys.executable, "scripts/build.py", "--jobs", "2", "--build-dir", str(build)]))
    commands += [
        ("cpu", [sys.executable,"scripts/verify.py","--device","cpu","--dtype","both","--build-dir",str(build),"--output-dir",str(out/"cpu")]),
        ("complex", [sys.executable,"scripts/library_complex.py","--build-dir",str(build),"--output-dir",str(out/"complex")]),
        ("consumer", [sys.executable,"scripts/library_consumer.py","--build-dir",str(build),"--output-dir",str(out/"consumer")])]
    try:
        if args.reuse_build:
            manifest = json.loads((build / "build-manifest.json").read_text())
            if manifest["cpp_source_sha256"] != source_hash(root):
                raise ValueError("reused build differs from source")
            for name, digest in manifest["binary_sha256"].items():
                if hashlib.sha256((build / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("reused binary changed: " + name)
            record["reused_build"] = manifest
        for name, command in commands:
            print(name + ": " + json.dumps(command), flush=True)
            result = subprocess.run(command, cwd=root, env=env)
            record["stages"].append(dict(name=name, command=command, exit_code=result.returncode))
            write_json(out / "result.json", record)
            if result.returncode:
                raise RuntimeError(f"{name} failed with exit {result.returncode}")
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during acceptance")
        record["state"] = "passed"
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", record)
    print(json.dumps(dict(state=record["state"], source=source)))


if __name__ == "__main__":
    main()
