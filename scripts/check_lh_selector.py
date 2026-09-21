#!/usr/bin/env python3
"""Build/run original LH component parity against a fingerprinted Tide CPU library."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from build_identity import source_hash, revision

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", required=True, choices=("cpu",))
parser.add_argument("--dtype", default="both", choices=("float32", "float64", "both"))
parser.add_argument("--snapshot", required=True)
parser.add_argument("--core-build-dir", default="build")
parser.add_argument("--oracle-build-dir", default="build/lh-oracle", help="reusable CMake cache; result/log directories remain unique")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--jobs", type=int, default=2)
parser.add_argument("--component", choices=("selector", "add", "full", "attention", "all"), default="selector")
args = parser.parse_args()
if args.jobs < 1:
    parser.error("positive build jobs required")
root = Path(__file__).resolve().parents[1]
core, snapshot, out = Path(args.core_build_dir).resolve(), Path(args.snapshot).resolve(), Path(args.output_dir).resolve()
oracle_build = Path(args.oracle_build_dir).resolve()
if oracle_build in (core, snapshot) or snapshot in oracle_build.parents or oracle_build in snapshot.parents:
    parser.error("oracle build directory must be separate from core and source snapshot")
build_record = json.loads((core/"build-manifest.json").read_text())
manifest = json.loads((snapshot/"manifest.json").read_text())
if manifest.get("schema") != "lh-cpp-snapshot-v1":
    parser.error("unsupported LH snapshot schema")
def snapshot_identity():
    actual = {str(p.relative_to(snapshot)) for p in snapshot.rglob("*") if p.is_file() and p != snapshot/"manifest.json"}
    if actual != manifest["files"].keys():
        raise ValueError("LH snapshot file inventory changed")
    digests = {}
    for name in manifest["files"]:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid snapshot relative path")
        digests[name] = hashlib.sha256((snapshot/path).read_bytes()).hexdigest()
    if digests != manifest["files"]:
        raise ValueError("LH snapshot contents changed")
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
if snapshot_identity() != manifest["identity"]:
    parser.error("LH snapshot identity mismatch")
library = core/"libtidegraph.a"
library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
if (build_record["cpp_source_sha256"] != source_hash(root)
        or build_record["binary_sha256"].get(library.name) != library_hash):
    parser.error("Tide core does not match current source/library; run scripts/build.py")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
if build_record["torch"] != torch.__version__ or build_record["cxx11_abi"] != torch.compiled_with_cxx11_abi():
    parser.error("core/oracle Torch environment mismatch")
out.mkdir(parents=True, exist_ok=False)
cmake = root/"tests/lh_oracle/CMakeLists.txt"
record = {"source": revision(root), "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
          "state": "running", "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "snapshot": manifest, "build": build_record, "device": "cpu", "resolution_reason": "explicit:cpu",
          "dtype": args.dtype, "component": args.component,
          "oracle_cmake_sha256": hashlib.sha256(cmake.read_bytes()).hexdigest(),
          "oracle_build_dir": str(oracle_build), "runs": []}
def save():
    temp = out/"result.tmp"; temp.write_text(json.dumps(record, indent=2)+"\n"); temp.replace(out/"result.json")
save()
try:
    components = ("selector", "add", "full", "attention") if args.component == "all" else (args.component,)
    with (out/"build.log").open("w") as log:
        subprocess.run(["cmake", "-S", str(cmake.parent), "-B", str(oracle_build), "-G", "Ninja",
                        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
                        f"-DTIDE_LH_SNAPSHOT={snapshot}", f"-DTIDE_CORE_LIBRARY={library}",
                        f"-DTIDE_LH_ADD={'OFF' if args.component == 'selector' else 'ON'}"], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["cmake", "--build", str(oracle_build), "--parallel", str(args.jobs), "--target",
                        *(f"lh-{component}-check" for component in components)], stdout=log, stderr=subprocess.STDOUT, check=True)
    record["binary_sha256"] = {}
    for component in components:
        binary = oracle_build/f"lh-{component}-check"
        record["binary_sha256"][component] = hashlib.sha256(binary.read_bytes()).hexdigest()
        for dtype in (("float64", "float32") if args.dtype == "both" else (args.dtype,)):
            with (out/f"{component}-{dtype}.log").open("w") as log:
                command = [str(binary), "--device", "cpu", "--dtype", dtype]
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record["runs"].append({"component": component, "command": command, "exit_code": result.returncode}); save()
            if result.returncode:
                raise RuntimeError(f"LH {component} comparison failed in {dtype}; inspect its log")
    if (source_hash(root) != build_record["cpp_source_sha256"] or snapshot_identity() != manifest["identity"]
            or hashlib.sha256(library.read_bytes()).hexdigest() != library_hash
            or hashlib.sha256(cmake.read_bytes()).hexdigest() != record["oracle_cmake_sha256"]):
        raise RuntimeError("source or library changed during qualification")
    record.update(state="passed", exit_code=0)
except Exception as error:
    record.update(state="failed", exit_code=1, error=str(error))
finally:
    record["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
print(json.dumps({"state": record["state"], "result": str(out/"result.json")}))
sys.exit(record["exit_code"])
