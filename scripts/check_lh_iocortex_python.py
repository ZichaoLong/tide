#!/usr/bin/env python3
"""Check independent Python schedules against hashed actual-LH whole-model fixtures."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from build_identity import revision, source_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", required=True, choices=("cpu",))
parser.add_argument("--oracle-result", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
if sys.flags.optimize:
    parser.error("the original-fixture checker requires Python assertions enabled")
root = Path(__file__).resolve().parents[1]
manifest = Path(args.oracle_result).resolve(); oracle = json.loads(manifest.read_text())
if oracle.get("scope") != "full":
    parser.error("a complete full-scope original oracle is required; smoke cannot qualify Python fixtures")
if (oracle["state"] != "passed" or oracle["source"] != revision(root)
        or oracle["build"]["cpp_source_sha256"] != source_hash(root)):
    parser.error("a passed original oracle from the current frozen source is required")
digests = oracle.get("fixture_sha256", {})
expected_dtypes = {"float32", "float64"} if oracle["dtype"] == "both" else {oracle["dtype"]}
files = {str(p.relative_to(manifest.parent)) for p in manifest.parent.glob("fixtures-*/*.json")}
if not digests or files != digests.keys(): parser.error("original fixture inventory mismatch")
for name, digest in digests.items():
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or hashlib.sha256((manifest.parent/path).read_bytes()).hexdigest() != digest:
        parser.error("original fixture path/hash mismatch")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
sys.path[:0] = [str(root/"python"), str(root/"tests")]
from iocortex_fixture import check
out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
def fingerprint():
    paths = sorted((root/"python").rglob("*.py"))+[root/"tests/iocortex_fixture.py",
        root/"tests/single_graph_adapter.py", root/"tests/single_graph_checks.py",
        root/"tests/single_graph_compare.py", Path(__file__)]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
record = {"state": "running", "source": revision(root),
          "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
          "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "oracle_result": str(manifest), "oracle_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
          "python_sha256": fingerprint(), "fixture_sha256": digests, "torch": torch.__version__,
          "device": "cpu", "resolution_reason": "explicit:cpu", "cases": 0, "events": 0, "single_cuts": 0}
def save():
    temp = out/"result.tmp"; temp.write_text(json.dumps(record, indent=2)+"\n"); temp.replace(out/"result.json")
save(); inventory = set()
try:
    with (out/"checks.log").open("w") as log:
        for name in sorted(digests):
            data = json.loads((manifest.parent/name).read_text())
            key = (data["dtype"], data["pool"], data["clear"], data["scenario"])
            if key in inventory: raise ValueError("duplicate original scenario")
            inventory.add(key)
            try: events, single_cuts = check(data)
            except Exception as error: raise RuntimeError(f"{name}: {error}") from error
            record["cases"] += 1; record["events"] += events; record["single_cuts"] += single_cuts
            log.write(f"passed {name}: {events} original candidate events, {single_cuts} single-PDG cuts\n"); log.flush(); save()
    expected = {(dtype, pool, clear, scenario) for dtype in expected_dtypes
                for pool in ("add", "sum", "mean", "linear", "active-softmax", "all-softmax")
                for clear in (False, True) for scenario in ("tokens", "ragged")}
    if (inventory != expected or record["python_sha256"] != fingerprint()
            or source_hash(root) != oracle["build"]["cpp_source_sha256"] or revision(root) != record["source"]
            or hashlib.sha256(manifest.read_bytes()).hexdigest() != record["oracle_sha256"]):
        raise ValueError("fixture scope, source or oracle identity changed during comparison")
    for name, digest in digests.items():
        if hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest() != digest:
            raise ValueError("original fixture changed during comparison")
    record.update(state="passed", exit_code=0)
except Exception as error:
    record.update(state="failed", exit_code=1, error=str(error))
finally:
    record["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
print(json.dumps({k: record.get(k) for k in ("state", "cases", "events", "single_cuts", "error")}))
sys.exit(record["exit_code"])
