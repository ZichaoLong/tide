#!/usr/bin/env python3
"""Copy actual LH C++ sources and JSON headers without modifying the reference tree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True)
parser.add_argument("--json-include", required=True, help="directory containing nlohmann/")
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
source, vendor, target = Path(args.source).resolve(), Path(args.json_include).resolve(), Path(args.output_dir).resolve()
cpp = source / "Connectome/cpp"
if not (cpp / "src/Selector.cpp").is_file() or not (vendor / "nlohmann/json.hpp").is_file():
    parser.error("LH Selector.cpp and nlohmann/json.hpp are required")
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
files = [(p, p.relative_to(cpp)) for directory in ("include", "src") for p in sorted((cpp/directory).rglob("*"))
         if p.is_file() and p.suffix in {".h", ".hpp", ".cpp"}]
files += [(p, Path("vendor")/p.relative_to(vendor)) for p in sorted((vendor/"nlohmann").rglob("*")) if p.is_file()]
target.mkdir(parents=True, exist_ok=False)
digests = {}
for path, relative in files:
    data = path.read_bytes(); dest = target/relative
    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
    digests[str(relative)] = hashlib.sha256(data).hexdigest()
for path, relative in files:
    if hashlib.sha256(path.read_bytes()).hexdigest() != digests[str(relative)]:
        raise RuntimeError(f"source changed during snapshot: {relative}; discard incomplete snapshot")
identity = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
record = {"schema": "lh-cpp-snapshot-v1", "identity": identity, "source_head": head,
          "source_dirty": dirty, "files": digests}
(target/"manifest.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({"snapshot": str(target), "identity": identity, "source_head": head, "files": len(files)}))
