#!/usr/bin/env python3
"""Dependency-free re-entry: source, current handoff and durable job status."""
from pathlib import Path
import argparse
import json
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--recent", type=int, default=3, help="number of recent terminal jobs (default: 3)")
parser.add_argument("--all-jobs", action="store_true", help="include all historical durable job records")
args = parser.parse_args()
if args.recent < 0:
    parser.error("--recent must be nonnegative")
root = Path(__file__).resolve().parents[1]
subprocess.run(["git", "status", "--short", "--branch"], cwd=root, check=True)
subprocess.run(["git", "log", "-3", "--oneline"], cwd=root, check=True)
print((root / "docs/STATUS.md").read_text())
records = [(p, json.loads(p.read_text())) for p in (root / "artifacts").glob("*/status.json")]
records.sort(key=lambda item: item[1].get("started", ""), reverse=True)
live = [(p, d) for p, d in records if d.get("state") in {"starting", "running"}]
terminal = [(p, d) for p, d in records if d.get("state") not in {"starting", "running"}]
selected = records if args.all_jobs else live + terminal[:args.recent]
print("Durable records: all history" if args.all_jobs else
      f"Durable records: all live and {args.recent} recent terminal jobs (--all-jobs for history)")
for status, data in selected:
    # Raw failures are retained, without repeatedly flooding re-entry with old
    # dirty-file lists or implying that every historical failure is still open.
    summary = {key: data[key] for key in ("state", "source", "started", "finished", "exit_code") if key in data}
    summary["dirty"] = bool(data.get("dirty"))
    summary["log"] = str((status.parent / "task.log").relative_to(root))
    print(f"{status.relative_to(root)}: {json.dumps(summary)}")
