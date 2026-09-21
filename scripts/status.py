#!/usr/bin/env python3
"""Dependency-free re-entry: source, current handoff and durable job status."""
from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parents[1]
subprocess.run(["git", "status", "--short", "--branch"], cwd=root, check=True)
subprocess.run(["git", "log", "-3", "--oneline"], cwd=root, check=True)
print((root / "docs/STATUS.md").read_text())
for status in sorted((root / "artifacts").glob("*/status.json")):
    data = json.loads(status.read_text())
    if data.get("state") in {"starting", "running", "failed"}:
        print(f"{status.relative_to(root)}: {json.dumps(data)}")
