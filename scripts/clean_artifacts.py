#!/usr/bin/env python3
"""Dry-run removal of old, successful, unreferenced project job artifacts."""
import argparse
import datetime
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--older-days", type=int, default=14)
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()
if args.older_days < 1:
    parser.error("retention must be at least one day")
root = Path(__file__).resolve().parents[1]
references = "\n".join(p.read_text() for p in (root / "docs").rglob("*.md"))
cut = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.older_days)
for status in sorted((root / "artifacts").glob("*/status.json")):
    directory = status.parent
    if directory.is_symlink() or directory.name in references:
        continue
    record = json.loads(status.read_text())
    if record.get("state") != "passed" or "finished" not in record:
        continue
    if datetime.datetime.fromisoformat(record["finished"]) >= cut:
        continue
    print(("REMOVE " if args.apply else "WOULD REMOVE ") + str(directory.relative_to(root)))
    if args.apply:
        shutil.rmtree(directory)
