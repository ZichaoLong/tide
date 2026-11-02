#!/usr/bin/env python3
"""Sequential fixed-stage assets; one failed model does not hide other results."""
import argparse
from pathlib import Path
from tidegraph.lm.assets import fetch
from tidegraph.lm.catalog import catalog
from tidegraph.experiment.records import now, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root")
parser.add_argument("--stage", type=int, choices=(1,2), required=True)
args = parser.parse_args()
root = Path(args.root).resolve()
results = {}
for key, spec in catalog().items():
    if spec["stage"] != args.stage:
        continue
    try:
        fetch(key, root/"models", root, project_gib=80 if args.stage==1 else 160)
        results[key] = dict(state="complete")
    except Exception as error:
        results[key] = dict(state="failed", error=str(error))
        print(f"{key}: failed: {error}", flush=True)
    write_json(root/f"asset-stage{args.stage}.json", dict(updated=now(), models=results))
raise SystemExit(0 if all(x["state"]=="complete" for x in results.values()) else 1)
