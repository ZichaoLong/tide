#!/usr/bin/env python3
"""Query explicit study directories, including failed and incomplete runs.

Print JSON or TSV. Inputs may be run.json, a run directory or its parent; only
one child level is searched. Never scan all historical artifacts implicitly.
"""
import argparse
import json
from pathlib import Path
from core5_study_records import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--format", choices=("json", "tsv"), default="tsv")
    p.add_argument("--state")
    p.add_argument("--case")
    p.add_argument("--client")
    p.add_argument("--device")
    p.add_argument("--dtype")
    a = p.parse_args()
    files = set()
    for path in a.paths:
        if path.is_file():
            files.add(path)
        elif (path/"run.json").is_file():
            files.add(path/"run.json")
        else:
            files.update(path.glob("*/run.json"))
    if not files:
        p.error("no study run records in explicit paths")
    records = []
    for path in sorted(files):
        r = summarize(path)
        if any(getattr(a, key) and getattr(a, key) != r[key] for key in ("state", "case")):
            continue
        if any(getattr(a, key) and getattr(a, key) != r["requested"][key] for key in ("client", "device", "dtype")):
            continue
        records.append(r)
    if a.format == "json":
        print(json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print("case\texecution\tstate\tqualified\tseconds\tparameters\tphase\trun")
        for r in records:
            print("\t".join(str(x) for x in (r["case"], r["execution_name"], r["state"],
                r["throughput_qualified"], r["measured_seconds"], r["parameters"]["trainable"], r["phase"], r["run"])))


if __name__ == "__main__":
    main()
