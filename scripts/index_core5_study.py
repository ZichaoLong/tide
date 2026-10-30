#!/usr/bin/env python3
"""Query explicit study directories, including failed and incomplete runs.

Print JSON or TSV. Inputs may be run.json, reviewed finite-performance evidence,
a run directory or its parent; only one child level is searched. Never scan all
historical artifacts implicitly. Reviewed evidence needs no old run directory.
"""
import argparse
import json
from pathlib import Path
from core5_study_records import query_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--format", choices=("json", "tsv", "markdown"), default="tsv")
    p.add_argument("--view", choices=("runs", "coverage", "models"), default="runs")
    p.add_argument("--state")
    p.add_argument("--case")
    p.add_argument("--client")
    p.add_argument("--device")
    p.add_argument("--dtype")
    p.add_argument("--placement")
    p.add_argument("--topology")
    p.add_argument("--family")
    p.add_argument("--profile")
    p.add_argument("--kind", choices=("correctness", "performance"))
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
        for r in query_records(path):
            if any(getattr(a, key) and getattr(a, key) != r[key] for key in ("state", "case")):
                continue
            if a.kind and a.kind != r["kind"]:
                continue
            if any(getattr(a, key) and getattr(a, key) != r["requested"][key] for key in ("client", "device", "dtype", "placement")):
                continue
            if a.view != "runs" or a.topology or a.family or a.profile:
                from core5_coverage import view
                v = view(r)
                if any(getattr(a, key) and getattr(a, key) != v[key] for key in ("topology", "profile")):
                    continue
                if a.family and a.family not in v["families"]:
                    continue
            records.append(r)
    if a.view != "runs":
        from core5_coverage import models, view, table
        records = models(records) if a.view == "models" else [view(r) for r in records]
    if a.format == "json":
        print(json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False))
    elif a.view != "runs":
        headers, rows = table(records, a.view)
        if a.format == "markdown":
            print("| "+" | ".join(headers)+" |")
            print("| "+" | ".join("---" for _ in headers)+" |")
            for row in rows: print("| "+" | ".join(row)+" |")
        else:
            for row in [headers, *rows]: print("\t".join(row))
    else:
        if a.format == "markdown": p.error("markdown requires --view models or coverage")
        print("case\texecution\tstate\tqualified\tseconds\tparameters\tphase\trun")
        for r in records:
            print("\t".join(str(x) for x in (r["case"], r["execution_name"], r["state"],
                r["throughput_qualified"], r["measured_seconds"], r["parameters"]["trainable"], r["phase"], r["run"])))


if __name__ == "__main__":
    main()
