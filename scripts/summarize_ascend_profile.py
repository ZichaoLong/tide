#!/usr/bin/env python3
"""Summarize exported CANN CSVs without mistaking task sums for wall time."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from durable_records import write_json
from source_identity import digest


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("profile duration must be finite and nonnegative")
    return value


def read_one(root, prefix, fields):
    matches = list(root.glob(prefix + "_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix} CSV, got {len(matches)}")
    with matches[0].open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not fields <= set(reader.fieldnames or []):
            raise ValueError(f"unrecognized {prefix} CSV schema")
        rows = list(reader)
    return matches[0], rows


def summarize(root):
    op_file, ops = read_one(root, "op_summary", {"Task Type", "OP Type", "Task Duration(us)", "Input Data Types"})
    api_file, apis = read_one(root, "api_statistic", {"Level", "API Name", "Time(us)", "Count"})
    if not ops:
        raise ValueError("profile contains no device operators")
    engines = defaultdict(lambda: dict(count=0, summed_task_us=0.))
    kinds = defaultdict(lambda: dict(count=0, summed_task_us=0.))
    for row in ops:
        us = number(row["Task Duration(us)"])
        for entry in (engines[row["Task Type"]], kinds[row["Task Type"], row["OP Type"], row["Input Data Types"]]):
            entry["count"] += 1
            entry["summed_task_us"] += us
    total = sum(x["summed_task_us"] for x in engines.values())
    if total <= 0: raise ValueError("profile task durations are empty")
    for entry in engines.values(): entry["summed_task_fraction"] = entry["summed_task_us"] / total
    api_rows = [dict(level=r["Level"], name=r["API Name"], summed_us=number(r["Time(us)"]),
                     count=int(r["Count"])) for r in apis]
    return dict(schema="tide-cann-csv-summary-v1", operators=len(ops), summed_task_us=total,
                engines=dict(engines), operators_by_type=[dict(engine=key[0], operator=key[1],
                    input_dtypes=key[2], **value) for key, value in sorted(kinds.items())],
                top_host_apis=sorted(api_rows, key=lambda x:x["summed_us"], reverse=True)[:20],
                inputs={f.name:digest(f) for f in (op_file, api_file)},
                limits=["Task durations may overlap across streams/devices; the sum is not wall time.",
                        "Host API levels can nest; never add them into an end-to-end total.",
                        "Scope/warmup must come from the launch record; no full-size extrapolation.",
                        "AiCPU is on the accelerator, distinct from host CPU and AiCore."])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.csv_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(a.output_dir/"summary.json", result)
    print(f"{result['operators']} device operators; summary: {a.output_dir/'summary.json'}")


if __name__ == "__main__": main()
