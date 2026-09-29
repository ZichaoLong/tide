#!/usr/bin/env python3
"""Summarize exported CANN CSVs without mistaking task sums for wall time."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import re
from durable_records import write_json
from source_identity import digest


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("profile duration must be finite and nonnegative")
    return value


def read_rows(path, fields):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not fields <= set(reader.fieldnames or []):
            raise ValueError(f"unrecognized {path.name} CSV schema")
        yield from reader


def read_one(root, prefix, fields):
    matches = list(root.glob(prefix + "_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix} CSV, got {len(matches)}")
    return matches[0], list(read_rows(matches[0], fields))


def operator_files(root):
    """Accept one export, including CANN's consecutive million-row slices."""
    matches = sorted(root.glob("op_summary_*.csv"))
    if len(matches) == 1 and not matches[0].name.startswith("op_summary_slice_"):
        return matches
    slices = [re.fullmatch(r"op_summary_slice_(\d+)_(\d+)\.csv", p.name) for p in matches]
    if not slices or any(m is None for m in slices):
        raise ValueError("expected one operator export or one consecutive sliced export")
    indices = [int(m[1]) for m in slices]
    if len({m[2] for m in slices}) != 1 or sorted(indices) != list(range(len(slices))):
        raise ValueError("operator export slices have gaps, duplicate indices or mixed timestamps")
    return [path for _, path in sorted(zip(indices, matches))]


def summarize(root):
    op_files = operator_files(root)
    api_file, apis = read_one(root, "api_statistic", {"Level", "API Name", "Time(us)", "Count"})
    engines = defaultdict(lambda: dict(count=0, summed_task_us=0.))
    kinds = defaultdict(lambda: dict(count=0, summed_task_us=0.))
    count = 0
    fields = {"Task Type", "OP Type", "Task Duration(us)", "Input Data Types"}
    for op_file in op_files:
        for row in read_rows(op_file, fields):
            count += 1
            us = number(row["Task Duration(us)"])
            for entry in (engines[row["Task Type"]], kinds[row["Task Type"], row["OP Type"], row["Input Data Types"]]):
                entry["count"] += 1
                entry["summed_task_us"] += us
    if not count:
        raise ValueError("profile contains no device operators")
    total = sum(x["summed_task_us"] for x in engines.values())
    if total <= 0: raise ValueError("profile task durations are empty")
    for entry in engines.values(): entry["summed_task_fraction"] = entry["summed_task_us"] / total
    api_rows = [dict(level=r["Level"], name=r["API Name"], summed_us=number(r["Time(us)"]),
                     count=int(r["Count"])) for r in apis]
    return dict(schema="tide-cann-csv-summary-v1", operators=count, summed_task_us=total,
                engines=dict(engines), operators_by_type=[dict(engine=key[0], operator=key[1],
                    input_dtypes=key[2], **value) for key, value in sorted(kinds.items())],
                top_host_apis=sorted(api_rows, key=lambda x:x["summed_us"], reverse=True)[:20],
                inputs={f.name:digest(f) for f in (*op_files, api_file)},
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
