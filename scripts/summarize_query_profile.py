#!/usr/bin/env python3
"""Summarize TorchNPU CSV exports, keeping task sums distinct from wall time."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from durable_records import write_json
from source_identity import digest


def rows(path, required):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"unsupported profile schema: {path.name}")
        yield from reader


def duration(text):
    value = float(text or 0)
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid duration")
    return value


def summarize(root):
    operators = defaultdict(lambda: dict(count=0, host_self_us=0., device_self_us=0.))
    for row in rows(root/"operator_details.csv", {"Name", "Host Self Duration(us)", "Device Self Duration(us)"}):
        item = operators[row["Name"]]; item["count"] += 1
        item["host_self_us"] += duration(row["Host Self Duration(us)"])
        item["device_self_us"] += duration(row["Device Self Duration(us)"])
    kernels = defaultdict(lambda: dict(count=0, summed_task_us=0.))
    for row in rows(root/"kernel_details.csv", {"Name", "Duration(us)"}):
        item = kernels[row["Name"]]; item["count"] += 1
        item["summed_task_us"] += duration(row["Duration(us)"])
    if not kernels:
        raise ValueError("no actual device records")
    steps = list(rows(root/"step_trace_time.csv", {"Computing", "Free", "Stage"}))
    tasks = defaultdict(lambda: dict(count=0, summed_task_us=0.))
    aicpu = defaultdict(lambda: dict(count=0, summed_task_us=0.))
    task_file = root/"task_time.csv"
    if task_file.exists():
        for row in rows(task_file, {"kernel_type", "kernel_name", "task_time(us)"}):
            kind = row["kernel_type"]
            item = tasks[kind]; item["count"] += 1
            item["summed_task_us"] += duration(row["task_time(us)"])
            if "AICPU" in kind.upper():
                item = aicpu[row["kernel_name"]]; item["count"] += 1
                item["summed_task_us"] += duration(row["task_time(us)"])
    files = [root/"operator_details.csv", root/"kernel_details.csv", root/"step_trace_time.csv"]
    if task_file.exists():
        files.append(task_file)
    return dict(schema="tide-query-profile-summary-v1", csv_sha256={p.name: digest(p) for p in
                files},
                device_kernel_records=sum(x["count"] for x in kernels.values()),
                top_host_self=sorted([dict(name=n, **v) for n, v in operators.items()],
                                     key=lambda x: x["host_self_us"], reverse=True)[:25],
                operators={n: operators[n] for n in sorted(operators) if any(s in n for s in
                    ("matmul", "bmm", "nonzero", "_local_scalar", "index", "select", "cat", "Private", "Query", "Region"))},
                kernels=[dict(name=n, **v) for n, v in sorted(kernels.items())], step_trace_time=steps,
                task_types=dict(tasks), aicpu_kernels=dict(aicpu),
                scope="Supplied CSV exports only; workload/source identity belongs to the enclosing run record. Task sums may overlap and host totals may nest. Free is vendor timeline classification, not proof of one unique bottleneck. No timing extrapolation to other workloads.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output, summarize(a.csv_dir))


if __name__ == "__main__":
    main()
