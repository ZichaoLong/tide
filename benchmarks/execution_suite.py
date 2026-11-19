"""Bounded serial paired repeats under one already-admitted device allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from benchmarks.programme import run_child
from scripts.job import atomic, enable_subreaper


def build_cases(args):
    subjects = [("tide", variant) for variant in args.variants]
    if args.include_transformer:
        subjects.append(("transformer", "baseline"))
    if not subjects or not 1 <= args.repeats <= 3 or len(subjects) > 5:
        raise ValueError("one to five subjects and one to three repeats required")
    cases = []
    for repeat in range(args.repeats):
        # Alternate order to expose systematic drift; all children keep the
        # same source, physical admission, initial seed and numerical policy.
        order = subjects if repeat % 2 == 0 else list(reversed(subjects))
        for subject, variant in order:
            identity = f"r{repeat}-{subject}-{variant}"
            output = args.output_dir / (identity+".json")
            command = [sys.executable, "-u", "-m", "benchmarks.execution_candidates",
                "--config", str(args.config.resolve()), "--subject", subject,
                "--variant", variant, "--mode", args.mode, "--device", args.device,
                "--batch", str(args.batch), "--length", str(args.length),
                "--chunk", str(args.chunk), "--decode", str(args.decode),
                "--warmup", str(args.warmup), "--samples", str(args.samples),
                "--threads", str(args.threads), "--repeat", str(repeat),
                "--pair-id", args.output_dir.name, "--output", str(output),
                "--allocated-budget-gib", "56", "--reserved-budget-gib", "56"]
            if args.fixed_large_transformer:
                command += ["--transformer-d", "4352", "--transformer-ffn-width", "15208",
                            "--transformer-head-dim", "64"]
            if repeat == 0 and args.observe_first:
                command += ["--observe-work", "--memory-ledger"]
            cases.append(dict(identity=identity, state="planned", command=command,
                output=str(output), log=str(args.output_dir/(identity+".log")),
                timeout_seconds=args.child_timeout, max_rss_bytes=192*(1 << 30)))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="*", choices=("baseline", "weight_access", "packing", "combined"), default=["baseline"])
    parser.add_argument("--include-transformer", action="store_true")
    parser.add_argument("--fixed-large-transformer", action="store_true")
    parser.add_argument("--mode", choices=("train", "inference"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--child-timeout", type=float, default=600)
    parser.add_argument("--total-seconds", type=float, default=6600)
    parser.add_argument("--observe-first", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= args.threads <= 32 or args.child_timeout <= 0 or args.total_seconds <= 0
            or len(set(args.variants)) != len(args.variants)):
        parser.error("invalid finite thread/time/variant bounds")
    args.output_dir = args.output_dir.resolve()
    cases = build_cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(state="planned", source=os.environ.get("TIDE_SOURCE_COMMIT"),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        rows=cases, timing="serial independent children; alternating order; no simultaneous paired model residency")
    def persist():
        atomic(args.output_dir/"suite.json", record)
    persist()
    if args.plan_only:
        return 0
    enable_subreaper()
    start = time.monotonic()
    record["state"] = "running"
    for row in cases:
        remaining = args.total_seconds-(time.monotonic()-start)
        if remaining <= 1:
            record["state"] = "time_budget_exhausted"
            break
        row["timeout_seconds"] = min(row["timeout_seconds"], remaining)
        run_child(row, persist, supervisor_pid=os.getpid())
        if row["state"] != "completed" or row.get("inner_state") != "completed_finite":
            record["state"] = "child_failed"
            break
    else:
        record["state"] = "completed"
    record["elapsed_seconds"] = time.monotonic()-start
    persist()
    print(json.dumps(dict(state=record["state"], output=str(args.output_dir/"suite.json"))))
    return int(record["state"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
