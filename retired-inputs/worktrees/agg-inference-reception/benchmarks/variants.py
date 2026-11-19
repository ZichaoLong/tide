"""Finite serial execution-candidate comparison on one admitted device group.

Thin orchestration of benchmarks.run and optional separate memory_ledger children.
The outer scripts/launch.py service owns admission; this driver acquires no device.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmarks.programme import _now, run_child
from benchmarks.references import file_identity
from scripts.job import atomic, enable_subreaper


VARIANTS = {
    "baseline": dict(local_attention_backend=None, region_batch_size=1, checkpoint_regions=False),
    "local_math": dict(local_attention_backend="math", region_batch_size=1, checkpoint_regions=False),
    "batch4": dict(local_attention_backend=None, region_batch_size=4, checkpoint_regions=False),
    "recompute": dict(local_attention_backend=None, region_batch_size=1, checkpoint_regions=True),
    "combined": dict(local_attention_backend="math", region_batch_size=4, checkpoint_regions=True),
    "batch_recompute": dict(local_attention_backend=None, region_batch_size=4, checkpoint_regions=True),
    "batch_reception": dict(local_attention_backend=None, region_batch_size=4, checkpoint_regions=False,
                            checkpoint_reception=True),
    "batch_prefix": dict(local_attention_backend=None, region_batch_size=4, checkpoint_regions=False,
                         ssd_summary_backend="prefix"),
    "batch_prefix_reception": dict(local_attention_backend=None, region_batch_size=4, checkpoint_regions=False,
                                   checkpoint_reception=True, ssd_summary_backend="prefix"),
}
DEFAULT_VARIANTS = ["baseline", "local_math", "batch4", "recompute", "combined"]
VARIANTS = {name: {"checkpoint_reception": False, "ssd_summary_backend": "block", **options}
            for name, options in VARIANTS.items()}


def build_plan(args, output):
    config = json.loads(Path(args.config).read_text())
    directory = output/"configs"
    directory.mkdir()
    plan = []
    for name in args.variants:
        resolved = dict(config, **VARIANTS[name])
        path = directory/(name+".json")
        atomic(path, resolved)
    for kind, names in (("normal", args.variants), ("ledger", args.ledger_variants)):
        for name in names:
            identity = f"{len(plan):02d}-{name}-{kind}"
            config_path = directory/(name+".json")
            result = output/(identity+".json")
            common = ["--config", str(config_path), "--device", args.device, "--precision", args.precision,
                      "--batch", str(args.batch), "--length", str(args.length), "--threads", str(args.threads),
                      "--seed", str(args.seed), "--warmup", str(args.warmup), "--activity", args.activity,
                      "--optimizer-foreach" if args.optimizer_foreach else "--no-optimizer-foreach", "--output", str(result)]
            if args.device_budget_gib is not None:
                common.extend(["--device-budget-gib", str(args.device_budget_gib)])
            flags = (["--subjects", "tide", "--modes", args.mode, "--steps", str(args.steps)] if kind == "normal" else
                     ["--subject", "tide", "--mode", args.mode, "--segments", str(args.ledger_segments)])
            command = [sys.executable, "-u", "-m", "benchmarks.run" if kind == "normal" else "benchmarks.memory_ledger", *common, *flags]
            plan.append(dict(identity=identity, variant=name, kind=kind, state="planned", command=command,
                             config_file=file_identity(config_path), output=str(result), log=str(output/(identity+".log")),
                             timeout_seconds=args.child_timeout, max_rss_bytes=int(args.rss_gib*2**30)))
    return plan


def summary(children, mode):
    rows = []
    for child in children:
        if child["kind"] != "normal" or child["state"] != "completed":
            continue
        raw = json.loads(Path(child["output"]).read_text())
        observed = next((row for row in raw.get("observations", []) if row["subject"] == "tide" and row["mode"] == mode), None)
        if observed is None or observed["state"] != "completed_finite":
            continue
        rows.append(dict(variant=child["variant"], result_file=file_identity(child["output"]),
                         initial_parameter_sha256=observed["initial_parameter_sha256"], input_sha256=observed["input_sha256"],
                         median_seconds=observed["median_seconds"], input_positions_per_second=observed["input_positions_per_second"],
                         seconds=observed["seconds"], sample_range_over_median=observed["sample_range_over_median"],
                         timed_peak_device_bytes=observed["peak_device_bytes"],
                         initial_work=raw["initial_audits"][mode]["work"],
                         post_update_work=observed["post_timing_audit"]["work"],
                         numerical_equivalence_qualification=False))
    baseline = next((row for row in rows if row["variant"] == "baseline"), None)
    if baseline:
        for row in rows:
            row.update(initial_parameter_input_equal=all(row[key] == baseline[key] for key in ("initial_parameter_sha256", "input_sha256")),
                       initial_work_summary_equal=row["initial_work"] == baseline["initial_work"],
                       candidate_over_baseline_time=row["median_seconds"]/baseline["median_seconds"],
                       candidate_over_baseline_peak=(row["timed_peak_device_bytes"]/baseline["timed_peak_device_bytes"]
                                                     if baseline["timed_peak_device_bytes"] else None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument("--ledger-variants", nargs="*", choices=VARIANTS, default=[])
    parser.add_argument("--ledger-segments", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--mode", choices=("inference", "grad_forward", "train"), default="train")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--activity", choices=("natural", "sparse03", "sparse10", "high"), default="natural")
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-budget-gib", type=float)
    parser.add_argument("--rss-gib", type=float, required=True)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--child-timeout", type=float, default=900)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    positive = [args.batch, args.length, args.threads, args.steps, args.ledger_segments, args.rss_gib, args.seconds, args.child_timeout]
    if args.device_budget_gib is not None:
        positive.append(args.device_budget_gib)
    if any(not math.isfinite(value) or value <= 0 for value in positive) or args.warmup < 0:
        parser.error("positive finite shapes/resource limits and nonnegative warmup required")
    if args.device != "cpu" and args.device_budget_gib is None:
        parser.error("accelerator variants require an explicit allocator budget")
    if len(set(args.variants)) != len(args.variants) or len(set(args.ledger_variants)) != len(args.ledger_variants):
        parser.error("duplicate variants do not create paired independent repetitions")
    if not set(args.ledger_variants) <= set(args.variants):
        parser.error("ledger variants must have a corresponding normal candidate")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"variants.json"
    if manifest.exists() or (output/"configs").exists():
        parser.error("choose a new output identity")
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(schema="tide-execution-variants-v1", source=source, state="planned",
                  harness_source=os.environ.get("TIDE_HARNESS_COMMIT", source),
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  config_file=file_identity(args.config), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified; admission belongs to the outer launcher"),
                  scope="fixed B/L and explicitly selected finite execution candidates; serial independent children on the same inherited device mapping; not capacity search or automatic numerical qualification",
                  timing_scope="normal benchmark samples only; semantic/runtime audits outside timing; selected ledger children run separately after all normal candidates",
                  pairing_scope="same seed/input shape; initial hashes are checked in summary; work summaries are not full support or trajectory equality",
                  order_scope="fixed declared candidate order; same device does not remove time/order/host-contention effects",
                  children=build_plan(args, output), summaries=[])
    persist = lambda: atomic(manifest, record)
    persist()
    if args.plan_only:
        print(json.dumps(dict(state="planned", manifest=str(manifest), children=len(record["children"]))))
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    enable_subreaper()
    deadline = time.monotonic()+args.seconds
    record.update(state="running", started=_now())
    persist()
    try:
        for child in record["children"]:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                child["state"] = "not_started_time_limit"
                persist()
                continue
            child["timeout_seconds"] = min(child["timeout_seconds"], remaining)
            run_child(child, persist, supervisor_pid=os.getpid())
            print(json.dumps(dict(identity=child["identity"], state=child["state"])), flush=True)
        record["summaries"] = summary(record["children"], args.mode)
        record["state"] = "completed" if all(child["state"] == "completed" for child in record["children"]) else "failed"
    except KeyboardInterrupt:
        record["state"] = "cancelled"
    except Exception as error:
        record.update(state="failed", error=repr(error))
    record["ended"] = _now()
    persist()
    return int(record["state"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
