"""Bounded single-axis capacity/throughput searches inside scripts/launch.py.

One identity owns one config, precision, mode and axis. Every audit and measured
subject has an independent process. 'common' rematches A-D at each common B/L;
'capacity' freezes the calibration models and searches B separately at fixed L.
The best admissible measured throughput is not a proven global optimum.
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

from benchmarks.programme import DEFAULT_SUBJECTS, _now, run_child
from benchmarks.references import file_identity
from scripts.job import atomic, enable_subreaper


def axis_values(start, maximum, count):
    if start < 1 or maximum < start or not 1 <= count <= 12:
        raise ValueError("positive increasing axis and 1..12 candidate points required")
    values = []
    while start <= maximum and len(values) < count:
        values.append(start)
        start *= 2
    return values


class ThroughputStop:
    def __init__(self, minimum_gain, patience):
        self.minimum_gain, self.patience = minimum_gain, patience
        self.best, self.flat = None, 0

    def observe(self, throughput, stable=True):
        if not stable:
            self.flat = 0
            return False
        if self.best is None or throughput > self.best*(1+self.minimum_gain):
            self.flat = 0
        else:
            self.flat += 1
        self.best = max(self.best or 0, throughput)
        return self.flat >= self.patience


def summarize_child(child, *, subject, mode, batch, length, maximum_noise):
    result = dict(subject=subject, batch=batch, length=length, state=child["state"],
                  admitted=False, stable=False, child=child["identity"],
                  sampled_process_tree_peak_rss_bytes=child.get("sampled_process_tree_peak_rss_bytes"))
    path = Path(child["output"])
    if not path.exists():
        return result
    raw = json.loads(path.read_text())
    result["result_file"] = file_identity(path)
    observations = [r for r in raw.get("observations", []) if r["subject"] == subject and r["mode"] == mode]
    if not observations:
        result["error"] = raw.get("initial_audit_error", raw.get("error"))
        return result
    row = observations[0]
    result.update(row_state=row["state"], error=row.get("error"), failure_kind=row.get("failure_kind"),
                  seconds=row.get("seconds", []), parameters=row.get("parameters"),
                  initial_parameter_sha256=row.get("initial_parameter_sha256"), input_sha256=row.get("input_sha256"),
                  reference_record=raw.get("reference_record"), calibration_record=raw.get("calibration_record"),
                  initial_audit_scope=raw.get("initial_audit_scope", "this initial workload"))
    if child["state"] != "completed" or row["state"] != "completed_finite":
        return result
    initial = raw.get("initial_audits", {}).get(mode)
    calibration = raw.get("calibration_audits", {}).get(mode)
    after = row.get("post_timing_audit")
    active = lambda audit: None if audit is None else audit["effective_touched"]["mean"]
    activity = dict(calibration_effective_touched_mean=active(calibration),
                    initial_effective_touched_mean=active(initial), post_update_effective_touched_mean=active(after))
    activity["initial_minus_calibration"] = (None if initial is None or calibration is None else active(initial)-active(calibration))
    if initial is not None:
        activity["initial_work"] = initial.get("work")
        activity["initial_packing"] = initial.get("query_packing")
    if after is not None:
        activity["post_update_work"] = after.get("work")
        activity["post_update_packing"] = after.get("query_packing")
    result.update(admitted=True, stable=row["sample_range_over_median"] <= maximum_noise,
                  sample_range_over_median=row["sample_range_over_median"],
                  median_seconds=row["median_seconds"], input_positions_per_second=row["input_positions_per_second"],
                  timed_peak_device_bytes=row["peak_device_bytes"], device_memory_by_device=row["device_memory_by_device"],
                  process_lifetime_peak_rss_kib=row["process_lifetime_peak_rss_kib"],
                  actual_storage=row["actual_storage"], activity=activity,
                  joint_match=None if subject == "tide" else row["spec"]["joint_match"])
    return result


def stop_reason(metric):
    if metric["admitted"]:
        return None
    if metric["state"] in {"rss_budget_exceeded", "timed_out", "preflight_rejected", "search_time_exhausted"}:
        return metric["state"]
    if metric.get("failure_kind") in {"device_budget_exceeded", "out_of_memory"}:
        return metric["failure_kind"]
    return "execution_failed"


def best_observation(rows):
    eligible = [r for r in rows if r["admitted"] and r["stable"]]
    return max(eligible, key=lambda r: r["input_positions_per_second"]) if eligible else None


def common_comparisons(rows):
    by_subject = {r["subject"]: r for r in rows if r["admitted"]}
    if "tide" not in by_subject:
        return []
    tide = by_subject["tide"]
    comparisons = []
    for name in "ABCD":
        baseline = by_subject.get("transformer_"+name)
        if baseline is None:
            continue
        if (tide["batch"], tide["length"]) != (baseline["batch"], baseline["length"]):
            raise ValueError("common-shape comparisons require the same B/L")
        comparisons.append(dict(joint_class=name, tide_over_transformer_time=tide["median_seconds"]/baseline["median_seconds"],
                                tide_over_transformer_peak=(tide["timed_peak_device_bytes"]/baseline["timed_peak_device_bytes"]
                                                           if baseline["timed_peak_device_bytes"] else None),
                                both_stable=tide["stable"] and baseline["stable"]))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-overrides", type=json.loads, default={},
                        help="JSON object of top-level execution/config overrides, frozen in output/config.json")
    parser.add_argument("--table", choices=("common", "capacity"), required=True)
    parser.add_argument("--axis", choices=("batch", "length"), default="batch")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-axis-value", type=int, required=True)
    parser.add_argument("--max-points", type=int, default=8)
    parser.add_argument("--subjects", nargs="+", choices=DEFAULT_SUBJECTS, default=DEFAULT_SUBJECTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--mode", choices=("inference", "grad_forward", "train"), default="train")
    parser.add_argument("--activity", choices=("natural", "sparse03", "sparse10", "high"), default="natural")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-budget-gib", type=float)
    parser.add_argument("--rss-gib", type=float, required=True)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--child-timeout", type=float, default=900)
    parser.add_argument("--minimum-gain", type=float, default=.03)
    parser.add_argument("--flat-patience", type=int, default=2)
    parser.add_argument("--maximum-noise", type=float, default=.25)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.table == "capacity" and args.axis != "batch":
        parser.error("capacity table holds L fixed and searches only batch")
    positive = [args.batch, args.length, args.threads, args.steps, args.rss_gib, args.seconds, args.child_timeout, args.flat_patience]
    if args.device_budget_gib is not None:
        positive.append(args.device_budget_gib)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or args.warmup < 0:
        parser.error("positive finite dimensions/resource bounds; nonnegative warmup required")
    if not 0 <= args.minimum_gain < 1 or not 0 <= args.maximum_noise < float("inf"):
        parser.error("minimum gain must be in [0,1), noise finite and nonnegative")
    if args.device != "cpu" and args.device_budget_gib is None:
        parser.error("device runs require an explicit per-device allocator budget")
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("duplicate subjects are not separate measurements")
    if not isinstance(args.config_overrides, dict):
        parser.error("config overrides must be a JSON object")
    try:
        values = axis_values(getattr(args, args.axis), args.max_axis_value, args.max_points)
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"search.json"
    if manifest.exists() or (output/"config.json").exists():
        parser.error("select a new output identity")
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    original_config = file_identity(args.config)
    resolved_config = json.loads(Path(args.config).read_text())
    resolved_config.update(args.config_overrides)
    atomic(output/"config.json", resolved_config)
    record = dict(schema="tide-capacity-search-v1", source=source,
                  harness_source=os.environ.get("TIDE_HARNESS_COMMIT", source), state="planned",
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  config_file=original_config, frozen_config=file_identity(output/"config.json"),
                  axis_values=values, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified; do not infer exclusive access"),
                  fixed_history="Transformer window 128; Tide adopted logical-time KV window unchanged; L is workload extent, not a new history contract",
                  measurement="one subject/mode per process; normal prefill timing excludes audits; complete AdamW and required Timed flush included",
                  budget_scope="sampled child process-tree RSS; device allocator peak checked after synchronization at construction/audit/warmup/sample; not hard allocation isolation",
                  search_scope="finite measured candidates; best stable admissible throughput, not proof of maximum feasible B/L or global optimum",
                  matching_scope=("initial per-shape A-D matching" if args.table == "common" else
                                  "models frozen at common initial calibration B/L; independent best B at fixed L; not a same-workload speedup"),
                  children=[], audits=[], lines=[], common_shape_table=[], capacity_table=[])
    persist = lambda: atomic(manifest, record)
    persist()
    if args.plan_only:
        print(json.dumps(dict(manifest=str(manifest), axis_values=values, state="planned")))
        return 0
    deadline = time.monotonic()+args.seconds
    cache = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    enable_subreaper()

    def child(batch, length, subject=None, *, reference=None, calibration=None):
        identity = f"{len(record['children']):03d}-b{batch}-l{length}-"+(subject or "audit")
        result = output/(identity+".json")
        command = [sys.executable, "-u", "-m", "benchmarks.run", "--config", str(output/"config.json"),
                   "--device", args.device, "--precision", args.precision, "--threads", str(args.threads),
                   "--batch", str(batch), "--length", str(length), "--modes", args.mode,
                   "--subjects", subject or "tide", "--activity", args.activity, "--seed", str(args.seed),
                   "--warmup", str(args.warmup), "--steps", str(args.steps), "--output", str(result),
                   "--optimizer-foreach" if args.optimizer_foreach else "--no-optimizer-foreach"]
        if subject is None:
            command.append("--audit-only")
        if reference is not None:
            command.extend(["--reference-record", str(reference)])
        if calibration is not None:
            command.extend(["--calibration-record", str(calibration)])
        if args.device_budget_gib is not None:
            command.extend(["--device-budget-gib", str(args.device_budget_gib)])
        remaining = deadline-time.monotonic()
        row = dict(identity=identity, command=command, output=str(result), log=str(output/(identity+".log")),
                   batch=batch, length=length, subject=subject or "audit", state="planned",
                   timeout_seconds=min(args.child_timeout, max(0, remaining)), max_rss_bytes=int(args.rss_gib*2**30))
        record["children"].append(row)
        if remaining <= 0:
            row["state"] = "search_time_exhausted"
            persist()
            return row
        # Only an exact persistent-parameter floor can reject before execution.
        # Activation/workspace allowances are observations, not safety proofs.
        if subject is not None and args.device_budget_gib is not None:
            frozen = json.loads(Path(calibration or reference).read_text())
            count = (frozen["initial_audits"][args.mode]["parameters"]["unique_parameters"] if subject == "tide" else
                     frozen["joint_matches_by_mode"][args.mode][subject[-1]]["actual"]["parameters"])
            floor = count*(16 if args.mode == "train" else 4)
            row["preflight_persistent_floor_bytes"] = floor
            if floor > args.device_budget_gib*2**30:
                row.update(state="preflight_rejected", reason="persistent parameter/gradient/AdamW floor exceeds device budget; no execution/OOM observed")
                persist()
                return row
        run_child(row, persist, supervisor_pid=os.getpid())
        print(json.dumps(dict(identity=identity, state=row["state"], elapsed_seconds=row.get("elapsed_seconds"))), flush=True)
        return row

    def audit(batch, length):
        key = batch, length
        if key not in cache:
            row = child(batch, length)
            cache[key] = row["output"] if row["state"] == "completed" and row.get("inner_state") == "audit_complete" else None
            record["audits"].append(dict(batch=batch, length=length, child=row["identity"], state=row["state"],
                                         result_file=None if cache[key] is None else file_identity(cache[key])))
            persist()
        return cache[key]

    def measured(batch, length, subject, reference, calibration=None):
        row = child(batch, length, subject, reference=reference, calibration=calibration)
        return summarize_child(row, subject=subject, mode=args.mode, batch=batch, length=length, maximum_noise=args.maximum_noise)

    record.update(state="running", started=_now())
    persist()
    try:
        if args.table == "common":
            line = dict(subjects=args.subjects, observations=[], stop_reason="candidate_limit")
            record["lines"].append(line)
            stops = {s: ThroughputStop(args.minimum_gain, args.flat_patience) for s in args.subjects}
            for value in values:
                batch, length = (value, args.length) if args.axis == "batch" else (args.batch, value)
                reference = audit(batch, length)
                if reference is None:
                    line["stop_reason"] = "initial_audit_failed_or_limited"
                    break
                rows = [measured(batch, length, subject, reference) for subject in args.subjects]
                line["observations"].extend(rows)
                record["common_shape_table"].append(dict(batch=batch, length=length, subjects=rows,
                                                         comparisons=common_comparisons(rows)))
                reasons = [stop_reason(r) for r in rows if not r["admitted"]]
                flattened = [stops[r["subject"]].observe(r["input_positions_per_second"], r["stable"])
                             for r in rows if r["admitted"]]
                if reasons or flattened and all(flattened):
                    line["stop_reason"] = reasons[0] if reasons else "all_subjects_flat_throughput"
                    break
                persist()
        else:
            calibration = audit(args.batch, args.length)
            if calibration is None:
                raise RuntimeError("initial calibration failed or exceeded its resource bound")
            record["calibration_record"] = file_identity(calibration)
            # Observe Tide first so corresponding initial active drift is available
            # for baseline points. A baseline can still exceed Tide's feasible B:
            # its model remains frozen, and missing Tide activity is explicit.
            subjects = sorted(args.subjects, key=lambda name: name != "tide")
            for subject in subjects:
                line = dict(subject=subject, observations=[], stop_reason="candidate_limit")
                record["lines"].append(line)
                stop = ThroughputStop(args.minimum_gain, args.flat_patience)
                for batch in values:
                    reference = audit(batch, args.length) if subject == "tide" else cache.get((batch, args.length))
                    if subject == "tide" and reference is None:
                        line["stop_reason"] = "initial_audit_failed_or_limited"
                        break
                    metric = measured(batch, args.length, subject, reference, calibration)
                    line["observations"].append(metric)
                    reason = stop_reason(metric)
                    if reason or stop.observe(metric["input_positions_per_second"], metric["stable"]):
                        line["stop_reason"] = reason or "flat_throughput"
                        break
                    persist()
                record["capacity_table"].append(dict(subject=subject, best=best_observation(line["observations"]),
                                                     stop_reason=line["stop_reason"], fixed_length=args.length,
                                                     calibration_record=record["calibration_record"],
                                                     ratio_scope="independent load optimum under the same declared budget; not a same-B comparison"))
        record["state"] = "completed"
    except KeyboardInterrupt:
        record["state"] = "cancelled"
    except Exception as error:
        record.update(state="failed", error=repr(error))
        import traceback
        traceback.print_exc()
    finally:
        record["ended"] = _now()
        record["child_states"] = {state: sum(c["state"] == state for c in record["children"])
                                  for state in sorted({c["state"] for c in record["children"]})}
        record["completion_scope"] = "search controller terminal; failed/limited children remain separate and are not measured passes"
        persist()
    print(json.dumps(dict(manifest=str(manifest), state=record["state"], child_states=record["child_states"])))
    return int(record["state"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
