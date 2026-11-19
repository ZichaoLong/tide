"""Finite one-candidate memory comparison; run inside the durable launcher.

This driver owns only serial child commands, reviewed outcomes and bounded batch
selection. Admission, frozen source, whole-service time/RSS and device ownership
remain with scripts/launch.py. Every normal inference point has a separate full
work audit; diagnostics are never added to normal timing.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

from benchmarks.programme import _now, run_child
from scripts.job import atomic, enable_subreaper


GIB = 2**30
SCREEN_BATCHES = (12, 16, 24, 32, 48, 64)
MAX_SCREENS = 3
ALLOCATED_GIB, RESERVED_GIB = 48, 56
CHILD_TIMEOUT = 1800


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def budget_refusal(result):
    """Recognize explicit sampled-budget refusals, never hardware OOM/errors."""
    error = result.get("error", "")
    return (result.get("failure_kind") == "device_budget_exceeded" or
            error.startswith("DeviceBudgetExceeded(") or
            error.startswith("MemoryError: ") and "observed " in error and
            "peak exceeds" in error and "GiB sampled budget" in error)


def select_best(points, *, formal_only=False):
    candidates = [point for point in points if point.get("within_budget") and
                  (not formal_only or point["formal"]) and
                  math.isfinite(point["stable_decode_positions_per_second"]) and
                  point["stable_decode_positions_per_second"] > 0]
    return max(candidates, key=lambda point: (point["stable_decode_positions_per_second"], -point["batch"])) if candidates else None


def project_budget(points, batch):
    """Conservative linear planning from observed peaks, not a capacity proof.

    Anchor at the largest measured batch and use the largest nonnegative adjacent
    secant, also retaining the anchor's per-batch excess over FP32 weight storage.
    Allocated and reserved are projected independently, using cold/normal maxima.
    """
    by_batch = {}
    for point in points:
        if not point.get("within_budget"):
            continue
        previous = by_batch.get(point["batch"])
        if previous is None:
            by_batch[point["batch"]] = point
        else:
            by_batch[point["batch"]] = dict(point, peak_bytes={kind: max(point["peak_bytes"][kind], previous["peak_bytes"][kind])
                for kind in ("allocated", "reserved")})
    ordered = [by_batch[key] for key in sorted(by_batch)]
    if not ordered or any(point["peak_bytes"].get(kind) is None for point in ordered for kind in ("allocated", "reserved")):
        return dict(state="unavailable", batch=batch, reason="no accelerator peak observation; do not infer fit")
    anchor = ordered[-1]
    if batch <= anchor["batch"]:
        raise ValueError("projection requires a larger batch than observed points")
    projected, slopes = {}, {}
    for kind in ("allocated", "reserved"):
        slope = max(0, (anchor["peak_bytes"][kind]-anchor["parameter_bytes"])/anchor["batch"])
        for left, right in zip(ordered, ordered[1:]):
            slope = max(slope, (right["peak_bytes"][kind]-left["peak_bytes"][kind])/(right["batch"]-left["batch"]))
        slopes[kind] = slope
        projected[kind] = math.ceil(anchor["peak_bytes"][kind]+slope*(batch-anchor["batch"]))
    fits = projected["allocated"] <= ALLOCATED_GIB*GIB and projected["reserved"] <= RESERVED_GIB*GIB
    return dict(state="projected_fit" if fits else "projected_budget_refusal", batch=batch,
                anchor_batch=anchor["batch"], source_identities=[point["identity"] for point in ordered],
                predicted_peak_bytes=projected, slope_bytes_per_batch=slopes,
                method="largest observed nonnegative adjacent slope or anchor excess over FP32 weights per batch, whichever is greater; not a hardware/allocator fit guarantee")


def next_screen(points):
    """Choose at most twice the previous batch; persist every considered bound."""
    anchor = max(point["batch"] for point in points)
    candidates = [batch for batch in SCREEN_BATCHES if anchor < batch <= 2*anchor]
    predictions = [project_budget(points, batch) for batch in candidates]
    fitting = [row["batch"] for row in predictions if row["state"] == "projected_fit"]
    return (max(fitting) if fitting else None), predictions


def _peaks(observations):
    values = {kind: [] for kind in ("allocated", "reserved")}
    for row in observations:
        for memory in row.get("allocator", {}).values():
            for kind in values:
                values[kind].append(memory["peak_"+kind+"_bytes"])
    return {kind: max(items) if items else None for kind, items in values.items()}


def inference_point(result, child):
    tide = result["subjects"]["tide"]
    observations = tide["observations"]
    diagnostics = tide["diagnostics"]
    if (result["state"] != "completed" or tide["state"] != "completed" or
            len(observations) != child["samples"] or
            not diagnostics["finite_outputs"] or not diagnostics["finite_private_state"] or
            "matching_reference" not in result):
        raise ValueError("inference completion, sample count, finite state or full work audit failed")
    rates = [row["stable_decode"]["positions_per_second"] for row in observations]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in rates):
        raise ValueError("stable decode has no finite positive throughput")
    peak = _peaks(observations + ([tide["cold"]] if "cold" in tide else []))
    within = all(peak[kind] is None or peak[kind] <= limit*GIB
                 for kind, limit in (("allocated", ALLOCATED_GIB), ("reserved", RESERVED_GIB)))
    if not within:
        raise ValueError("completed inference result exceeds its explicit budget")
    median = lambda values: statistics.median(values) if all(value is not None for value in values) else None
    audit = result["matching_reference"]["audit"]
    return dict(identity=child["identity"], batch=child["batch"], formal=child["formal"],
                samples=len(observations), warmup=child["warmup"], within_budget=True,
                stable_decode_positions_per_second=statistics.median(rates), stable_decode_samples=rates,
                first_backbone_output_seconds=median([row["first_any_backbone_output_seconds"] for row in observations]),
                prefill_positions_per_second=median([row["prefill"]["positions_per_second"] for row in observations]),
                complete_positions_per_second=median([row["complete_positions_per_second"] for row in observations]),
                peak_bytes=peak, normal_peak_bytes=_peaks(observations),
                cold_peak_bytes=_peaks([tide["cold"]] if "cold" in tide else []),
                parameters=tide["parameters"], parameter_bytes=4*tide["parameters"],
                final_private_state=observations[-1]["state_memory"],
                output=child["output"], result_sha256=child["result_sha256"],
                source=result["source"], physical_devices=result.get("physical_devices"),
                input_sha256=tide["input_sha256"], initial_parameters=tide["initial_parameters"],
                work=audit["work"], query_packing=audit["query_packing"],
                audit_location="matching_reference.audit in output; includes parameter touches, actual matrices/dispatch and retained stream identity",
                timing_scope="median of independent zero-state normal requests; prefill128/chunk32/decode16; stable decode omits first incremental call; no task head")


def _child(output, identity, module, flags, **metadata):
    result = output/(identity+".json")
    return dict(identity=identity, state="planned", command=[sys.executable, "-u", "-m", module, *flags, "--output", str(result)],
                output=str(result), log=str(output/(identity+".log")), timeout_seconds=CHILD_TIMEOUT,
                max_rss_bytes=96*GIB, **metadata)


def _inference_child(args, output, batch, *, formal, label):
    warmup, samples = (2, 5) if formal else (1, 2)
    flags = ["--config", str(args.config), "--subjects", "tide", "--devices", args.device,
             "--precision", "fp32", "--threads", str(args.threads), "--batch", str(batch),
             "--prefill", "128", "--prefill-chunk", "32", "--decode", "16", "--stream-positions", "0",
             "--warmup", str(warmup), "--samples", str(samples), "--audit", "full",
             "--allocated-budget-gib", str(ALLOCATED_GIB), "--reserved-budget-gib", str(RESERVED_GIB)]
    return _child(output, f"{label}-b{batch}", "benchmarks.inference", flags,
                  phase="inference", batch=batch, formal=formal, warmup=warmup, samples=samples)


def _train_children(args, output, batch):
    common = ["--device", args.device, "--precision", "fp32", "--threads", str(args.threads),
              "--batch", str(batch), "--prefill-chunk", "32", "--decode", "0", "--device-budget-gib", str(ALLOCATED_GIB)]
    checkpoint = output/f"train-b{batch}.pt"
    return [
        _child(output, f"train-b{batch}", "benchmarks.continuous",
               ["--config", str(args.config), *common, "--mode", "train", "--prefill", "128",
                "--updates", "20", "--warmup", "1", "--checkpoint", str(checkpoint)],
               phase="train", batch=batch, checkpoint=str(checkpoint)),
        _child(output, f"restore-b{batch}", "benchmarks.restore",
               ["--checkpoint", str(checkpoint), *common, "--length", "128"],
               phase="restore", batch=batch, checkpoint=str(checkpoint)),
        _child(output, f"ledger-b{batch}", "benchmarks.memory_ledger",
               ["--config", str(args.config), *common, "--mode", "train", "--length", "128",
                "--warmup", "1", "--segments", "2", "--continue-state"], phase="ledger", batch=batch),
    ]


def _terminal_identities(child):
    live = []
    for identity in child.get("descendant_identities", []):
        try:
            fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[19]) == identity["start_ticks"] and fields[0] != "Z":
                live.append(identity)
        except FileNotFoundError:
            pass
    return dict(checked=_now(), live=live, all_recorded_descendants_terminal=not live)


def _execute(child, record, persist):
    record["children"].append(child)
    if file_hash(record["config_file"]["path"]) != record["config_file"]["sha256"]:
        raise RuntimeError("candidate input changed during the finite suite")
    run_child(child, persist, supervisor_pid=os.getpid())
    child["terminal_review"] = _terminal_identities(child)
    result = json.loads(Path(child["output"]).read_text()) if Path(child["output"]).exists() else {}
    child["outcome"] = ("budget_refused" if child["state"] in {"failed", "inner_failed", "completed"} and budget_refusal(result) else
                        "completed" if child["state"] == "completed" else "failed")
    if not child["terminal_review"]["all_recorded_descendants_terminal"]:
        child["outcome"] = "failed"
    persist()
    print(json.dumps({key: child[key] for key in ("identity", "state", "outcome", "elapsed_seconds")}), flush=True)
    return result


def _review_training(result, child):
    phase = child["phase"]
    if phase == "train":
        rows, checkpoint = result["observations"], result["checkpoint"]
        if result["state"] != "completed" or len(rows) != 20:
            raise ValueError("continuous training did not complete twenty measured updates")
        checks = [value for value in checkpoint.values() if isinstance(value, dict) and "finite" in value]
        if (checkpoint["state"] not in {"within_threshold", "finite_deviation"} or
                not checkpoint["saved_data"]["exact"] or any(not row["finite"] or row["metadata_mismatches"] or row["support_mismatches"] for row in checks)):
            raise ValueError("checkpoint has a nonfinite, saved-data, metadata or support failure")
        for row in rows:
            if (not all(row[key] for key in ("finite_output", "finite_parameters", "finite_gradients")) or
                    not all(row["continuation"][key] for key in ("private_state_finite", "private_state_detached", "optimizer_finite"))):
                raise ValueError("continuous state/gradient/optimizer invariant failed")
        stable = rows[2:]
        return dict(measured_updates=20, batch=child["batch"], stable_updates="3-20",
                    stable_input_positions_per_second=sum(row["input_positions"] for row in stable)/sum(row["complete_seconds"] for row in stable),
                    checkpoint_comparison=checkpoint["state"], checkpoint_sha256=file_hash(child["checkpoint"]),
                    note="checkpoint paired update uses one next chunk outside measured updates; original tolerance failures retained")
    expected = "completed_finite"
    if result["state"] != expected:
        raise ValueError(f"{phase} did not complete finitely")
    if phase == "restore":
        if result["optimizer_step_increment_errors"] or result["restored_state_position"] <= result["loaded_state_position"]:
            raise ValueError("fresh-process restored update did not advance optimizer/state")
        return dict(batch=child["batch"], loaded_state_position=result["loaded_state_position"],
                    restored_state_position=result["restored_state_position"],
                    next_update_input_positions_per_second=result["next_update_input_positions_per_second"],
                    qualification="fresh-process actual update; paired numerical comparison belongs to continuous child")
    if len(result["observations"]) != 2 or not all(row["finite"] for row in result["observations"]):
        raise ValueError("two continuing finite memory-ledger segments required")
    return dict(batch=child["batch"], segments=2, timing="intrusive storage/runtime diagnostic; not normal training throughput")


def _review(child, persist, operation):
    try:
        value = operation()
    except Exception as error:
        child.update(outcome="failed", review_state="failed", review_error=repr(error))
        persist()
        raise
    child["review_state"] = "passed"
    persist()
    return value


def run_train(args, output, record, persist):
    for batch in ([args.train_batch, 4] if args.train_batch == 8 else [4]):
        children = _train_children(args, output, batch)
        for child in children:
            if child["phase"] == "restore":
                identity = file_hash(child["checkpoint"])
                child["command"].extend(["--expected-checkpoint-sha256", identity, "--checkpoint-source", record["source"]])
            result = _execute(child, record, persist)
            if child["outcome"] == "budget_refused":
                record["decisions"].append(dict(kind="training_budget_stop", batch=batch, child=child["identity"],
                    fallback_batch=4 if batch == 8 else None, reason="explicit sampled allocation budget; original failed attempt retained"))
                persist()
                break
            if child["outcome"] != "completed":
                return "failed"
            child["review"] = _review(child, persist, lambda: _review_training(result, child))
            persist()
        else:
            record["selected_training_batch"] = batch
            return "completed" if batch == args.train_batch else "completed_after_budget_fallback"
    return "failed_budget"


def run_inference(args, output, record, persist):
    points = record["measurements"]
    stopped = False
    for batch in (1, 8):
        child = _inference_child(args, output, batch, formal=True, label="common")
        result = _execute(child, record, persist)
        if child["outcome"] == "budget_refused":
            record["decisions"].append(dict(kind="observed_budget_stop", identity=child["identity"]))
            stopped = True
            if batch == 8 and points:
                fallback = _inference_child(args, output, 4, formal=True, label="budget-fallback")
                record["decisions"].append(dict(kind="common_budget_fallback", refused_identity=child["identity"],
                    fallback_identity=fallback["identity"], rule="one formal B4 after sampled B8 budget refusal; no larger batches afterwards"))
                result = _execute(fallback, record, persist)
                if fallback["outcome"] == "completed":
                    points.append(_review(fallback, persist, lambda: inference_point(result, fallback)))
                elif fallback["outcome"] != "budget_refused":
                    return "failed"
            break
        if child["outcome"] != "completed":
            return "failed"
        points.append(_review(child, persist, lambda: inference_point(result, child)))
        persist()
    for index in range(MAX_SCREENS if not stopped else 0):
        batch, projections = next_screen(points)
        record["decisions"].append(dict(kind="screen_selection", measurement_opportunity=index+1,
                                        predictions=projections, selected_batch=batch))
        persist()
        if batch is None:
            record["stop_reason"] = "no larger ladder point with projected budget fit"
            break
        previous = max(points, key=lambda point: point["batch"])
        child = _inference_child(args, output, batch, formal=False, label="screen")
        result = _execute(child, record, persist)
        if child["outcome"] == "budget_refused":
            record["decisions"].append(dict(kind="observed_budget_stop", identity=child["identity"]))
            record["stop_reason"] = "observed sampled budget refusal"
            stopped = True
            break
        if child["outcome"] != "completed":
            return "failed"
        point = _review(child, persist, lambda: inference_point(result, child))
        points.append(point)
        gain = point["stable_decode_positions_per_second"]/previous["stable_decode_positions_per_second"]-1
        record["decisions"].append(dict(kind="measured_gain", identity=point["identity"], previous=previous["identity"],
                                        relative_gain=gain, stop_below=.03))
        persist()
        if gain < .03:
            record["stop_reason"] = "less than 3% measured gain over previous batch; finite screen stop"
            break
    else:
        record["stop_reason"] = "three screening measurements exhausted" if not stopped else "observed budget refusal"
    best = select_best(points)
    record["best_common_or_screen"] = None if best is None else best["identity"]
    if best is not None and not best["formal"]:
        child = _inference_child(args, output, best["batch"], formal=True, label="confirm")
        result = _execute(child, record, persist)
        if child["outcome"] == "budget_refused":
            stopped = True
            record["decisions"].append(dict(kind="confirmation_budget_refused", screen_identity=best["identity"], identity=child["identity"]))
        elif child["outcome"] != "completed":
            return "failed"
        else:
            point = _review(child, persist, lambda: inference_point(result, child))
            point["confirms_screen_identity"] = best["identity"]
            points.append(point)
    formal = select_best(points, formal_only=True)
    record["best_formal_measured"] = None if formal is None else formal["identity"]
    persist()
    if formal is None:
        return "failed_no_formal_point"
    return "completed_with_budget_stop" if stopped else "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "inference"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-batch", type=int, choices=(4, 8), default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    args.config, output = args.config.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error("output directory must be a new identity")
    configuration = json.loads(args.config.read_text())
    output.mkdir(parents=True)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(schema="tide-memory-candidate-suite-v1", source=source, state="planned", started=_now(),
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  config_file=dict(path=str(args.config), sha256=file_hash(args.config)), config=configuration,
                  cwd=os.getcwd(), executable=sys.executable, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
                  scope="finite one-candidate FP32 study; completion is not model-quality or strict numerical-equivalence qualification",
                  policy=dict(child_timeout_seconds=CHILD_TIMEOUT, child_sampled_rss_gib=96, allocated_gib=ALLOCATED_GIB,
                              inference_reserved_gib=RESERVED_GIB, common_batches=[1, 8], screen_ladder=list(SCREEN_BATCHES),
                              common_budget_fallback="only a sampled B8 refusal after successful B1 permits one formal B4; no further screening",
                              max_screen_measurements=MAX_SCREENS, minimum_relative_gain=.03,
                              selection="highest observed median stable-decode throughput; formal confirmation kept distinct; finite measured best, not global optimum",
                              training="20 continuous updates, same-batch fresh-process restored update, two continuing memory-ledger segments; B8 may fall back to B4 only after explicit sampled device budget refusal"),
                  children=[], decisions=[], measurements=[])
    manifest = output/"suite.json"
    persist = lambda: atomic(manifest, record)
    if args.plan_only:
        record["planned_children"] = (_train_children(args, output, args.train_batch) if args.phase == "train" else
                                      [_inference_child(args, output, batch, formal=True, label="common") for batch in (1, 8)])
        persist()
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    record["state"] = "running"
    persist()
    try:
        enable_subreaper()
        record["descendant_supervision"] = "Linux child subreaper; run_child PID/start tracking and per-child terminal identity review"
        record["state"] = (run_train if args.phase == "train" else run_inference)(args, output, record, persist)
    except KeyboardInterrupt as error:
        record.update(state="cancelled", error=repr(error))
    except Exception as error:
        record.update(state="failed", error=repr(error))
        import traceback
        traceback.print_exc()
    finally:
        record.update(ended=_now(), all_executed_children_successful=all(row.get("outcome") == "completed" for row in record["children"]))
        persist()
    print(json.dumps(dict(output=str(manifest), state=record["state"], children=len(record["children"]))))
    return int(not record["state"].startswith("completed"))


if __name__ == "__main__":
    raise SystemExit(main())
