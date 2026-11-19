#!/usr/bin/env python3
"""Read four frozen breakdown jobs and atomically publish a summary snapshot.

This script never launches work, imports torch, modifies raw observations, or
claims systemd/descendant terminal verification. Missing/running/failed records
are retained. The final lifecycle review belongs to the parent release audit.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.job import atomic

SOURCE = "7196fcabd7ab3b401c1ebdb6107442450a99996c"
JOBS = ("infer-kv128", "infer-tf32", "train-kv128", "train-tf32")
COMPLETE = {"completed", "completed_finite"}
INPUTS, ERRORS = {}, []


def path_string(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT/path
    identity = dict(path=path_string(path), state="missing")
    data = {}
    try:
        raw = path.read_bytes()
        identity.update(state="available", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        data = json.loads(raw)
    except FileNotFoundError:
        pass
    except Exception as error:
        identity.update(state="read_error", error=repr(error))
        ERRORS.append(identity)
    INPUTS[identity["path"]] = identity
    return data, identity


def pick(data, *keys):
    return {key: data[key] for key in keys if key in data}


def same_known(first, second):
    return None if first is None or second is None else first == second


def median(values):
    return statistics.median(values) if values and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) else None


def metrics(rows, getters):
    result = {}
    for key, getter in getters.items():
        values = [getter(row) for row in rows]
        finite = bool(values) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        result[key] = dict(mean=statistics.fmean(values) if finite else None, median=median(values), values=values)
    return result


def peak(memories):
    devices = {device for memory in memories for device in memory}
    return {device: {kind: max((memory.get(device, {}).get("peak_"+kind+"_bytes", 0) for memory in memories), default=0)
                     for kind in ("allocated", "reserved")} for device in sorted(devices)}


def shape(data):
    spec = data.get("spec", {})
    if not spec:
        return None
    if data.get("subject") == "transformer" or spec.get("family") == "transformer":
        return dict(kind="Transformer", **pick(spec, "d", "layers", "heads", "head_dim", "ffn_width", "window", "chunk", "backend", "contiguous_qkv"),
                    parameters=data.get("parameters"))
    width = spec["d"]
    return dict(kind="Tide", topology=spec["topology"], content_width=width, summary_width=width//4,
        kv_total_qkv_width=spec.get("kv_width") or width, kv_heads=(spec.get("kv_width") or width)//32,
        kv_head_dim=32, ffn_width=spec.get("ffn_width") or 4*width,
        **pick(spec, "memory", "sharing", "memory_sharing", "region_batch_size", "checkpoint_reception", "aggregation"),
        parameters=data.get("parameters"), label_scope="kv128 names the d768 candidate; the d192 training representative actually has KV total width32")


def inference_normal(data):
    normal = data.get("normal", {})
    rows = normal.get("observations", [])
    batch = data.get("arguments", {}).get("batch")
    schedule = data.get("schedule", {})
    decode = schedule.get("decode_steps")
    result = dict(state=normal.get("state", "pending"), observed_samples=len(rows), requested_samples=data.get("arguments", {}).get("samples"),
        recorded_summary=normal.get("summary"), warmup=normal.get("warmup", []), samples=rows,
        normal_peak_bytes_by_device=peak([row.get("allocator", {}) for row in rows]),
        first_output_support_ready=bool(rows) and all(row.get("first_any_backbone_output_seconds") is not None for row in rows))
    result["computed"] = metrics(rows, {
        "first_backbone_output_seconds": lambda r: r.get("first_any_backbone_output_seconds"),
        "prompt_output_ready_seconds": lambda r: r.get("prompt_output_ready_seconds"),
        "prefill_seconds": lambda r: r.get("prefill", {}).get("seconds"),
        "prefill_positions_per_second": lambda r: r.get("prefill", {}).get("positions_per_second"),
        "first_incremental_call_seconds": lambda r: r.get("first_incremental_call_seconds"),
        "decode_seconds": lambda r: r.get("decode", {}).get("seconds"),
        "decode_positions_per_second": lambda r: r.get("decode", {}).get("positions_per_second"),
        "stable_decode_seconds": lambda r: r.get("stable_decode", {}).get("seconds"),
        "stable_decode_positions_per_second": lambda r: r.get("stable_decode", {}).get("positions_per_second"),
        "stable_decode_median_call_seconds": lambda r: r.get("stable_decode", {}).get("median_call_seconds"),
        "stable_decode_p95_call_seconds": lambda r: r.get("stable_decode", {}).get("p95_call_seconds"),
        "complete_seconds": lambda r: r.get("complete_seconds"),
        "complete_positions_per_second": lambda r: r.get("complete_positions_per_second"),
        "independent_complete_sequences_per_second": lambda r: batch/r["complete_seconds"] if batch and r.get("complete_seconds") else None})
    result["timing_complete"] = normal.get("state") in COMPLETE and len(rows) == result["requested_samples"]
    result["decode_denominators"] = dict(full_decode_calls=decode, stable_decode_calls=None if decode is None else max(0, decode-1),
        full_decode_input_positions=None if batch is None or decode is None else batch*decode,
        stable_decode_input_positions=None if batch is None or decode is None else batch*max(0, decode-1),
        interpretation="normal stable decode omits first incremental call (15 calls here); diagnostic decode captures all16. Compare diagnostic decode with normal full decode, or explicitly select matching call indices")
    result["qualification_scope"] = "normal samples retain timing/state/allocation metadata; finite/output/support identities come from separate diagnostic calls, not a timed output capture"
    return result


def training_normal(data):
    rows = data.get("normal", [])
    args = data.get("arguments", {})
    seconds = sum(row.get("complete_seconds", 0) for row in rows)
    positions = sum(row.get("input_positions", 0) for row in rows)
    keys = ("prefill_seconds", "prefill_positions_per_second", "input_window_seconds", "input_plus_flush_seconds",
            "backward_seconds", "optimizer_seconds", "complete_seconds")
    partition = {"forward_window": [r["input_plus_flush_seconds"] for r in rows],
                 "backward_interval": [r["backward_seconds"] for r in rows],
                 "optimizer": [r["optimizer_seconds"] for r in rows],
                 "unpartitioned": [r["complete_seconds"]-r["input_plus_flush_seconds"]-r["backward_seconds"]-r["optimizer_seconds"] for r in rows]}
    return dict(observed_samples=len(rows), requested_samples=args.get("samples"), samples=rows,
        warmup=data.get("warmup", []), recorded_summary=data.get("normal_summary"),
        normal_peak_bytes_by_device=peak([row.get("allocator", {}) for row in rows]),
        complete_partition=dict(mean_seconds={key: statistics.fmean(values) if values else None for key, values in partition.items()},
            total_seconds={key: sum(values) for key, values in partition.items()},
            definition="four disjoint run_stream clock intervals sum to complete time on the same samples; forward_window includes zero_grad/chunk scheduling/readout/sync; backward_interval can include pending loss; unpartitioned includes intervening Python/loss submission and detach, not a pure device-kernel category"),
        computed=dict(**metrics(rows, {key: lambda r, k=key: r.get(k) for key in keys}),
            input_positions=positions, complete_seconds_total=seconds,
            phase_seconds_total={key: sum(row.get(key, 0) for row in rows) for key in keys if key.endswith("_seconds")},
            aggregate_input_positions_per_second=positions/seconds if seconds else None,
            throughput_definition="sum of actual sample input positions / sum complete seconds; phase means and complete-time mean use these same samples; mean/median of per-sample rates are separate statistics"),
        state_windows=[pick(row, "state_before", "state_after", "continuation", "input_positions", "loss",
            "finite_output", "finite_parameters", "finite_gradients") for row in rows],
        timing_complete=len(rows) == args.get("samples") and bool(data.get("normal_summary")),
        qualification_scope="actual continuing normal updates; finite/continuation checks excluded from timed complete interval but can contribute to sampled allocator peaks",
        phase_limit="legacy run_stream backward interval may include pending concatenation/loss; isolated loss/backward diagnosis belongs to the separate stages pass")


def phase(path):
    parts = path.split("/")
    return parts[1] if parts[0] == "training" and len(parts) > 1 else parts[0]


def stage_report(report):
    rows = report.get("rows", [])
    totals = defaultdict(float)
    for row in rows:
        totals[phase(row["path"])] += row.get("exclusive_seconds", 0)
    return dict(**pick(report, "boundary", "interpretation", "boundary_failures"), rows=rows,
        exclusive_seconds_by_phase=dict(sorted(totals.items())),
        exclusive_seconds_sum=sum(row.get("exclusive_seconds", 0) for row in rows),
        aggregation="sum exclusive scope times only; training root residual remains training; never add inclusive nesting or mix with operator-pass times")


def operator_report(report, source):
    rows = report.get("rows", [])
    totals = defaultdict(lambda: dict(calls=0, seconds=0.))
    for row in rows:
        target = totals[phase(row["path"]), row["operator"]]
        target["calls"] += row["calls"]
        target["seconds"] += row["seconds"]
    return dict(**pick(report, "dispatched_calls", "operator_seconds", "failed_calls", "failures", "boundary", "interpretation"),
        recorded_group_count=len(rows),
        by_phase_operator=[dict(phase=p, operator=op, **values) for (p, op), values in sorted(totals.items())],
        largest_groups=sorted(rows, key=lambda r: r["seconds"], reverse=True)[:60],
        complete_groups=dict(raw_file=source, json_location="diagnostics.operators.report.rows"),
        aggregation="operator pass only; groups retain tensor shapes/strides/dtypes/devices in raw and top groups; sums exclude Python gaps/entry waits and are not normal wall time, stage time or native kernel time")


def diagnostics(data, source):
    result = {}
    for kind in ("stages", "operators"):
        value = data.get("diagnostics", {}).get(kind)
        if value is None:
            result[kind] = dict(state="pending")
            continue
        result[kind] = {key: item for key, item in value.items() if key != "report"}
        if "calls" in value:
            phases = sorted({row["phase"] for row in value["calls"]})
            result[kind]["captured_phase_denominators"] = {p: dict(calls=sum(row["phase"] == p for row in value["calls"]),
                input_positions=sum(row["input_positions"] for row in value["calls"] if row["phase"] == p)) for p in phases}
        result[kind]["report"] = stage_report(value.get("report", {})) if kind == "stages" else operator_report(value.get("report", {}), source)
    return result


def job(name):
    run = ROOT/"artifacts"/f"breakdown-{name}-v1"
    raw, file_id = read(run/"breakdown.json")
    lifecycle = {}
    for kind in ("launch", "status", "queue", "dependency"):
        data, identity = read(run/(kind+".json"))
        lifecycle[kind] = dict(file=identity, **pick(data, "state", "status", "source", "source_path", "unit", "unit_observation", "pid", "child_pid",
            "started", "finished", "exit_code", "return_code", "physical_ids", "logical_devices", "elapsed", "peak_rss_bytes", "descendant_identities", "error",
            "units", "observations", "checked", "deadline_seconds"))
    result = dict(name=name, file=file_id, lifecycle=lifecycle, state=raw.get("state", file_id["state"]),
                  result_complete=raw.get("state") in COMPLETE, live_terminal_verification="separate parent audit, not inferred from result completion")
    if raw:
        result.update(pick(raw, "schema", "last_stage", "source", "subject", "arguments", "schedule", "config", "spec", "parameters", "precision",
            "physical_devices", "resource_identity", "cpu_affinity", "torch_threads", "torch", "torch_npu", "allow_hf32", "initial_parameters",
            "final_parameters", "input_sha256", "input_shape", "shared_input_prefix", "optimizer", "transformer_match", "comparison", "boundary",
            "placement", "construction_allocator", "anchor", "first_measured_update", "replay_checks", "failure_kind", "error", "process_peak_rss_kib"))
        try:
            result["shape"] = shape(raw)
            result["normal"] = inference_normal(raw) if name.startswith("infer") else training_normal(raw)
            result["diagnostics"] = diagnostics(raw, file_id)
            result["expected_source_matches"] = raw.get("source", {}).get("source_commit") == SOURCE
        except Exception as error:
            result["summary_error"] = repr(error)
            ERRORS.append(dict(file=file_id, error=repr(error)))
    return result


def paired(jobs, mode):
    left, right = (next(row for row in jobs if row["name"] == mode+"-"+subject) for subject in ("kv128", "tf32"))
    a, b = left.get("arguments", {}), right.get("arguments", {})
    common = ("batch", "seed", "warmup", "samples", "prefill", "prefill_chunk", "decode") if mode == "infer" else (
        "batch", "seed", "warmup", "samples", "length", "chunk", "optimizer_foreach")
    prefixes = [row.get("shared_input_prefix") for row in (left, right)]
    result = dict(point_names=[left["name"], right["name"]], states=[left["state"], right["state"]],
        actual_shapes=[left.get("shape"), right.get("shape")], shared_settings={key: [a.get(key), b.get(key)] for key in common},
        settings_match=None if any(key not in a or key not in b for key in common) else all(a[key] == b[key] for key in common),
        shared_input_prefix=prefixes, shared_input_prefix_matches=None if any(p is None for p in prefixes) else prefixes[0] == prefixes[1],
        initial_parameter_sha256=[row.get("initial_parameters", {}).get("sha256") for row in (left, right)],
        full_input_sha256=[row.get("input_sha256") for row in (left, right)],
        full_input_shapes=[row.get("input_shape") for row in (left, right)],
        identity_scope="mathematical functions and tensor widths differ; compare shared input prefix, not full input/parameter hash equality",
        scope="9B graph fixed-FFN KV128 vs existing near8.778B Transformer32; total parameters differ" if mode == "infer" else
              "medium d192/KV32 ~129M vs near-total32-layer Transformer; neither9B training nor A-D replacement")
    ready = all(row.get("normal", {}).get("timing_complete") for row in (left, right))
    result["normal_pair_complete"] = ready
    result["complete_job_pair"] = all(row["result_complete"] for row in (left, right))
    if ready:
        if mode == "infer":
            keys = ("prefill_positions_per_second", "stable_decode_positions_per_second", "complete_positions_per_second", "independent_complete_sequences_per_second")
            result["transformer_over_tide_normal_throughput"] = {key: right["normal"]["computed"][key]["median"]/left["normal"]["computed"][key]["median"] for key in keys}
        else:
            result["transformer_over_tide_normal_throughput"] = right["normal"]["computed"]["aggregate_input_positions_per_second"]/left["normal"]["computed"]["aggregate_input_positions_per_second"]
    return result


def differences(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    return {key: dict(previous=a.get(key), current=b.get(key)) for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)}


def prior_records(jobs):
    by_name = {row["name"]: row for row in jobs}
    rows = []
    old, identity = read(ROOT/"artifacts/memory-infer-kv128-v1/results/common-b8.json")
    if old:
        subject = old["subjects"]["tide"]
        converted = dict(old, subject="tide", spec=subject["spec"], parameters=subject["parameters"],
                         normal=dict(state=subject["state"], observations=subject["observations"]))
        current = by_name["infer-kv128"]
        rows.append(dict(label="previous KV128 9B common B8", file=identity, state=old["state"], source=old["source"],
            physical_devices=old["physical_devices"], arguments=old["arguments"], shape=shape(converted), normal=inference_normal(converted),
            initial_parameters=subject["initial_parameters"], input_sha256=subject["input_sha256"],
            current_identity_comparison=dict(initial_parameter_sha256=same_known(subject["initial_parameters"].get("sha256"), current.get("initial_parameters", {}).get("sha256")),
                input_sha256=same_known(subject["input_sha256"], current.get("input_sha256")), specification_differences=differences(subject["spec"], current.get("spec"))),
            boundary="previous source/card and cold->diagnostic->warmup ordering retained; current normal precedes diagnostics. No relabelling as repeated identical timing procedure"))
    evidence, evidence_id = read(ROOT/"docs/evidence/decode-capacity-20261118.json")
    if evidence:
        matches = [row for row in evidence["records"]["summary"]["runs"] if row["name"] == "cap-9b-tf32-c1-b8-screen-v1"]
        for record in matches:
            old, current = record["result"], by_name["infer-tf32"]
            converted = dict(old, subject="transformer", arguments=dict(batch=8, samples=len(old["observations"])),
                             normal=dict(state=old["state"], observations=old["observations"]))
            rows.append(dict(label="previous Transformer32 B8 two-sample screen", evidence=evidence_id, raw_files=record["files"],
                name=record["name"], phase=record["phase"], state=old["state"], source=old["source"],
                physical_devices=old.get("physical_devices"), shape=shape(converted), normal=inference_normal(converted),
                initial_parameters=old.get("initial_parameters"), input_sha256=old.get("input_sha256"),
                current_identity_comparison=dict(initial_parameter_sha256=same_known(old.get("initial_parameters", {}).get("sha256"), current.get("initial_parameters", {}).get("sha256")),
                    input_sha256=same_known(old.get("input_sha256"), current.get("input_sha256")), specification_differences=differences(old.get("spec"), current.get("spec"))),
                boundary="reused published auxiliary baseline evidence; two-sample screen, not a new five-sample confirmation; original source/hardware identity retained"))
    old, identity = read(ROOT/"artifacts/memory-train-kv128-b16-v1/continuous.json")
    if old:
        selected = old["observations"][2:20]
        first_five = old["observations"][2:7]
        current = by_name["train-kv128"]
        seconds = sum(row["complete_seconds"] for row in selected)
        rows.append(dict(label="previous medium KV32 B16 continuous training", file=identity, state=old["state"], source=old["source"],
            harness_source=old.get("harness_source"), physical_devices=old["physical_devices"], arguments=old["arguments"], spec=old["spec"], parameters=old["parameters"],
            initial_parameter_sha256=old["initial_parameters_sha256"], input_sha256=old["input_sha256"],
            stable_updates=[3, 20], stable_input_positions=sum(row["input_positions"] for row in selected), stable_complete_seconds=seconds,
            stable_input_positions_per_second=sum(row["input_positions"] for row in selected)/seconds,
            original_update3to7=[pick(row, "index", "state_position", "input_positions", "loss", "complete_seconds", "prefill_seconds", "backward_seconds", "optimizer_seconds", "device_memory") for row in first_five],
            current_identity_comparison=dict(initial_parameter_sha256=same_known(old["initial_parameters_sha256"], current.get("initial_parameters", {}).get("sha256")),
                input_sha256=same_known(old["input_sha256"], current.get("input_sha256")), specification_differences=differences(old["spec"], current.get("spec"))),
            boundary="previous inference warmup then20 actual updates; published rate updates3-20. Current two training warmups then five normal updates correspond to update indices3-7, but use a new source/card/timing session and retain its own measurement identity"))
    old, identity = read(ROOT/"artifacts/memory-ad-kv128-v1/results/000-small-kv128-fp32-all-modes.json")
    if old:
        train = [row for row in old["observations"] if row["mode"] == "train"]
        tide_rate = next(row["input_positions_per_second"] for row in train if row["subject"] == "tide")
        rows.append(dict(label="previous small KV32 A-D train", file=identity, state=old["state"], source=old["source"],
            physical_devices=old["physical_devices"], arguments=old["arguments"], topology=old["topology"], config=old["config"],
            points=[dict(**pick(row, "subject", "spec", "parameters", "initial_parameter_sha256", "input_sha256", "median_seconds", "input_positions_per_second"),
                         throughput_over_tide=row["input_positions_per_second"]/tide_rate) for row in train],
            boundary="small17-node ~10M, B8 L128 zero-state prefill; A/B17 blocks and C/D5 blocks. This is neither medium continuing Transformer32 nor9B training evidence"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"summary.json")
    args = parser.parse_args()
    jobs = [job(name) for name in JOBS]
    result = dict(schema="tide-kv128-transformer-breakdown-summary-v1", generated_at=datetime.now(timezone.utc).isoformat(),
        generator=dict(path=path_string(__file__), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), expected_source=SOURCE,
        jobs=jobs, paired_comparisons={mode: paired(jobs, mode) for mode in ("infer", "train")}, original_points=prior_records(jobs),
        units=dict(latency="seconds", throughput="independent input positions/s; whole request sequences/s separately labelled", memory="bytes; divide by2**30 for GiB"),
        scope="read-only rolling snapshot; atomic raw files are read individually, not as a cross-file transaction; completion is result state, not descendant termination",
        boundaries=["Normal, stages and operators are three different observations. Never add their times or convert intrusive stage/operator fractions into normal wall-time shares.",
            "Inference uses the9B graph and a fixed32-layer8.778B reference; training uses the129M d192/KV32 representative and a separate32-layer near-total shape.",
            "Inference first-output availability is populated after a separate diagnostic support witness; before that completes first-output latency remains unknown, even if normal throughput samples are already complete.",
            "Normal inference stable decode excludes its first incremental call (15 calls); diagnostic decode includes all16. The normal full-decode fields retain the matching16-call boundary.",
            "No task head, token sampler or quality evaluation; first backbone output is not full language-model TTFT.",
            "Normal training legacy backward can include pending loss/concatenation; diagnostic loss/backward have distinct synchronized boundaries.",
            "Training diagnostics compare the first real normal update after restoring parameters/gradients/AdamW/RNG/private state. completed_finite is execution status: numerical_state and threshold/support details must also be read.",
            "Raw identities retain source, physical device, initial parameters, actual input shapes and hashes; full input hashes differ across widths while paired feature prefixes should match.",
            "Parent owns resource admission and terminal process/cgroup verification. This script does not call systemctl, devices or experiment launchers."])
    result["inputs"] = list(INPUTS.values())
    result["summary_errors"] = ERRORS
    result["observed_states"] = {row["name"]: row["state"] for row in jobs}
    result["state"] = "snapshot_with_errors" if ERRORS else "complete_snapshot" if all(row["result_complete"] for row in jobs) else "partial_snapshot"
    atomic(args.output, result)
    print(json.dumps(dict(output=path_string(args.output), state=result["state"], observed_states=result["observed_states"], summary_errors=len(ERRORS),
        normal_complete={row["name"]: row.get("normal", {}).get("timing_complete", False) for row in jobs},
        paired_prefix_matches={key: value["shared_input_prefix_matches"] for key, value in result["paired_comparisons"].items()})))
    return int(bool(ERRORS))


if __name__ == "__main__":
    raise SystemExit(main())
