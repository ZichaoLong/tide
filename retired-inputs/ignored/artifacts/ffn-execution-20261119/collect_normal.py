#!/usr/bin/env python3
"""Aggregate saved normal FFN timing records; never run a model or device probe."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


SUITES = (
    "ffn-medium-cpu-timing-v1", "ffn-large-cpu-timing-v1",
    "ffn-large-npu-timing-v1", "ffn-long-npu-timing-v2",
    "ffn-cpu16-calibration-v1", "ffn-cpu32-calibration-v1",
    "ffn-medium-npu-timing-v1",
)
LABELS = {"baseline": "baseline", "weight_access": "W", "packing": "P", "combined": "WP"}


def read(path, root):
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path.relative_to(root)), bytes=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest())


def atomic(path, value):
    raw = value.encode() if isinstance(value, str) else (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    fd, staged = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".staged", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)


def stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite timing statistic in retained evidence")
    return dict(n=len(values), min=min(values), median=statistics.median(values),
                max=max(values), mean=statistics.mean(values), sum=sum(values))


def rate(units, seconds):
    return units / seconds if seconds > 0 else None


def label(data):
    return "Transformer32" if data["subject"] == "transformer" else LABELS[data["arguments"]["variant"]]


def planned_arguments(launch):
    command = launch.get("arguments", {}).get("command", [])
    fields = {"variants": [], "include_transformer": False, "fixed_large_transformer": False}
    numbers = {"batch", "length", "chunk", "decode", "warmup", "samples", "threads", "repeats"}
    for i, token in enumerate(command):
        if token in {"--include-transformer", "--fixed-large-transformer", "--observe-first"}:
            fields[token[2:].replace("-", "_")] = True
        elif token == "--variants":
            for item in command[i + 1:]:
                if item.startswith("--"):
                    break
                fields["variants"].append(item)
        elif token.startswith("--") and i + 1 < len(command):
            key = token[2:].replace("-", "_")
            if key in numbers:
                fields[key] = int(command[i + 1])
            elif key in {"mode", "device", "config"}:
                fields[key] = command[i + 1]
    return fields


def sample_extract(sample, mode, batch):
    complete = sample["complete_seconds"]
    if mode == "train":
        exclusive = dict(forward_window_including_preparation_seconds=sample["input_plus_flush_seconds"],
                         backward_seconds=sample["backward_seconds"], optimizer_seconds=sample["optimizer_seconds"])
        auxiliary = {k: sample.get(k) for k in ("prefill_seconds", "input_window_seconds", "flush_seconds",
                     "first_backbone_output_seconds", "phase_times")}
        auxiliary["decode_call_seconds"] = sample.get("decode_seconds", [])
        prefill_seconds = sample["prefill_seconds"]
        # continuous.run_stream records calls.inputs = stop-start, i.e. one
        # sequence's length, not batch positions. Convert exactly once here.
        prefill_positions = batch * sum(c["inputs"] for c in sample["calls"] if c["phase"] == "prefill")
        first = sample.get("first_backbone_output_seconds")
        stable_seconds = stable_positions = 0
    else:
        exclusive = {p + "_seconds": sum(c["seconds"] for c in sample["calls"] if c["phase"] == p)
                     for p in ("prefill", "decode", "stream", "drain")}
        auxiliary = {k: sample.get(k) for k in ("prefill", "decode", "stable_decode", "chunked_stream",
                     "first_incremental_call_seconds", "first_any_backbone_output_seconds", "prompt_ingested_seconds",
                     "prompt_output_ready_seconds", "input_window_seconds", "stable_decode_rule")}
        auxiliary["forward_readout_sync_sum_seconds"] = sum(c["forward_readout_sync_seconds"] for c in sample["calls"])
        prefill_seconds = sample["prefill"]["seconds"]
        prefill_positions = sample["prefill"]["input_positions"]
        stable_seconds = sample["stable_decode"]["seconds"]
        stable_positions = sample["stable_decode"]["input_positions"]
        first = sample.get("first_any_backbone_output_seconds")
    exclusive["other_complete_boundary_seconds_by_subtraction"] = complete - sum(exclusive.values())
    return dict(complete_seconds=complete, input_positions=sample["input_positions"],
                output_slots=sample.get("output_slots"), first_backbone_output_seconds=first,
                prefill_seconds=prefill_seconds, prefill_input_positions=prefill_positions,
                stable_decode_seconds=stable_seconds, stable_decode_input_positions=stable_positions,
                exclusive_phase_seconds=exclusive, nonadditive_phase_subitems=auxiliary,
                state_before=sample.get("state_before"), state_after=sample.get("state_after", sample.get("final_state_position")),
                loss=sample.get("loss"), finite_checks={k: sample[k] for k in
                    ("finite_output", "finite_parameters", "finite_gradients") if k in sample},
                continuation=sample.get("continuation"), allocator=sample.get("allocator", {}),
                rss=sample.get("rss"), state_memory=sample.get("state_memory"))


def repeat_summary(data, suite_row, file):
    args, mode = data["arguments"], data["arguments"]["mode"]
    rows = [sample_extract(s, mode, args["batch"]) for s in data["normal"]]
    n, batch = len(rows), args["batch"]
    complete = sum(s["complete_seconds"] for s in rows)
    positions = sum(s["input_positions"] for s in rows)
    prefill_positions = sum(s["prefill_input_positions"] for s in rows)
    prefill_seconds = sum(s["prefill_seconds"] for s in rows)
    stable_positions = sum(s["stable_decode_input_positions"] for s in rows)
    stable_seconds = sum(s["stable_decode_seconds"] for s in rows)
    devices = sorted({device for s in rows for device in s["allocator"]})
    memory = {device: {key: max(s["allocator"].get(device, {}).get(key, 0) for s in rows)
                      for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")}
              for device in devices}
    phases = {key: stats([s["exclusive_phase_seconds"][key] for s in rows]) for key in rows[0]["exclusive_phase_seconds"]}
    accounting_error = max(abs(sum(s["exclusive_phase_seconds"].values()) - s["complete_seconds"]) for s in rows)
    result = dict(identity=suite_row["identity"], repeat=args["repeat"], subject=data["subject"],
        variant=args["variant"], label=label(data), file=file, source=data["source"],
        mode=mode, batch=batch, normal_samples=n, warmup_samples=len(data["warmup"]),
        total_input_positions=positions, total_independent_batch_units=n * batch,
        total_complete_seconds=complete, complete_positions_per_second=rate(positions, complete),
        independent_batch_units_per_second=rate(n * batch, complete),
        prefill_input_positions=prefill_positions, prefill_seconds=prefill_seconds,
        prefill_positions_per_second=rate(prefill_positions, prefill_seconds),
        stable_decode_input_positions=stable_positions, stable_decode_seconds=stable_seconds,
        stable_decode_positions_per_second=rate(stable_positions, stable_seconds),
        first_backbone_output_seconds=stats([s["first_backbone_output_seconds"] for s in rows]),
        mean_complete_seconds=complete / n, exclusive_phase_seconds=phases,
        phase_mean_additivity=dict(mean_phase_sum_seconds=sum(v["mean"] for v in phases.values()),
            mean_complete_seconds=complete / n, max_per_sample_accounting_abs_error_seconds=accounting_error,
            smallest_unattributed_remainder_seconds=min(s["exclusive_phase_seconds"]["other_complete_boundary_seconds_by_subtraction"] for s in rows),
            note="Means of non-overlapping boundaries plus explicit residual add to mean complete time. Phase medians and nested subitems are not added."),
        denominator=dict(complete=f"{positions} input positions / {n} normal complete times summed",
            independent=f"{n * batch} independent batch stream windows (train) or requests (inference) / the same summed complete time",
            prefill=f"{prefill_positions} prefill positions / summed prefill call seconds",
            stable_decode=None if not stable_positions else f"{stable_positions} positions / summed decode seconds after omitting each request's first decode call",
            scope="All warmup times, construction, hashes, observers and IO excluded from normal timing.",
            training_prefill_units="calls.inputs is per-sequence stop-start; collector multiplies by B once at sample extraction. Full training throughput uses sample.input_positions (2048 at B16/L128) over complete_seconds; prefill-only throughput is auxiliary."),
        normal_allocator_max_by_device=memory or None,
        cpu_allocator_scope="CPU allocator object is empty/unavailable; RSS is host process lifetime high-water, never HBM." if not devices else None,
        normal_end_process_lifetime_peak_rss_bytes=1024 * data.get("normal_end_rss", {}).get("process_lifetime_peak_rss_kib", 0),
        final_process_lifetime_peak_rss_including_diagnostics_bytes=1024 * data.get("final_rss", {}).get("process_lifetime_peak_rss_kib", 0),
        child_sampled_process_tree_peak_rss_including_diagnostics_bytes=suite_row.get("sampled_process_tree_peak_rss_bytes"),
        initial_parameters_sha256=data["initial_parameters"]["sha256"], input_sha256=data["input_sha256"],
        normal_final_parameters_sha256=data.get("normal_final_parameters", {}).get("sha256"),
        input_shape=data["input_shape"], shared_input_prefix=data.get("shared_input_prefix"),
        parameters=data["parameters"], spec=data["spec"], cpu_affinity=data["cpu_affinity"],
        torch_threads=data["torch_threads"], physical_devices=data.get("physical_devices"),
        ffn_weight_backend=data.get("ffn_weight_backend"),
        resource_identity=data.get("resource_identity"), precision=data["precision"],
        recorded_normal_summary=data.get("normal_summary"), recorded_boundary=data.get("boundary"),
        host_load_before=data.get("host_load_before"), host_load_after=data.get("host_load_after"),
        warmed_anchor=data.get("anchor"), raw_normal_samples=rows)
    return result


def work_summary(diagnostics):
    entry = diagnostics.get("work")
    if not entry:
        return None
    work = entry.get("actual_work", entry)
    if not work:
        return None
    return {k: work.get(k) for k in ("input_positions", "effective_touched", "work", "query_packing", "ffn_packing",
                                    "node_matrices", "matrix_cost_definition", "definition", "deduplication")}


def ledger_summary(ledger):
    return {phase: dict(unique_storage_union_bytes=value["unique_storage_union_bytes"],
                exclusive_category_bytes={k:v["exclusive_unique_storage_bytes"] for k,v in value["categories"].items()},
                allocator=value.get("allocator"),
                live_allocator_minus_visible_union_bytes=value.get("live_allocator_minus_visible_union_bytes"),
                process_lifetime_peak_rss_kib=value.get("process_lifetime_peak_rss_kib"))
            for phase,value in ledger.get("phases", {}).items()}


def group_summary(repeats, expected):
    grouped = defaultdict(list)
    baselines = {r["repeat"]:r for r in repeats if r["subject"] == "tide" and r["variant"] == "baseline"}
    pairs = []
    for row in repeats:
        grouped[row["label"]].append(row)
        baseline = baselines.get(row["repeat"])
        if baseline:
            pair = dict(repeat=row["repeat"], label=row["label"], subject=row["subject"],
                        reference_identity=baseline["identity"], candidate_identity=row["identity"],
                        complete_throughput_ratio=row["complete_positions_per_second"] / baseline["complete_positions_per_second"],
                        mean_complete_time_ratio=row["mean_complete_seconds"] / baseline["mean_complete_seconds"],
                        prefill_throughput_ratio=row["prefill_positions_per_second"] / baseline["prefill_positions_per_second"],
                        same_total_input_positions=row["total_input_positions"] == baseline["total_input_positions"],
                        shared_input_prefix_matches=row["shared_input_prefix"] == baseline["shared_input_prefix"])
            if row["stable_decode_positions_per_second"] and baseline["stable_decode_positions_per_second"]:
                pair["stable_decode_throughput_ratio"] = row["stable_decode_positions_per_second"] / baseline["stable_decode_positions_per_second"]
            pairs.append(pair)
    summary = []
    for name, values in grouped.items():
        values.sort(key=lambda r:r["repeat"])
        paired = [p for p in pairs if p["label"] == name]
        fields = ("complete_positions_per_second", "independent_batch_units_per_second", "prefill_positions_per_second",
                  "stable_decode_positions_per_second", "mean_complete_seconds")
        summary.append(dict(label=name, subject=values[0]["subject"], variant=values[0]["variant"],
            parameters=values[0]["parameters"], completed_repeats=len(values), expected_repeats=expected,
            all_expected_repeats_complete=len(values) == expected,
            per_repeat=[{k:r[k] for k in ("repeat", *fields)} for r in values],
            across_repeat={k:stats([r[k] for r in values]) for k in fields},
            first_backbone_output_repeat_medians=stats([r["first_backbone_output_seconds"]["median"] for r in values if r["first_backbone_output_seconds"]]),
            paired_complete_throughput_ratio=stats([p["complete_throughput_ratio"] for p in paired]),
            paired_prefill_throughput_ratio=stats([p["prefill_throughput_ratio"] for p in paired]),
            paired_stable_decode_throughput_ratio=stats([p.get("stable_decode_throughput_ratio") for p in paired]),
            pooled_complete_positions_per_second=rate(sum(r["total_input_positions"] for r in values), sum(r["total_complete_seconds"] for r in values)),
            normal_allocator_max_by_device={device:{key:max(r["normal_allocator_max_by_device"].get(device, {}).get(key, 0) for r in values if r["normal_allocator_max_by_device"])
                for key in ("peak_allocated_bytes", "peak_reserved_bytes")} for device in
                sorted({d for r in values for d in (r["normal_allocator_max_by_device"] or {})})} or None,
            max_normal_end_process_lifetime_peak_rss_bytes=max(r["normal_end_process_lifetime_peak_rss_bytes"] for r in values)))
    return summary, pairs


def collect_suite(root, name, identities):
    path = root / "artifacts" / name
    if not (path / "launch.json").exists():
        return dict(name=name, state="pending", reason="launch record absent")
    launch, launch_file = read(path / "launch.json", root)
    status, status_file = read(path / "status.json", root)
    args = planned_arguments(launch)
    answer = dict(name=name, state="pending", launch=launch_file, status=status_file,
        outer_state=status.get("state"), source_commit=launch["source"], arguments=args,
        worker_requested_affinity=launch["arguments"].get("affinity"),
        parent_service_sampled_peak_rss_bytes=status.get("peak_rss_bytes"),
        lifecycle_scope="Saved state only; terminal unit/descendant audit is separate resource-audit.json.",
        normal_repeats=[], pending_or_failed=[], r0_diagnostics={},
        expected_children=args.get("repeats", 0) * (len(args.get("variants", [])) + int(args.get("include_transformer", False))))
    if not (path / "results/suite.json").exists():
        answer["reason"] = "suite output not yet created (possibly dependency/admission wait)"
        answer["remaining_children_count"] = answer["expected_children"]
        return answer
    suite, suite_file = read(path / "results/suite.json", root)
    args = suite["arguments"]
    answer.update(arguments=args, suite=suite_file, suite_state=suite["state"],
                  timing=suite.get("timing"), physical_devices=suite.get("physical_devices"),
                  expected_children=len(suite["rows"]), expected_repeats=args["repeats"],
                  actual_child_order=[r["identity"] for r in suite["rows"]])
    for row in suite["rows"]:
        raw_path = path / "results" / (row["identity"] + ".json")
        if not raw_path.exists():
            answer["pending_or_failed"].append(dict(identity=row["identity"], child_state=row["state"], result_state="missing"))
            continue
        data, file = read(raw_path, root)
        a = data["arguments"]
        if data.get("initial_parameters", {}).get("sha256") and data.get("input_sha256"):
            identities.append(dict(suite=name, identity=row["identity"], result_state=data["state"],
                subject=data["subject"], variant=a["variant"], mode=a["mode"], device=a["device"],
                scale="large" if args.get("fixed_large_transformer") else "medium",
                batch=a["batch"], length=a["length"], chunk=a["chunk"], decode=a["decode"] if a["mode"] == "inference" else 0,
                seed=a["seed"], input_shape=data["input_shape"], parameters=data["parameters"],
                initial_parameters_sha256=data["initial_parameters"]["sha256"], input_sha256=data["input_sha256"],
                shared_input_prefix=data.get("shared_input_prefix"), tide_content_sha256=data["source"]["tide_content_sha256"],
                spec=data["spec"], file=file))
        ready = (row["state"] == "completed" and row.get("inner_state") == "completed_finite" and data["state"] == "completed_finite")
        ready = ready and len(data.get("normal", [])) == a["samples"] and len(data.get("warmup", [])) == a["warmup"]
        if ready and row.get("result_sha256") != file["sha256"]:
            raise ValueError(f"Completed child result hash changed: {name}/{row['identity']}")
        if not ready:
            observed = datetime.now(timezone.utc)
            elapsed = ((observed - datetime.fromisoformat(row["started"])).total_seconds()
                       if row.get("started") and row["state"] == "running" else row.get("elapsed_seconds"))
            answer["pending_or_failed"].append(dict(identity=row["identity"], child_state=row["state"],
                result_state=data["state"], recorded_normal_samples=len(data.get("normal", [])),
                recorded_warmup_samples=len(data.get("warmup", [])),
                last_saved_stage=data.get("last_stage"), saved_diagnostic_keys=list(data.get("diagnostics", {})),
                saved_normal_complete_seconds=[x["complete_seconds"] for x in data.get("normal", [])],
                child_started=row.get("started"), child_timeout_seconds=row.get("timeout_seconds"),
                snapshot_observed_at=observed.isoformat(), child_wall_elapsed_seconds_estimate=elapsed,
                seconds_to_declared_child_timeout_estimate=(row["timeout_seconds"] - elapsed) if elapsed is not None and row.get("timeout_seconds") else None,
                progress_scope="Last saved JSON checkpoint only; post-normal hashes/availability/work/ledger may run without another persist until completion. It does not locate the currently executing operator. Pending samples are not aggregated as a completed repeat.",
                file=file))
            continue
        result = repeat_summary(data, row, file)
        answer["normal_repeats"].append(result)
        if a["repeat"] == 0:
            diagnostic = data.get("diagnostics", {})
            answer["r0_diagnostics"][result["label"]] = dict(file=file,
                requested_work=a.get("observe_work"), requested_ledger=a.get("memory_ledger"),
                scope="Separate untimed pass; training restores each candidate's warmed first-normal anchor. Not the trace of every timed update.",
                work=diagnostic.get("work"), work_summary=work_summary(diagnostic),
                memory_ledger=diagnostic.get("memory_ledger"),
                ledger_phase_summary=ledger_summary(diagnostic.get("memory_ledger", {})),
                availability=diagnostic.get("availability"))
    complete = (suite["state"] == "completed" and not answer["pending_or_failed"] and status.get("state") == "passed")
    answer["state"] = "complete" if complete else "terminal_incomplete" if status.get("state") in {"failed", "terminated"} else "pending"
    answer["normal_summary"], answer["same_repeat_pairs"] = group_summary(answer["normal_repeats"], args["repeats"])
    answer["aggregation_scope"] = "Complete suite" if complete else "Provisional completed children only; suite still incomplete. No missing repeat imputed."
    answer["calibration_only"] = "calibration" in name
    answer["remaining_children_count"] = answer["expected_children"] - len(answer["normal_repeats"])
    return answer


def cross_identity(identities):
    grouped = defaultdict(list)
    fields = ("scale", "subject", "variant", "mode", "batch", "length", "chunk", "decode", "seed", "parameters")
    for row in identities:
        grouped[tuple(row[f] for f in fields)].append(row)
    result = []
    for key, rows in grouped.items():
        devices = {r["device"].split(":")[0] for r in rows}
        if not {"cpu", "npu"}.issubset(devices):
            continue
        result.append(dict(fixture=dict(zip(fields, key)),
            inputs_equal=len({r["input_sha256"] for r in rows}) == 1,
            initial_parameters_equal=len({r["initial_parameters_sha256"] for r in rows}) == 1,
            input_shapes_equal=len({tuple(r["input_shape"]) for r in rows}) == 1,
            specs_equal=len({json.dumps(r["spec"], sort_keys=True) for r in rows}) == 1,
            production_source_equal=len({r["tide_content_sha256"] for r in rows}) == 1,
            members=rows,
            scope="Same initial tensor hashes only. Does not equate CPU/NPU outputs, warmed trajectories, hardware, thread counts, affinities or timing sample counts."))
    return result


def qualification_refs(root):
    paths = ["artifacts/ffn-execution-20261119/d768-diagnosis-review.json",
             "artifacts/ffn-execution-20261119/small-npu-review.json",
             "artifacts/ffn-execution-20261119/cpu-combined-review.json",
             "artifacts/ffn-medium-cpu-qualification-v1/qualification/result.json",
             "artifacts/ffn-medium-npu-qualification-v1/qualification/result.json"]
    result = []
    for name in paths:
        if not (root / name).exists():
            continue
        data, file = read(root / name, root)
        item = dict(file=file, state=data.get("state"), qualified=data.get("qualified"),
                    tolerance=data.get("tolerance", data.get("threshold", data.get("thresholds"))), error=data.get("error"))
        item["candidate_states"] = [{k:c.get(k) for k in ("name", "state", "counts")} for c in data.get("candidates", [])]
        if "retained_original_qualification_failures" in data:
            item["retained_original_qualification_failures"] = data["retained_original_qualification_failures"]
        result.append(item)
    return result


def calibration_comparison(suites):
    selected = {s["name"]:s for s in suites}
    one = selected["ffn-cpu16-calibration-v1"]
    two = selected["ffn-cpu32-calibration-v1"]
    result = []
    for left in one.get("normal_repeats", []):
        right = next((r for r in two.get("normal_repeats", []) if r["label"] == left["label"]), None)
        if right is None:
            continue
        result.append(dict(label=left["label"], threads16_positions_per_second=left["complete_positions_per_second"],
            threads32_positions_per_second=right["complete_positions_per_second"],
            throughput_ratio_32_over_16=right["complete_positions_per_second"] / left["complete_positions_per_second"],
            initial_parameters_match=left["initial_parameters_sha256"] == right["initial_parameters_sha256"],
            input_match=left["input_sha256"] == right["input_sha256"],
            affinity16=left["cpu_affinity"], affinity32=right["cpu_affinity"],
            scope="One repeat of two warmup plus five normal continuing updates per thread setting; differing host observations and trajectories remain. Calibration only, not a global thread optimum or a causal scaling estimate."))
    return result


def locality_reference(root):
    path = root / "artifacts/ffn-execution-20261119/cpu-locality-observation.json"
    if not path.exists():
        return None
    data, file = read(path, root)
    observations = []
    for name, values in data.get("processes", {}).items():
        nodes = values.get("resident_kib_by_numa", {})
        total = sum(nodes.values())
        biggest = max(nodes, key=nodes.get) if nodes else None
        observations.append(dict(job=name, recorded_affinity=values.get("affinity"),
            largest_resident_node=biggest, largest_node_resident_fraction=nodes[biggest] / total if total else None,
            scope=values.get("boundary")))
    return dict(file=file, observed=data.get("observed"), cpu_frequency_snapshot=data.get("cpus"),
        saved_process_locality=observations,
        interpretation="This later snapshot records approximately 3 GHz/performance governor and >99.5% pages on each observed worker's local NUMA node. It does not explain older timings. Do not attribute the NUMA3 single diagnostic TF time (~3.64 s) versus NUMA6 calibration TF median (~10.94 s) to frequency or remote memory; diagnostic timings remain excluded from normal results.")


def superseded_attempts(root):
    result = []
    for name in ("ffn-infer-cpu-stages-transformer-v1", "ffn-infer-npu-stages-transformer-v1", "ffn-long-npu-timing-v1"):
        path = root / "artifacts" / name
        if not (path / "resubmission-reason.json").exists():
            continue
        reason, reason_file = read(path / "resubmission-reason.json", root)
        status, status_file = read(path / "status.json", root)
        launch, launch_file = read(path / "launch.json", root)
        result.append(dict(name=name, source_commit=launch["source"], launch=launch_file,
            outer_state=status["state"], outer_exit_code=status.get("exit_code"), status=status_file,
            reason_file=reason_file, reason=reason, queue_record_exists=(path / "queue.json").exists(),
            normal_result_files=[str(f.relative_to(root)) for f in sorted((path / "results").glob("r*.json"))],
            breakdown_result_exists=(path / "breakdown.json").exists(),
            included_in_normal_aggregation=False,
            scope="Original CPU argument failure and NPU/dependent-long cancellations before device/model workload are retained. Replacement v2 has a separate identity; no model results are imputed to v1."))
    return result


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(report):
    lines = ["常规计时只读汇总。未完成套件标为 pending；下列已有数值只来自已完成且文件哈希匹配的子任务。", "",
        "吞吐按每个 repeat 的总输入位置数 / 所有 normal 完整时间之和计算；median/range 汇总 repeat，不把 sample 中位数当配对 repeat。校准只有一个 repeat。阶段均值采用互不重叠边界及显式剩余时间；首输出、stable decode 和其他子项不重复加总。", "",
        "| 套件 | 状态 | 已完成/计划子任务 | warm / normal / repeat | CPU worker affinity |", "|---|---|---:|---|---|"]
    for suite in report["suites"]:
        a=suite.get("arguments", {})
        lines.append(f"| {suite['name']} | {suite['state']} | {len(suite.get('normal_repeats', []))}/{suite.get('expected_children', '待创建')} | {a.get('warmup')}/{a.get('samples')}/{a.get('repeats')} | {suite.get('worker_requested_affinity', '—')} |")
    if report["superseded_attempts"]:
        lines += ["", "长 decode 选择 `ffn-long-npu-timing-v2`。v1 因前置 Transformer 诊断误传 `--config` 而在进入 queue 前取消，无模型结果；CPU TF v1 参数检查失败、NPU TF v1 取消及各自 replacement 身份也保留在 JSON，均不并入 normal 数字。"]
    lines += ["", "CPU32 中尺度 worker 为 240–271，大尺度为 0–31；另做的阶段诊断使用 120–151，其时间不进入这里的 normal 分母。监督器 affinity 调整也不改写 worker 身份。当前约 3GHz/performance governor、worker 页超过 99.5% 在各自本地 NUMA 的只读快照，并未解释旧时间差；不能把 NUMA3 诊断单次 TF 约 3.64s 与 NUMA6 校准约 10.94s 混成正式加速，或自动归因频率/远端内存。", "",
        "中尺度 Transformer：32 层，d496、FFN2048、31×16 heads、window128，129,039,856 参数；大尺度固定参考：32 层，d4352、FFN15208、68×64 heads、window128，8,778,371,328 参数。大尺度 Tide KV128 为 7,268,408,784 参数；该固定 Transformer 是辅助参考，不能替代 A–D。", "",
        "| 套件 / 对象 | 每 repeat 完整 positions/s | median [min,max] | 同 repeat 吞吐比 / Tide baseline（median） | allocated/reserved peak GiB | normal-end RSS GiB |", "|---|---|---|---:|---|---:|"]
    for suite in report["suites"]:
        for s in suite.get("normal_summary", []):
            v=s["across_repeat"]["complete_positions_per_second"]
            rates=", ".join(f"r{r['repeat']}={r['complete_positions_per_second']:.3f}" for r in s["per_repeat"])
            memory=s["normal_allocator_max_by_device"]
            mem="CPU 无 device allocator" if not memory else "; ".join(f"{d}: {p['peak_allocated_bytes']/2**30:.3f}/{p['peak_reserved_bytes']/2**30:.3f}" for d,p in memory.items())
            ratio=s["paired_complete_throughput_ratio"]
            lines.append(f"| {suite['name']} / {s['label']} | {rates} | {v['median']:.3f} [{v['min']:.3f},{v['max']:.3f}] | {fmt(ratio['median'] if ratio else None)} | {mem} | {s['max_normal_end_process_lifetime_peak_rss_bytes']/2**30:.3f} |")
    cpu_training = next((s for s in report["suites"] if s["name"] == "ffn-medium-cpu-timing-v1"), None)
    if cpu_training and cpu_training.get("normal_repeats"):
        lines += ["", "中尺度 CPU 每 repeat 的完整训练均值（秒）。forward window 含准备和窗口调度，prefill 是其子项；其他边界时间由完整时间减去前三项得到。均值可加，子项不再次相加。", "",
            "| 对象/repeat | forward window | backward | optimizer | 其他边界 | 完整均值 | 同 repeat 吞吐比 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for row in sorted(cpu_training["normal_repeats"], key=lambda r:(r["label"], r["repeat"])):
            phase=row["exclusive_phase_seconds"]
            pair=next((p for p in cpu_training["same_repeat_pairs"] if p["label"] == row["label"] and p["repeat"] == row["repeat"]), None)
            lines.append(f"| {row['label']}/r{row['repeat']} | {phase['forward_window_including_preparation_seconds']['mean']:.6f} | {phase['backward_seconds']['mean']:.6f} | {phase['optimizer_seconds']['mean']:.6f} | {phase['other_complete_boundary_seconds_by_subtraction']['mean']:.6f} | {row['mean_complete_seconds']:.6f} | {fmt(pair['complete_throughput_ratio'] if pair else None, 6)} |")
    lines += ["", "| 推理套件 / 对象 | 首主干输出 repeat-median 的中位数 s | prefill positions/s 中位数 | stable decode positions/s 中位数 | 独立 batch 请求/s 中位数 |", "|---|---:|---:|---:|---:|"]
    for suite in report["suites"]:
        if suite.get("arguments", {}).get("mode") != "inference":
            continue
        for s in suite.get("normal_summary", []):
            values=s["across_repeat"]
            first=s["first_backbone_output_repeat_medians"]
            stable=values["stable_decode_positions_per_second"]
            lines.append(f"| {suite['name']} / {s['label']} | {fmt(first['median'] if first else None, 6)} | {fmt(values['prefill_positions_per_second']['median'])} | {fmt(stable['median'] if stable else None)} | {fmt(values['independent_batch_units_per_second']['median'])} |")
    lines += ["", "训练每 normal 更新为 B16×128=2048 个输入位置；5 normal 的 repeat 分母为 10240 / Σ完整秒。大 CPU 是 1 warm + 2 normal × 3 repeat；短 NPU 是 2 warm + 5 normal × 3 repeat；长 NPU 是 1 warm + 2 normal × 3 repeat。短推理每请求 B8×(128+16)=1152 位置，稳定 decode 为 B8×15=120 位置；长推理每请求 3072 位置，稳定 decode 为 B8×255=2040 位置。warmup 不入 normal 分母。", "",
        "CPU/NPU 同形初始数据检查："]
    for item in report["cross_cpu_npu_identity"]:
        f=item["fixture"]
        lines.append(f"- {f['scale']} {f['subject']}/{f['variant']}：input={item['inputs_equal']}，initial parameters={item['initial_parameters_equal']}，shape={item['input_shapes_equal']}。")
    lines += ["", "16/32 线程校准（各仅一个 repeat，不代表全局线程最优）："]
    for item in report["cpu16_cpu32_calibration"]:
        lines.append(f"- {item['label']}：{item['threads16_positions_per_second']:.3f} / {item['threads32_positions_per_second']:.3f} positions/s，32/16 比 {item['throughput_ratio_32_over_16']:.3f}。")
    lines += ["", "W/P 原有限数值失败保留，normal 完成不代表严格等价。训练经过 warmup 后，参数、优化器及私有状态轨迹不同，r0 实际活动也不完全相同；这些是固定形状、初始数据与 batch 的有限执行比较。诊断 work 不是每个计时更新的跟踪；ledger 的阶段峰值累计，不能相加，allocator residual 也不能全部叫临时工作区。", "",
        "CPU allocator 空值使用独立的 RSS 记录；RSS 含进程生命周期内构造及 host snapshot，不称 HBM。NPU normal allocator、诊断 ledger、最终 RSS 和监督器进程树 RSS 分开保留。训练 allocator 在 run_stream 返回后采样，包含该函数里计时外的有限性检查，先于另做的 continuation 检查。", "",
        "每个 repeat 的原始 normal、阶段均值可加性、同 repeat 比值、r0 work/matrix/padding 和完整 ledger、源码/输入指纹及原资格记录见 `normal-review.json`。首输出是主干可用输出，不是完整语言模型 TTFT；此证据不评估下游任务质量。", "",
        "复现汇总：`python artifacts/ffn-execution-20261119/collect_normal.py`。仅读取现有证据并原子写入汇总，不运行模型或设备。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root=args.root.resolve()
    identities=[]
    suites=[collect_suite(root, name, identities) for name in SUITES]
    existing=root / "artifacts/ffn-execution-20261119/medium-npu-timing-review.json"
    reference=None
    if existing.exists():
        data, file=read(existing, root)
        reference=dict(file=file, state=data["state"], source=data["source"],
                       trajectory=data["trajectory"], boundaries=data["boundaries"])
    script=Path(__file__).resolve()
    script_bytes=script.read_bytes()
    report=dict(schema="tide-ffn-normal-cost-read-review-v1",
        state="pending_suites" if any(s["state"] == "pending" for s in suites) else "completed_with_incomplete_suites" if any(s["state"] != "complete" for s in suites) else "completed_read_review",
        observed_at=datetime.now(timezone.utc).isoformat(),
        script=dict(path=str(script.relative_to(root)), bytes=len(script_bytes), sha256=hashlib.sha256(script_bytes).hexdigest()),
        suites=suites, medium_npu_previous_review=reference,
        cross_cpu_npu_identity=cross_identity(identities), qualification_records=qualification_refs(root),
        cpu16_cpu32_calibration=calibration_comparison(suites),
        cpu_locality_saved_observation=locality_reference(root),
        superseded_attempts=superseded_attempts(root),
        memory_boundary_correction=dict(correcting_documentation_commit="46317a8d", raw_json_unchanged=True,
            applies_to="execution_candidates boundary.memory text in a5853660/2b8851a7 records that says before checks",
            corrected_scope="run_stream finite checks are excluded from complete_seconds, but allocator is sampled after run_stream returns and includes those temporary allocations; it precedes the subsequent continuation audit."),
        policies=dict(strict_equivalence=False,
            numerical="Original W/P finite failures and resource-aborted qualification attempts are preserved; timing completion does not qualify numerical equivalence.",
            paired="candidate throughput / Tide baseline throughput at the same repeat; TF is a different auxiliary architecture, not an implementation-speedup proof.",
            phases="Training top-level forward_window + backward + optimizer + explicitly derived residual; prefill/flush/input-window/first-output are subitems. Inference prefill + all decode + stream + drain + residual; stable decode is a decode subset. Never sum a phase with its children.",
            aggregation="Each repeat uses summed input positions / summed complete time; median/min/max across available completed repeats; pooled rates separately labeled. Incomplete suites stay pending; missing repeats are never imputed.",
            state="Training continues through normal updates; each r0 diagnostic restores its own warmed anchor and is not the exact trace of all timed samples. Actual work is not guaranteed equal across variants or CPU/NPU.",
            memory="Device allocated/reserved, host process-lifetime RSS, sampled process-tree RSS and intrusive storage ledgers have different boundaries. CPU empty allocator is unavailable, not zero HBM. Diagnostic phase peaks are cumulative, not additive.",
            hardware="One local Ascend device per admitted NPU job; CPU32 medium affinity240-271, large0-31; separate stage diagnostics120-151 are excluded from normal aggregation. Host load and real child affinity retained.",
            output="First backbone output only; no task head, token sampling or language-model TTFT claim. No downstream quality conclusion."))
    destination=root / "artifacts/ffn-execution-20261119"
    atomic(destination / "normal-review.json", report)
    atomic(destination / "normal-review.md", markdown(report))
    print(json.dumps(dict(state=report["state"], suites=[dict(name=s["name"], state=s["state"],
        completed_children=len(s.get("normal_repeats", [])), remaining_children=s.get("remaining_children_count"), rates={x["label"]:x["across_repeat"]["complete_positions_per_second"]["median"] for x in s.get("normal_summary", [])}) for s in suites],
        cross_cpu_npu_initial_hashes_match=all(c["inputs_equal"] and c["initial_parameters_equal"] for c in report["cross_cpu_npu_identity"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
