"""Read-only round-two review: matched prefixes, real continuation and provenance.

Raw records remain authoritative. This module never loads checkpoints, starts
workers, changes frozen inputs, or copies long step logs into review artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from . import report as base


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_FIELDS = ("model", "optimizer", "private_state", "logits", "gradients")
EXACT_STEP_FIELDS = ("next_input_exact", "rng_before_exact", "rng_after_exact", "stream_after_exact",
                     "step_exact", "state_reset_exact")
RECIPE_DEFAULTS = {"stream_chunks": 4, "precision": "fp32", "warmup_steps": 10,
                   "weight_decay": 0.01, "grad_clip": 1.0, "measurement_warmup_steps": 3,
                   "checkpoint_interval_steps": 0, "eval_batches": 2, "decode_steps": 8}
RECIPE_FIELDS = ("model", "backend", "seed", "batch_size", "sequence_length", "cpu_threads", "lr")
LIMITATIONS = [
    "第二轮逐步训练计时包含 input/RNG 指纹；第一轮测量边界不同，跨轮吞吐不作为纯性能回归或架构排名。",
    "配对比值只使用相同模型、配方、源码、CPU mask（NPU 还须相同 physical card）且输入/RNG 完全一致的共同 step 前缀，并去掉共同 warmup。",
    "配对固定声明工作负载与公共输入/RNG，未逐更新捕获或验证路由及全部活动张量相等；冷态活动探针不补足此证据，比值不称为严格实现等价加速。",
    "训练区间由 steps.jsonl 的 time_unix 和 seconds 重建，终点为记录时间，存在记录开销偏移；交集仅覆盖主训练更新，不覆盖验证、恢复和未知外部负载。",
    "worker 同时存活与训练更新重叠分别统计；实际并发不足时不声称达到计划并发条件。",
    "lr=0 成本任务执行前向、反向、裁剪和 AdamW；其参数不变符合设计，token 不计为学习证据。",
    "学习 token 只计保存轨迹中的主训练更新；同 owner 参考更新和新进程恢复验证不增加该 checkpoint 的学习曝光。",
    "18 个父任务的三档学习率与两个种子仅用于模型内探索筛选；固定文章留出探针不能建立架构质量排名。",
    "恢复逐步比较保留 atol=1e-5、rtol=1e-4，整数和结构严格相等；梯度记录位于全局范数裁剪后。",
    "因果探针在每段开始时使用空私有状态、B=1、16 token；接续段使用恢复后的权重，不认证全部训练状态和轨迹。",
    "RSS 是进程树软件看守的离散采样，共享页可能重复计入，采样间可超过阈值；不等于内核硬上限，也不包含文件缓存或 NPU 设备内存。",
    "退出状态、浮点、离散、恢复、学习和性能分别报告；缺失或失败的测量不会填成零。",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def record_identity(path: Path) -> dict | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def jsonl_rows(path: Path, errors: list[dict]):
    if not path.is_file():
        return
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected a JSON object")
                yield value
            except (ValueError, TypeError) as exc:
                errors.append({"path": str(path), "line": line_number, "error": str(exc)})


def numeric_summary(values) -> dict:
    observed = sorted(value for item in values if (value := base.number(item)) is not None)
    result = base.distribution(observed)
    result["mean"] = statistics.mean(observed) if observed else None
    result["p95"] = observed[min(len(observed) - 1, int((len(observed) - 1) * .95))] if observed else None
    return result


def merged_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_intersection(left, right):
    answer, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            answer.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return answer


def training_intervals(rows: list[dict]) -> list[tuple[float, float]]:
    intervals = []
    for row in rows:
        end, duration = base.number(row.get("time_unix")), base.number(row.get("seconds"))
        if end is not None and duration is not None and duration > 0:
            intervals.append((end - duration, end))
    return merged_intervals(intervals)


def interval_seconds(intervals):
    return sum(end - start for start, end in intervals)


def recipe(task: dict) -> dict:
    return {**{key: task.get(key) for key in RECIPE_FIELDS},
            **{key: task.get(key, default) for key, default in RECIPE_DEFAULTS.items()}}


def exposure_kind(task: dict) -> str:
    lr = base.number(task.get("lr"))
    if task.get("purpose") == "cost_calibration" or lr == 0:
        return "cost"
    return "learning" if lr is not None and lr > 0 else "unknown"


def recovery_summary(raw: dict, task: dict, worker: dict) -> dict:
    rows = raw.get("per_step")
    records = rows if isinstance(rows, list) else []
    requested = task.get("resume_verify_steps", 1)
    start = raw.get("from_step")
    comparisons = raw.get("comparisons", {})
    condensed = []
    missing, failed = [], []
    if records:
        for key in COMPARISON_FIELDS:
            for flag in ("within_tolerance", "finite", "structure_equal", "integer_equal"):
                value = comparisons.get(key, {}).get(flag)
                if value is not True:
                    (failed if value is False else missing).append(f"aggregate:{key}:{flag}")
    for index, row in enumerate(records):
        fields = row.get("comparisons", {})
        checks = {key: {name: fields.get(key, {}).get(name) for name in
                       ("within_tolerance", "exact", "finite", "structure_equal", "integer_equal", "max_abs",
                        "failed_field_count", "failed_fields")}
                  for key in COMPARISON_FIELDS}
        good = True
        for key, check in checks.items():
            for flag in ("within_tolerance", "finite", "structure_equal", "integer_equal"):
                if check[flag] is not True:
                    good = False
                    (failed if check[flag] is False else missing).append(f"step{index + 1}:{key}:{flag}")
            if fields.get(key, {}).get("atol") != 1e-5 or fields.get(key, {}).get("rtol") != 1e-4:
                good = False
                failed.append(f"step{index + 1}:{key}:threshold_changed_or_missing")
        for key in EXACT_STEP_FIELDS:
            if row.get(key) is not True:
                good = False
                (failed if row.get(key) is False else missing).append(f"step{index + 1}:{key}")
        correct_step = type(start) is int and row.get("step") == start + index + 1
        if not correct_step:
            failed.append(f"step{index + 1}:global_step_sequence")
        condensed.append({"step": row.get("step"), **{key: row.get(key) for key in EXACT_STEP_FIELDS},
                          "continued_private_state": row.get("continued_private_state"),
                          "recorded_within_tolerance": row.get("within_tolerance"),
                          "strict_step_qualified": good and correct_step,
                          "reference_loss": row.get("reference_loss"), "restored_loss": row.get("restored_loss"),
                          "comparisons": checks})
    fresh = raw.get("fresh_pid_result", {})
    validations = fresh.get("validation", {})
    boundary_checks = {key: raw.get(key) for key in
                       ("saved_data_exact", "next_input_exact", "rng_before_exact", "rng_after_exact",
                        "stream_after_exact", "continued_private_state")}
    boundary_checks.update({key: validations.get(key) for key in
                            ("identity_exact", "contract_exact", "state_digest_exact")})
    boundary_checks["fresh_owner"] = (fresh.get("pid") != worker.get("pid")
                                       if fresh.get("pid") is not None and worker.get("pid") is not None else None)
    for key, value in boundary_checks.items():
        if value is not True:
            (failed if value is False else missing).append(key)
    count_exact = (type(requested) is int and requested > 0 and len(records) == requested
                   and raw.get("verified_steps") == requested and raw.get("requested_steps") == requested
                   and type(start) is int and raw.get("to_step") == start + requested)
    strict = (raw.get("status") == "within_tolerance" and count_exact and not failed and not missing
              and bool(records) and all(row["strict_step_qualified"] for row in condensed))
    return {"status": raw.get("status", "unknown"), "requested_steps": requested,
            "recorded_steps": len(records) if rows is not None else None,
            "recorded_count_and_range_exact": count_exact,
            "from_step": start, "to_step": raw.get("to_step"), "seconds": raw.get("seconds"),
            "strict_scope_qualified": strict, "eight_step_qualified": strict and requested >= 8,
            "boundary_checks": boundary_checks, "private_history": raw.get("private_history"),
            "comparisons": comparisons, "per_step": condensed,
            "failed_checks": failed[:40], "missing_checks": missing[:40],
            "failed_check_count": len(failed), "missing_check_count": len(missing),
            "test_scope": "same original owner versus fresh interpreter, consecutive real optimizer updates",
            "fresh_process": {key: fresh.get(key) for key in
                              ("pid", "parent_pid", "status", "started_unix", "ended_unix", "memory")}}


def source_reference(identity: Any, catalog: dict) -> str | None:
    if not isinstance(identity, dict) or not identity:
        return None
    key = canonical_sha256(identity)
    catalog.setdefault(key, identity)
    return key


def observed_equal(left, right):
    return left == right if left is not None and right is not None else None


def summarize_task(item: dict, planned: dict, errors: list, catalog: dict):
    raw = item["raw_records"]
    task, result, process = raw["task"], raw["result"] or {}, raw["process"] or {}
    worker = raw["worker"] or {}
    path = Path(item["directory"])
    declared = base.read_record(path / "declared-task.json", errors)
    selection = base.read_record(path / "selection.json", errors)
    resolution = process.get("task_resolution", {})
    sources = item["source_record_identities"]
    for name in ("declared-task.json", "selection.json", "steps.jsonl", "checkpoint.json"):
        sources[name] = record_identity(path / name)
    rows = list(jsonl_rows(path / "steps.jsonl", errors))
    intervals = training_intervals(rows)
    runtime = result.get("checkpoint_identity")
    expected_ref = source_reference(task.get("checkpoint_identity"), catalog)
    runtime_ref = source_reference(runtime, catalog)
    checkpoint = item.get("checkpoint")
    checkpoint_ref = source_reference((checkpoint or {}).get("identity"), catalog)
    declared_digest = canonical_sha256(declared) if declared is not None else None
    resolved_digest = canonical_sha256(task) if sources.get("task.json") is not None else None
    changed = sorted(key for key in set(declared or {}) | set(task)
                     if (declared or {}).get(key) != task.get(key)) if declared is not None else None
    allowed = ({"resume_from", "resume_checkpoint_sha256"}
               if declared and (declared.get("resume_select") or declared.get("resume_task")) else set())
    if declared and declared.get("resume_select"):
        allowed.update(("resume_select", "lr", "resume_task"))
    item.pop("raw_records")
    item.pop("configuration_matches_plan")
    item.update(phase=task.get("phase"), purpose=task.get("purpose"), exposure_kind=exposure_kind(task),
                selection_family=task.get("selection_family"), comparison_group=task.get("comparison_group"),
                comparison_role=task.get("comparison_role"), recipe=recipe(task))
    item["configuration"] = {"declared_record_present": declared is not None,
        "declared_matches_plan": observed_equal(declared, planned),
        "declared_task_sha256": declared_digest, "resolved_task_sha256": resolved_digest,
        "declared_hash_matches_resolution": observed_equal(declared_digest, resolution.get("declared_task_sha256")),
        "resolved_hash_matches_resolution": observed_equal(resolved_digest, resolution.get("resolved_task_sha256")),
        "resolved_file_hash_matches_process": observed_equal((sources.get("task.json") or {}).get("sha256"),
                                                              process.get("task_file_sha256")),
        "worker_task_matches_resolved": observed_equal(result.get("task"), task if resolved_digest else None),
        "changed_fields_at_resolution": changed,
        "only_declared_resolution_fields_changed": set(changed) <= allowed if changed is not None else None,
        "task_resolution": resolution,
        "selection_content_hash_matches_resolution": observed_equal(
            canonical_sha256(selection) if selection is not None else None,
            resolution.get("selection", {}).get("sha256"))}
    item["source_identity"] = {"expected": expected_ref, "observed": runtime_ref, "checkpoint": checkpoint_ref,
                               "expected_matches_observed": observed_equal(expected_ref, runtime_ref),
                               "checkpoint_matches_observed": observed_equal(checkpoint_ref, runtime_ref),
                               "data_manifest_sha256": result.get("data_manifest_sha256")}
    if checkpoint:
        item["checkpoint"] = {key: value for key, value in checkpoint.items() if key != "identity"}
    item["recovery"] = recovery_summary(result.get("recovery", {}), task, worker)
    item["continuation"] = result.get("continuation")
    item["resources"] = {"estimated_rss_bytes": task.get("estimated_rss_bytes"),
                         "estimated_output_bytes": task.get("estimated_output_bytes"),
                         "cpu_affinity_observed": worker.get("affinity", process.get("cpu_affinity")),
                         "worker_and_scheduler_affinity_agree": observed_equal(worker.get("affinity"), process.get("cpu_affinity")),
                         "process_tree_peak_rss_bytes": process.get("peak_rss_bytes"),
                         "rss_samples": process.get("rss_samples"),
                         "memory_admission": process.get("memory_admission")}
    if item["training"]:
        training = item["training"]
        training["step_seconds_distribution"] = numeric_summary(training.pop("step_seconds", []))
        training["segment_steps"] = training.get("steps")
        training["global_steps"] = training.get("global_steps", training.get("to_step"))
    item["training_interval_evidence"] = {
        "recorded_step_rows": len(rows) if sources["steps.jsonl"] else None,
        "updates_with_time_boundaries": sum(base.number(row.get("time_unix")) is not None
                                             and base.number(row.get("seconds")) is not None for row in rows),
        "interval_count_after_merge": len(intervals) if rows else None,
        "first_update_start_estimate_unix": intervals[0][0] if intervals else None,
        "last_update_end_recorded_unix": intervals[-1][1] if intervals else None,
        "update_interval_seconds": interval_seconds(intervals) if intervals else None,
        "scope": "sum/union of reconstructed update intervals; checkpoint, evaluation and restore gaps excluded"}
    item["checks"]["causality_test_design"]["model_stage"] = (
        "segment_start_restored_weights" if task.get("resume_from") else "initialization")
    if item["exposure_kind"] == "cost":
        item["parameter_updates_interpretation"] = "lr=0 cost calibration: unchanged parameters are expected and are not a learning failure"
        item["validation_probe"]["interpretation"] = "cost-task probe; not counted as learning evidence"
    return item, {"task": task, "declared": declared, "result": result, "process": process,
                  "selection": selection, "rows": rows, "intervals": intervals}


def concurrency_exposure(identity: str, target_intervals, contexts: dict, task_by_id: dict, policy: dict) -> dict:
    target = task_by_id[identity]
    events = defaultdict(lambda: {"target": 0, "cpu": 0, "npu": 0})
    for start, end in target_intervals:
        events[start]["target"] += 1
        events[end]["target"] -= 1
    peers = []
    for other_id, context in contexts.items():
        other = task_by_id[other_id]
        if other_id == identity or other.get("phase") != target.get("phase"):
            continue
        overlaps = interval_intersection(target_intervals, context["intervals"])
        if not overlaps:
            continue
        backend = other["backend"]
        peers.append({"task_id": other_id, "backend": backend, "overlap_seconds": interval_seconds(overlaps)})
        for start, end in overlaps:
            events[start][backend] += 1
            events[end][backend] -= 1
    active = {"target": 0, "cpu": 0, "npu": 0}
    duration = defaultdict(float)
    previous = None
    for when, deltas in sorted(events.items()):
        if previous is not None and active["target"]:
            cpu = active["cpu"] + (target["backend"] == "cpu")
            npu = active["npu"] + (target["backend"] == "npu")
            duration[cpu, npu] += when - previous
        for key, value in deltas.items():
            active[key] += value
        previous = when
    total = sum(duration.values())
    target_cpu, target_npu = policy.get("cpu_slots"), policy.get("npu_slots")
    full = sum(seconds for (cpu, npu), seconds in duration.items()
               if target_cpu is not None and target_npu is not None and cpu >= target_cpu and npu >= target_npu)
    any_overlap = sum(seconds for (cpu, npu), seconds in duration.items() if cpu + npu > 1)
    return {"observed_update_seconds": total if duration else None, "overlapping_tasks": peers,
            "seconds_by_concurrency": [{"cpu": cpu, "npu": npu, "seconds": seconds}
                                       for (cpu, npu), seconds in sorted(duration.items())],
            "peak_simultaneous_training_tasks": max((cpu + npu for cpu, npu in duration), default=None),
            "time_weighted_training_concurrency": sum((cpu + npu) * seconds for (cpu, npu), seconds in duration.items()) / total if total else None,
            "any_training_overlap_fraction": any_overlap / total if total else None,
            "planned_concurrency": {"cpu": target_cpu, "npu": target_npu},
            "planned_concurrency_fraction": full / total if total else None,
            "planned_concurrency_reached": bool(full) if duration else None,
            "planned_concurrency_throughout_measured_prefix": full >= total - 1e-9 if total else None,
            "scope": "training-update interval intersections in this phase; other job stages/external loads not inferred"}


def prefix_evidence(left: list[dict], right: list[dict], warmup: int) -> dict:
    common, matched, first_problem = min(len(left), len(right)), 0, None
    for index, (a, b) in enumerate(zip(left, right)):
        for key in ("step", "tokens", "state_reset", "input_sha256", "rng_before_sha256", "rng_after_sha256"):
            if key not in a or key not in b or a[key] != b[key]:
                first_problem = {"index": index, "field": key, "left_present": key in a,
                                 "right_present": key in b, "reason": "missing_or_different"}
                break
            if key.endswith("sha256") and (not isinstance(a[key], str) or len(a[key]) != 64):
                first_problem = {"index": index, "field": key, "reason": "invalid_digest"}
                break
        if first_problem:
            break
        if type(a.get("step")) is not int or a["step"] < 1:
            first_problem = {"index": index, "field": "step", "reason": "invalid_global_step"}
            break
        if index and (a["step"] != left[index - 1]["step"] + 1 or b["step"] != right[index - 1]["step"] + 1):
            first_problem = {"index": index, "field": "step", "reason": "nonconsecutive"}
            break
        matched += 1
    start = max(0, warmup)
    measured = max(0, common - start)
    values = []
    for rows in (left, right):
        selected = rows[start:common]
        valid = bool(selected) and all((base.number(row.get("seconds")) or 0) > 0
                                      and (base.number(row.get("tokens")) or 0) > 0 for row in selected)
        seconds = sum(base.number(row["seconds"]) for row in selected) if valid else None
        tokens = sum(base.number(row["tokens"]) for row in selected) if valid else None
        values.append({"tokens": tokens, "seconds": seconds,
                       "tokens_per_second": tokens / seconds if seconds else None})
    return {"common_step_count": common, "matching_prefix_steps": matched,
            "all_common_inputs_and_rng_exact": common > 0 and matched == common,
            "first_problem": first_problem, "warmup_steps_excluded": min(start, common),
            "measured_prefix_steps": measured, "first_global_step": left[start].get("step") if measured else None,
            "last_global_step": left[common - 1].get("step") if measured else None,
            "isolated": values[0], "concurrent": values[1]}


def paired_comparisons(tasks, contexts, phases):
    grouped, task_by_id = defaultdict(list), {task["id"]: task for task in tasks}
    for task in tasks:
        if task.get("comparison_group"):
            grouped[task["comparison_group"]].append(task)
    output = []
    for group, members in sorted(grouped.items()):
        isolated = [task for task in members if task.get("comparison_role") == "isolated"]
        concurrent = [task for task in members if task.get("comparison_role") == "concurrent"]
        if len(isolated) != 1 or not concurrent or (isolated[0]["backend"] == "npu" and len(concurrent) != 1):
            output.append({"comparison_group": group, "status": "isolated_only_design" if len(isolated) == 1 and not concurrent else "insufficient_pair_members",
                           "isolated_ids": [task["id"] for task in isolated],
                           "concurrent_ids": [task["id"] for task in concurrent], "throughput_ratio": None})
            continue
        a = isolated[0]
        for b in concurrent:
            reasons = []
            same_recipe = a["recipe"] == b["recipe"]
            mask_a, mask_b = a["resources"]["cpu_affinity_observed"], b["resources"]["cpu_affinity_observed"]
            same_mask = bool(mask_a) and bool(mask_b) and sorted(mask_a) == sorted(mask_b)
            same_source = (a["source_identity"]["observed"] is not None
                           and a["source_identity"]["observed"] == b["source_identity"]["observed"])
            same_npu = (a["backend"] != "npu" or bool(a["device_mapping"]["physical_ids"])
                        and a["device_mapping"]["physical_ids"] == b["device_mapping"]["physical_ids"])
            checks = {"same_model_and_recipe": same_recipe, "same_cpu_mask": same_mask,
                      "same_execution_source": same_source, "same_npu_physical": same_npu,
                      "same_actual_cpu_threads": a["workload"]["cpu_threads_actual"] is not None
                      and a["workload"]["cpu_threads_actual"] == b["workload"]["cpu_threads_actual"] == a["recipe"]["cpu_threads"],
                      "observed_source_matches_declared": a["source_identity"]["expected_matches_observed"] is True
                      and b["source_identity"]["expected_matches_observed"] is True,
                      "cost_design_lr_zero": a["recipe"].get("lr") == b["recipe"].get("lr") == 0}
            reasons.extend(key for key, value in checks.items() if not value)
            left, right = contexts[a["id"]]["rows"], contexts[b["id"]]["rows"]
            warmup = contexts[a["id"]]["task"].get("measurement_warmup_steps", 3)
            prefix = prefix_evidence(left, right, warmup)
            if not prefix["all_common_inputs_and_rng_exact"]:
                reasons.append("common_prefix_fingerprint_not_exact")
            if prefix["isolated"]["tokens_per_second"] is None or prefix["concurrent"]["tokens_per_second"] is None:
                reasons.append("missing_measured_prefix")
            common = prefix["common_step_count"]
            exposure = concurrency_exposure(b["id"], training_intervals(right[warmup:common]), contexts,
                                            task_by_id, phases.get(b["phase"], {}))
            isolated_exposure = concurrency_exposure(a["id"], training_intervals(left[warmup:common]), contexts,
                                                     task_by_id, phases.get(a["phase"], {}))
            if not (exposure["any_training_overlap_fraction"] or 0) > 0:
                reasons.append("concurrent_training_overlap_not_observed")
            if (isolated_exposure["any_training_overlap_fraction"] or 0) > 0:
                reasons.append("isolated_training_overlaps_round_peers")
            valid = not reasons
            ratio = prefix["concurrent"]["tokens_per_second"] / prefix["isolated"]["tokens_per_second"] if valid else None
            output.append({"comparison_group": group, "isolated_id": a["id"], "concurrent_id": b["id"],
                           "backend": a["backend"], "status": "matched_prefix" if valid else "not_qualified",
                           "qualification_checks": checks, "reasons": reasons, "prefix": prefix,
                           "training_overlap": exposure, "isolated_training_overlap": isolated_exposure,
                           "target_load_status": "observed_throughout_prefix" if exposure["planned_concurrency_throughout_measured_prefix"]
                           else "insufficient_or_variable_training_concurrency",
                           "throughput_ratio": ratio,
                           "ratio_definition": "concurrent / isolated tokens-per-second on identical measured step prefix",
                           "cold_probe_activity_agreement": {stage: observed_equal(a["work_observations"].get(stage),
                                                                                   b["work_observations"].get(stage))
                                                              for stage in ("initial", "final")},
                           "activity_scope": "task work_observations contain B=1 cold-state summaries; per-update routing and all activity tensors were not captured or compared",
                           "ratio_scope": "matched declared workload/public input and observed overlap; no strict implementation-equivalence speedup claim; underfilled phases are not full planned-load measurements"})
    return output


def memory_summary(round_path: Path, scheduler: dict, errors: list) -> dict:
    guard = scheduler.get("memory_guard", {})
    scans, intervals, totals, available = [], [], [], []
    task_peaks, rows_count = {}, 0
    for row in jsonl_rows(round_path / "memory.jsonl", errors):
        rows_count += 1
        scans.append(row.get("scan_seconds"))
        intervals.append(row.get("interval_since_previous_seconds"))
        totals.append(row.get("total_rss_bytes"))
        available.append(row.get("host", {}).get("available_bytes"))
        for identity, value in row.get("task_rss_bytes", {}).items():
            value = base.number(value)
            if value is not None:
                task_peaks[identity] = max(task_peaks.get(identity, value), value)
    total_summary = numeric_summary(totals)
    return {"requested": guard.get("requested"), "effective": guard.get("effective"),
            "enforcement": guard.get("enforcement"),
            "kernel_memory_limit_established": guard.get("kernel_memory_limit_established"),
            "rss_scope": guard.get("rss_scope"), "limitations": guard.get("limitations"),
            "scheduler_reported_samples": guard.get("samples"), "observed_sample_records": rows_count,
            "sample_count_matches": observed_equal(guard.get("samples"), rows_count)
            if (round_path / "memory.jsonl").is_file() else None,
            "sampled_peak_rss_bytes": total_summary["max"],
            "scheduler_reported_peak_rss_bytes": guard.get("peak_rss_bytes"),
            "peak_records_agree": observed_equal(total_summary["max"], guard.get("peak_rss_bytes")),
            "scan_seconds": numeric_summary(scans), "actual_scan_interval_seconds": numeric_summary(intervals),
            "sampled_host_available_bytes": numeric_summary(available), "task_sampled_peak_rss_bytes": task_peaks,
            "error": guard.get("error"), "source_record_identity": record_identity(round_path / "memory.jsonl")}


def phase_summaries(tasks: list, contexts: dict, plan: dict, scheduler: dict):
    policies = {phase["id"]: phase for phase in plan.get("phases", [])}
    recorded = {phase["id"]: phase for phase in scheduler.get("phases", [])}
    result = []
    for identity in dict.fromkeys([*policies, *recorded]):
        phase = recorded.get(identity, {})
        members = [task for task in tasks if task.get("phase") == identity]
        by_backend = {backend: [interval for task in members if task["backend"] == backend
                               for interval in contexts[task["id"]]["intervals"]] for backend in ("cpu", "npu")}
        by_backend["all"] = by_backend["cpu"] + by_backend["npu"]
        peaks = {backend: base.interval_peak(intervals) if intervals else
                 {"peak": None, "first_peak_unix": None, "interval_count": 0}
                 for backend, intervals in by_backend.items()}
        result.append({"id": identity, "policy": policies.get(identity), "status": phase.get("status", "unknown"),
                       "started_unix": phase.get("started_unix"), "ended_unix": phase.get("ended_unix"),
                       "wall_seconds": base.difference(phase.get("ended_unix"), phase.get("started_unix")),
                       "task_count": len(members),
                       "execution_counts": dict(Counter(task["execution"]["status"] for task in members)),
                       "recorded_training_task_count": sum(bool(contexts[task["id"]]["intervals"]) for task in members),
                       "training_interval_peak": peaks,
                       "training_union_seconds": interval_seconds(merged_intervals(by_backend["all"]))
                       if by_backend["all"] else None})
    return result


def token_totals(tasks: list) -> dict:
    output = {}
    for kind in ("cost", "learning", "unknown"):
        members = [task for task in tasks if task["exposure_kind"] == kind]
        values = [base.number((task.get("training") or {}).get("tokens")) for task in members]
        known = [value for value in values if value is not None]
        output[kind] = {"task_count": len(members), "observed_training_records": len(known),
                        "missing_training_records": len(values) - len(known),
                        "tokens": sum(known) if known else None}
    output["scope"] = "main training segment updates only; continuation adds only new tokens; verifier reference/replay excluded"
    return output


def selection_summary(tasks: list, contexts: dict, scheduled: dict, round_path: Path):
    from .selection import resolve_selection
    by_id = {task["id"]: task for task in tasks}
    candidates = []
    for task in tasks:
        if task.get("purpose") != "equal_token_lr_screen":
            continue
        training = task.get("training") or {}
        candidates.append({"id": task["id"], "family": task.get("selection_family"),
                           "lr": task["recipe"]["lr"], "seed": task["recipe"]["seed"],
                           "execution_status": task["execution"]["status"],
                           "worker_status": task["execution"]["worker_status"],
                           "segment_steps": training.get("steps"), "global_steps": training.get("global_steps"),
                           "tokens": training.get("tokens"),
                           "validation_nll": (task["validation_probe"].get("final") or {}).get("nll"),
                           "strict_eight_step_recovery": task["recovery"]["eight_step_qualified"],
                           "source_record_identities": {key: task["source_record_identities"].get(key)
                                                        for key in ("task.json", "result.json")}})
    children = []
    for task in tasks:
        if task.get("purpose") != "selected_multistep_continuation":
            continue
        context, selection = contexts[task["id"]], contexts[task["id"]]["selection"]
        resolved, declared = context["task"], context["declared"]
        record = {"id": task["id"], "family": task.get("selection_family"),
                  "execution_status": task["execution"]["status"], "seed": resolved.get("seed"),
                  "resolved_lr": resolved.get("lr"), "selection_record_present": selection is not None,
                  "selection_file_identity": task["source_record_identities"].get("selection.json"),
                  "selected_parent": selection.get("selected_parent") if selection else None,
                  "winner": selection.get("winner") if selection else None,
                  "groups": selection.get("groups") if selection else None,
                  "candidate_count": len(selection["candidates"]) if selection and "candidates" in selection else None,
                  "selection_recomputed_exact": None, "selection_recompute_error": None,
                  "training": {key: (task.get("training") or {}).get(key) for key in
                               ("steps", "tokens", "from_step", "to_step", "global_steps")},
                  "strict_eight_step_recovery": task["recovery"]["eight_step_qualified"]}
        if selection is not None and declared is not None:
            try:
                _, recomputed = resolve_selection(declared, scheduled, round_path)
                record["selection_recomputed_exact"] = canonical_sha256(recomputed) == canonical_sha256(selection)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                record["selection_recompute_error"] = f"{type(exc).__name__}: {exc}"
                record["selection_recomputed_exact"] = False
        parent_id = record["selected_parent"] or resolved.get("resume_task")
        parent = by_id.get(parent_id)
        continuation = task.get("continuation") or {}
        resolution = task["configuration"]["task_resolution"].get("resume_parent", {})
        declared_for_budget = declared or resolved
        spec = declared_for_budget.get("resume_select", {})
        expected_parent, additional = spec.get("updates"), declared_for_budget.get("max_steps")
        record["planned_range"] = {"parent_updates": expected_parent, "additional_updates": additional,
                                   "target_global_steps": expected_parent + additional
                                   if type(expected_parent) is int and type(additional) is int else None}
        if parent is not None:
            parent_checkpoint = parent.get("checkpoint") or {}
            parent_context = contexts[parent_id]
            training = task.get("training") or {}
            hashes = [parent_checkpoint.get("sha256"), resolved.get("resume_checkpoint_sha256"),
                      resolution.get("sha256"), continuation.get("sha256")]
            contract_parent = recipe(parent_context["task"])
            record["inheritance"] = {
                "parent_id": parent_id, "parent_checkpoint_step": parent_checkpoint.get("step"),
                "resolved_parent_matches_selection": observed_equal(parent_id, resolved.get("resume_task")),
                "resolution_parent_matches_selection": observed_equal(parent_id, resolution.get("task_id")),
                "recipe_inherited": contract_parent == recipe(resolved),
                "parent_seed_matches": parent["recipe"].get("seed") == resolved.get("seed"),
                "source_identity_matches": observed_equal(parent["source_identity"]["observed"], task["source_identity"]["observed"]),
                "recorded_checkpoint_hashes_consistent": len(set(hashes)) == 1 if all(hashes) else None,
                "global_start_matches_parent_checkpoint": observed_equal(training.get("from_step"), parent_checkpoint.get("step")),
                "global_end_equals_start_plus_segment": training.get("to_step") == training["from_step"] + training["steps"]
                if type(training.get("from_step")) is int and type(training.get("steps")) is int else None,
                "checkpoint_identity_validated_by_worker": continuation.get("identity_exact"),
                "contract_validated_by_worker": continuation.get("contract_exact"),
                "application_state_digest_validated_by_worker": continuation.get("state_digest_exact"),
                "first_input_reused_private_history": continuation.get("first_input_reused_private_history"),
                "selected_parent_strict_eight_step": parent["recovery"]["eight_step_qualified"],
                "budget_overrides": continuation.get("budget_overrides")}
        else:
            record["inheritance"] = None
        winner_ids = (selection or {}).get("winner", {}).get("candidate_ids", [])
        record["winner_all_parents_strict_eight_step"] = (
            all(by_id[identity]["recovery"]["eight_step_qualified"] for identity in winner_ids)
            if winner_ids and all(identity in by_id for identity in winner_ids) else None)
        children.append(record)
    families = []
    for family in sorted({row["family"] for row in candidates if row["family"] is not None}):
        members = [row for row in candidates if row["family"] == family]
        groups = []
        for lr in sorted({row["lr"] for row in members}):
            group = [row for row in members if row["lr"] == lr]
            known = [row["validation_nll"] for row in group if base.number(row["validation_nll"]) is not None]
            groups.append({"lr": lr, "candidate_ids": [row["id"] for row in group],
                           "seeds": [row["seed"] for row in group], "observed_nll_count": len(known),
                           "mean_nll_all_declared_seeds": statistics.mean(known) if len(known) == len(group) and group else None,
                           "all_strict_eight_step_qualified": all(row["strict_eight_step_recovery"] for row in group),
                           "all_completed": all(row["execution_status"] == row["worker_status"] == "completed" for row in group)})
        families.append({"family": family, "lr_groups": groups,
                         "selected_children": [row["id"] for row in children if row["family"] == family]})
    return {"parent_count": len(candidates), "continuation_count": len(children),
            "selection_records_observed": sum(row["selection_record_present"] for row in children),
            "parents": candidates, "families": families, "continuations": children}


def verify_source_files(root: Path, mapping: dict) -> dict:
    mismatches, missing = [], []
    for relative, expected in mapping.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            mismatches.append({"path": relative, "reason": "unsafe_relative_path"})
            continue
        expected_hash = expected.get("sha256") if isinstance(expected, dict) else expected
        identity = record_identity(root / path)
        if identity is None:
            missing.append(relative)
        elif identity["sha256"] != expected_hash:
            mismatches.append({"path": relative, "expected": expected_hash, "observed": identity["sha256"]})
    return {"root": str(root), "files_declared": len(mapping), "missing": missing,
            "mismatches": mismatches, "all_observed_files_match": bool(mapping) and not missing and not mismatches}


def source_review(round_path: Path, scheduler: dict, catalog: dict, errors: list):
    source = Path(scheduler["source"]) if scheduler.get("source") else None
    candidates = [source.parent / "identity.json", source / "identity.json"] if source else []
    selected_path, frozen = None, None
    for path in candidates:
        frozen = base.read_record(path, errors)
        if frozen is not None:
            selected_path = path
            break
    foundation = Path(scheduler["foundation_source"]) if scheduler.get("foundation_source") else None
    validations = {}
    for identity, value in catalog.items():
        validations[identity] = {
            "experiment": verify_source_files(source / "pilotlab", value.get("experiment_files", {})) if source else None,
            "foundation": verify_source_files(foundation / "tide", value.get("foundation_files", {})) if foundation else None}
    manifest_files = None
    if source and frozen:
        declared = frozen.get("source_sha256", frozen.get("source_files", frozen.get("files")))
        if isinstance(declared, dict):
            manifest_files = verify_source_files(selected_path.parent if "source_sha256" in frozen else source, declared)
    plan_identity = record_identity(round_path / "plan.json")
    return {"source": str(source) if source else None, "foundation_source": str(foundation) if foundation else None,
            "resolved_environment": scheduler.get("resolved_environment"),
            "frozen_identity_file": record_identity(selected_path) if selected_path else None,
            "frozen_identity": frozen, "manifest_source_files_validation": manifest_files,
            "checkpoint_source_file_validations": validations,
            "round_plan_sha256": (plan_identity or {}).get("sha256"),
            "scheduler_plan_sha256": scheduler.get("plan_sha256"),
            "round_plan_file_matches_scheduler_sha256": observed_equal((plan_identity or {}).get("sha256"), scheduler.get("plan_sha256")),
            "scope": "source roots explicitly recorded by this scheduler; no earlier frozen-directory fallback"}


def build_report(round_path: Path, *, generated_unix=None) -> dict:
    report = base.build_report(round_path, generated_unix=generated_unix)
    round_path = Path(report["round_directory"])
    plan, scheduler = report["plan"], report["scheduler_record"]
    planned = {task["id"]: task for task in plan.get("tasks", [])}
    scheduled = {task["id"]: task for task in scheduler.get("tasks", [])}
    contexts, catalog, tasks = {}, {}, []
    for item in report["tasks"]:
        task, context = summarize_task(item, planned.get(item["id"]), report["read_errors"], catalog)
        tasks.append(task)
        contexts[task["id"]] = context
    phases = {phase["id"]: phase for phase in plan.get("phases", [])}
    report.update(schema="tide-pilot-round-review-v2", reporter_file_identity=record_identity(Path(__file__)),
                  reporter_dependencies={name: record_identity(Path(__file__).with_name(name))
                                         for name in ("report.py", "selection.py")},
                  limitations=LIMITATIONS, tasks=tasks,
                  phases=phase_summaries(tasks, contexts, plan, scheduler),
                  paired_comparisons=paired_comparisons(tasks, contexts, phases),
                  token_exposure=token_totals(tasks), source_identity_catalog=catalog,
                  memory_guard=memory_summary(round_path, scheduler, report["read_errors"]),
                  selection=selection_summary(tasks, contexts, scheduled, round_path),
                  source_provenance=source_review(round_path, scheduler, catalog, report["read_errors"]))
    report["worker_concurrency"] = report.pop("concurrency")
    report["strict_recovery_counts"] = {"qualified_scope": sum(task["recovery"]["strict_scope_qualified"] for task in tasks),
                                         "qualified_at_least_eight_steps": sum(task["recovery"]["eight_step_qualified"] for task in tasks)}
    report["mechanism_observations"] = base.mechanism_observations(
        [task for task in tasks if task["exposure_kind"] != "cost"])
    report["plan_summary"] = {key: value for key, value in plan.items() if key != "tasks"}
    report["scheduler_summary"] = {key: scheduler.get(key) for key in
                                   ("schema_version", "status", "stop_reason", "signal", "started_unix", "updated_unix",
                                    "deadline_unix", "dispatch_deadline_unix", "effective_cpu_affinity", "foundation_source",
                                    "foundation_source_requested", "orphan_processes", "all_task_processes_terminated")}
    for key in ("plan", "scheduler_record", "data_manifest", "source_input_identity", "source_identity_file"):
        report.pop(key, None)
    for task in tasks:
        observed = report["memory_guard"]["task_sampled_peak_rss_bytes"].get(task["id"])
        task["resources"]["memory_log_sampled_peak_rss_bytes"] = observed
    report["scheduler_snapshot_changed_during_read"] |= (
        report["scheduler_file_identity"] != record_identity(round_path / "scheduler.json"))
    report["review_ready"] = (report["scheduler_terminal_observed"] and report["all_task_processes_terminated"] is True
                              and not report["scheduler_snapshot_changed_during_read"])
    return report


def render_markdown(report: dict) -> str:
    fmt, cell = base.fmt, base.cell
    tokens = report["token_exposure"]
    lines = ["# 第二轮试运行结果", "",
             f"任务 {report['reported_task_count']}/{report['planned_task_count']}；调度状态 `{report['scheduler_status']}`；"
             f"所有任务进程终止：`{report['all_task_processes_terminated']}`。",
             f"主训练曝光：学习 {fmt(tokens['learning']['tokens'], 0)} token，成本计算 {fmt(tokens['cost']['tokens'], 0)} token。"
             "缺测保留 unknown，恢复参考与重放不计入学习曝光。", "", "## 阶段与资源", "",
             "| 阶段 | 状态 | 秒 | 任务 | CPU/NPU 训练峰值 |", "|---|---|---:|---:|---:|"]
    for phase in report["phases"]:
        peaks = phase["training_interval_peak"]
        lines.append(f"| {cell(phase['id'])} | {cell(phase['status'])} | {fmt(phase['wall_seconds'])} | {phase['task_count']} | "
                     f"{fmt(peaks['cpu']['peak'], 0)}/{fmt(peaks['npu']['peak'], 0)} |")
    memory = report["memory_guard"]
    lines.extend(["", f"进程树采样 RSS 峰值 {fmt(memory['sampled_peak_rss_bytes'] / 1024**3 if memory['sampled_peak_rss_bytes'] is not None else None, 2)} GiB；"
                  f"实际扫描间隔中位数/最大值 {fmt(memory['actual_scan_interval_seconds']['median'], 3)}/{fmt(memory['actual_scan_interval_seconds']['max'], 3)} 秒。"
                  f"看守方式 `{memory.get('enforcement')}`，内核内存硬上限建立状态 `{memory.get('kernel_memory_limit_established')}`。",
                  "", "## 配对并发观察", "",
                  "比值为相同测量前缀的并发吞吐/独立吞吐；计划负载不足仍明确列出实际观察范围。",
                  "", "| 组 / 并发任务 | 共同/测量步 | 比值 | 训练并发峰值 | 计划负载覆盖 | 资格 |",
                  "|---|---:|---:|---:|---:|---|"])
    for pair in report["paired_comparisons"]:
        prefix, overlap = pair.get("prefix", {}), pair.get("training_overlap", {})
        fraction = overlap.get("planned_concurrency_fraction")
        lines.append(f"| {cell(pair['comparison_group'])} / {cell(pair.get('concurrent_id'))} | "
                     f"{fmt(prefix.get('common_step_count'), 0)}/{fmt(prefix.get('measured_prefix_steps'), 0)} | "
                     f"{fmt(pair.get('throughput_ratio'), 3)} | {fmt(overlap.get('peak_simultaneous_training_tasks'), 0)} | "
                     f"{fmt(fraction * 100 if fraction is not None else None)}% | {cell(pair['status'])}; {cell(pair.get('target_load_status'))} |")
    selection = report["selection"]
    lines.extend(["", "## 学习率筛选与接续", "",
                  f"声明父任务 {selection['parent_count']} 个、接续任务 {selection['continuation_count']} 个；"
                  f"观察到 selection.json {selection['selection_records_observed']} 份。", "",
                  "| 模型 | LR | 种子 | 全种子平均 NLL | 全部完成 / 严格 8 步 |", "|---|---:|---|---:|---|"])
    for family in selection["families"]:
        for group in family["lr_groups"]:
            lines.append(f"| {cell(family['family'])} | {group['lr']:g} | {cell(group['seeds'])} | "
                         f"{fmt(group['mean_nll_all_declared_seeds'], 5)} | {group['all_completed']} / {group['all_strict_eight_step_qualified']} |")
    lines.extend(["", "| 接续任务 | 父任务 | 获选 LR / 种子均值 NLL | 全局范围 | 新增步 | 严格 8 步 |", "|---|---|---:|---|---:|---|"])
    for child in selection["continuations"]:
        training, winner = child["training"], child.get("winner") or {}
        lines.append(f"| {cell(child['id'])} | {cell(child['selected_parent'])} | {fmt(winner.get('lr'), 5)} / "
                     f"{fmt(winner.get('mean_validation_nll'), 5)} | {cell(training.get('from_step'))}→{cell(training.get('to_step'))} | "
                     f"{cell(training.get('steps'))} | {child['strict_eight_step_recovery']} |")
    lines.extend(["", "## 每任务记录", "",
                  "| 任务 | 用途 | 状态 | token/s | worker 秒 | 段步 / 全局终点 | RSS GiB / NPU GiB | 恢复步数 / 严格 |",
                  "|---|---|---|---:|---:|---:|---:|---|"])
    for task in report["tasks"]:
        training = task.get("training") or {}
        rss = task["resources"].get("memory_log_sampled_peak_rss_bytes")
        npu = (task["memory"].get("training") or {}).get("npu_peak_allocated_bytes")
        recovery = task["recovery"]
        lines.append(f"| {cell(task['id'])} | {cell(task['exposure_kind'])} | {cell(task['execution']['status'])} | "
                     f"{fmt(training.get('tokens_per_second'))} | {fmt(task['timing']['worker_execution_seconds'])} | "
                     f"{cell(training.get('steps'))}/{cell(training.get('global_steps'))} | "
                     f"{fmt(rss / 1024**3 if rss is not None else None, 2)}/{fmt(npu / 1024**3 if npu is not None else None, 2)} | "
                     f"{cell(recovery['recorded_steps'])}/{recovery['strict_scope_qualified']} |")
    lines.extend(["", "## 解释范围", "", *["- " + text for text in report["limitations"]], "",
                  "JSON 包含每步恢复判据、输入身份、解析记录、选择复算及逐任务测量；长日志只保留原路径、SHA256 和汇总证据。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, required=True, dest="round_path")
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/pilot-round2-20261119.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/pilot-round2-results.md"))
    args = parser.parse_args(argv)
    report = build_report(args.round_path)
    if not report["review_ready"]:
        raise ValueError("final review requires terminal scheduler, terminated processes and a stable scheduler snapshot")
    markdown = (args.markdown if args.markdown.is_absolute() else ROOT / args.markdown).resolve()
    if not markdown.is_relative_to(ROOT / "docs") or markdown.suffix != ".md":
        raise ValueError("Markdown review must be an .md file within repository docs/")
    from scripts.review_io import write_review_json
    write_review_json(args.output, report, root=ROOT)
    base.atomic_text(markdown, render_markdown(report))
    print(json.dumps({"reported_tasks": report["reported_task_count"], "scheduler_status": report["scheduler_status"],
                      "strict_recovery_counts": report["strict_recovery_counts"],
                      "json": str(args.output), "markdown": str(markdown)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
