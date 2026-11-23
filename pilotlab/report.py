"""Review a finite pilot from its own records, without starting any workload.

Missing observations remain unknown.  Process completion is never promoted to
numerical, discrete-semantic, learning-quality, or recovery qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "completed_with_failures", "completed_with_limits", "failed", "budget_exhausted"}
LIMITATIONS = [
    "本轮没有独占运行与并发运行的配对比较；重叠区间证明同时执行的事实，不能证明并发无吞吐损失或同卡并发能力。",
    "吞吐来自各任务实际配置与并发环境；不同参数量、batch、序列长度和算子路径不可据此直接排名架构效率。",
    "验证数据是 WikiText 上游训练文章的固定试运行留出；每次仅有少量样本，验证损失变化不构成模型质量排名。",
    "执行完成、浮点容差、结构/整数检查、checkpoint 数据、输入流与 RNG 恢复分别报告；缺失阶段保持 unknown。",
    "已保存的恢复比较仅覆盖记录的 logits、梯度、参数、优化器及私有状态；不外推全部组合、尺度或后端。",
    "因果探针仅在初始化、空私有状态下使用 1 个 16-token 样本，改动后 8 个 token 并比较前 8 个 logits；不认证训练后或其他状态、长度及 batch 的因果性。",
    "恢复梯度在全局范数裁剪后捕获；其一致性不等同于未裁剪梯度一致性。",
    "参数变化与最后一步非 None 梯度不能证明每个模块有效学习；参数变化可能包含 AdamW weight decay，零梯度与未连通梯度也需分别解释。",
    "活动量来自 B=1、空初始状态探针的聚合计数；它不认证完整离散轨迹或训练全程活动量。",
    "当前 resume-verify 仅检查新进程的一次实际优化器更新及私有状态延续，尚非长程断点任务队列功能。",
    "训练前失败的任务没有可测训练吞吐，也没有恢复资格；缺测保留 unknown，不作为 0 吞吐参与统计。",
    "训练吞吐按 worker 的测量边界统计；推理为驻留模型、包含任务头的真实文本 teacher-forced 继续，排除分词与主机传输。",
]


def file_identity(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_record(path: Path, errors: list[dict[str, str]]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, ValueError) as exc:
        errors.append({"path": str(path), "error": str(exc)})
        return None


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def timestamp(value: Any) -> float | None:
    numeric = number(value)
    if numeric is not None:
        return numeric
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return None


def difference(end: Any, start: Any) -> float | None:
    end, start = timestamp(end), timestamp(start)
    return end - start if end is not None and start is not None and end >= start else None


def compare_classification(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {"floating": "unknown", "structure": "unknown", "integer": "unknown", "exact": "unknown"}
    floating = ("nonfinite" if record.get("finite") is False else
                "within_tolerance" if record.get("within_tolerance") is True else
                "not_isolated_from_discrete_failure" if record.get("within_tolerance") is False
                and (record.get("structure_equal") is False or record.get("integer_equal") is False) else
                "outside_tolerance" if record.get("within_tolerance") is False
                and record.get("structure_equal") is True and record.get("integer_equal") is True else
                "comparison_failed_unattributed" if record.get("within_tolerance") is False else "unknown")

    def equality(name: str) -> str:
        return "equal" if record.get(name) is True else "unequal" if record.get(name) is False else "unknown"

    return {"floating": floating, "structure": equality("structure_equal"),
            "integer": equality("integer_equal"), "exact": equality("exact"),
            "atol": record.get("atol"), "rtol": record.get("rtol"),
            "max_abs": record.get("max_abs"), "fields": record.get("fields"),
            "failed_field_count": record.get("failed_field_count"),
            "failed_fields": record.get("failed_fields", [])}


def physical_mapping(result: dict, process: dict, queue: dict) -> dict[str, Any]:
    visibility = result.get("physical_visibility")
    ids = None
    if isinstance(visibility, str) and visibility.strip():
        try:
            ids = [int(item.strip()) for item in visibility.split(",")]
        except ValueError:
            pass
    effective_queue = queue or process.get("queue", {})
    if ids is None and isinstance(effective_queue, dict):
        ids = effective_queue.get("physical_ids")
        if ids is None and "physical_id" in effective_queue:
            ids = [effective_queue["physical_id"]]
    return {"physical_ids": ids, "logical_device": result.get("device"),
            "physical_visibility": visibility,
            "queue_mapping": {key: effective_queue.get(key) for key in
                              ("physical_ids", "logical_devices", "physical_id", "logical_device")}
            if isinstance(effective_queue, dict) else {}}


def execution_failure_stage(result: dict[str, Any]) -> str | None:
    """Identify only phases explicitly named by the recorded traceback."""
    error = result.get("error")
    if not isinstance(error, dict):
        return None
    trace = str(error.get("traceback", ""))
    if 'result["validation_initial"]' in trace or "result['validation_initial']" in trace:
        return "initial_validation_before_training"
    if 'result["causality"]' in trace or "result['causality']" in trace:
        return "initial_causality_probe_before_training"
    if 'result["validation_final"]' in trace or "result['validation_final']" in trace:
        return "final_validation"
    if 'result["inference"]' in trace or "result['inference']" in trace:
        return "inference_measurement"
    return "unknown"


def task_summary(task: dict, scheduler_record: dict, process: dict, result: dict,
                 queue: dict, worker: dict, round_start: Any, generated: float) -> dict[str, Any]:
    model = task.get("model", {})
    config = model.get("config", {}) if model.get("family") == "tide" else model
    backend = task.get("backend", scheduler_record.get("backend", "unknown"))
    started = timestamp(result.get("started_unix", worker.get("started_unix", process.get("worker_started_unix"))))
    ended = timestamp(result.get("ended_unix", process.get("worker_ended_unix")))
    launched = timestamp(process.get("process_started_unix"))
    admitted = timestamp(process.get("queue_admitted_unix"))
    status = process.get("status", scheduler_record.get("status", result.get("status", "unknown")))
    training, inference = result.get("training", {}), result.get("inference", {})
    recovery = result.get("recovery", {})
    comparisons = recovery.get("comparisons", {})
    initial, final = result.get("validation_initial", {}), result.get("validation_final", {})
    nll_before, nll_after = number(initial.get("nll")), number(final.get("nll"))
    continued = recovery.get("continued_private_state")
    checks = {name: compare_classification(value) for name, value in comparisons.items()}
    causality = compare_classification(result.get("causality"))
    error = result.get("error", {})
    return {
        "id": task.get("id", scheduler_record.get("id")), "backend": backend,
        "workload": {"batch_size": task.get("batch_size"), "sequence_length": task.get("sequence_length"),
                     "d": config.get("d"), "family": model.get("family"), "config": config,
                     "cpu_threads_requested": task.get("cpu_threads"), "cpu_threads_actual": result.get("cpu_threads"),
                     "cpu_affinity": process.get("cpu_affinity", worker.get("affinity")),
                     "precision": result.get("precision", task.get("precision")),
                     "seed": task.get("seed"), "learning_rate": task.get("lr"),
                     "readout_policy": result.get("readout_policy"),
                     "graph_nodes": task.get("graph_nodes"), "graph_regions": task.get("graph_regions")},
        "execution": {"status": status, "worker_status": result.get("status", "unknown"),
                      "return_code": process.get("return_code"), "reason": process.get("outcome_reason", scheduler_record.get("outcome_reason")),
                      "termination_reason": process.get("termination_reason"),
                      "descendants_terminated": process.get("descendants_terminated"),
                      "failure_stage": execution_failure_stage(result),
                      "error": error or None},
        "device_mapping": physical_mapping(result, process, queue),
        "timing": {"worker_started_unix": started, "worker_ended_unix": ended,
                   "worker_end_observed": ended is not None,
                   "ongoing_observed_until_unix": generated if started is not None and ended is None and status in {"running", "queued"} else None,
                   "process_started_unix": launched, "process_ended_unix": process.get("process_ended_unix"),
                   "round_pending_seconds": difference(launched, round_start),
                   "device_queue_seconds": difference(admitted, launched) if backend == "npu" else 0.0,
                   "launch_to_worker_seconds": difference(started, launched),
                   "worker_execution_seconds": difference(ended, started),
                   "initialization_seconds": result.get("initialization_seconds"),
                   "timing_note": "device_queue uses observed admission; launch_to_worker includes waiting and interpreter startup; missing admission is unknown"},
        "actual_parameters": result.get("parameters"),
        "training": training or None,
        "inference": inference or None,
        "memory": {"training": result.get("training_memory"), "final": result.get("memory_final"),
                   "restored_process": recovery.get("fresh_pid_result", {}).get("memory")},
        "checkpoint": result.get("checkpoint"), "parameter_updates": result.get("parameter_updates"),
        "parameter_updates_interpretation": "parameter changes may include AdamW decay; last-step non-None gradients do not establish effective learning of every module",
        "work_observations": {"initial": result.get("work_initial"), "final": result.get("work_final"),
                              "scope": "aggregate B=1 cold-state probe counts; not a complete discrete trajectory or all training work"},
        "validation_probe": {"initial": initial or None, "final": final or None,
                             "nll_delta_final_minus_initial": nll_after - nll_before if nll_before is not None and nll_after is not None else None,
                             "same_probe_token_count": initial.get("tokens") == final.get("tokens")
                             if initial.get("tokens") is not None and final.get("tokens") is not None else None,
                             "interpretation": "small fixed pilot article sample; not a quality ranking"},
        "checks": {"causality": causality, "recovery_comparisons": checks,
                   "causality_test_design": {"model_stage": "initialization", "private_state": "empty",
                                             "batch_size": 1, "sequence_length": 16,
                                             "perturbed_positions": [8, 16], "compared_positions": [0, 8],
                                             "observed_result_available": isinstance(result.get("causality"), dict)},
                   "recovery_gradient_capture": "after global-norm clipping; no un-clipped-gradient equivalence claim",
                   "independent_formula_checks": "unknown_not_recorded_per_task",
                   "full_discrete_trajectory_equivalence": "unknown_not_recorded_per_task"},
        "recovery": {"status": recovery.get("status", "unknown"),
                     "test_scope": "fresh interpreter, one actual optimizer update and private-state continuation; long-running queue resume not validated",
                     "saved_data_exact": recovery.get("saved_data_exact"),
                     "next_input_exact": recovery.get("next_input_exact"),
                     "rng_after_exact": recovery.get("rng_after_exact"),
                     "continued_private_state": continued,
                     "from_step": recovery.get("from_step"), "to_step": recovery.get("to_step"),
                     "seconds": recovery.get("seconds"), "return_code": recovery.get("returncode"),
                     "all_recorded_comparisons_exact": all(v.get("exact") is True for v in comparisons.values()) if comparisons else None,
                     "all_recorded_comparisons_within_tolerance": all(v.get("within_tolerance") is True for v in comparisons.values()) if comparisons else None,
                     "all_recorded_structures_equal": all(v.get("structure_equal") is True for v in comparisons.values()) if comparisons else None,
                     "all_recorded_integers_equal": all(v.get("integer_equal") is True for v in comparisons.values()) if comparisons else None,
                     "comparisons": comparisons, "raw_record": recovery or None},
    }


def interval_peak(intervals: list[tuple[float, float]]) -> dict[str, Any]:
    events = [(start, 1) for start, end in intervals if end > start]
    events += [(end, -1) for start, end in intervals if end > start]
    active = peak = 0
    at = None
    for when, delta in sorted(events, key=lambda row: (row[0], row[1])):
        active += delta
        if active > peak:
            peak, at = active, when
    return {"peak": peak, "first_peak_unix": at,
            "interval_count": sum(end > start for start, end in intervals)}


def distribution(values: list[Any]) -> dict[str, Any]:
    observed = [x for value in values if (x := number(value)) is not None]
    return {"observations": len(observed), "min": min(observed) if observed else None,
            "median": statistics.median(observed) if observed else None,
            "max": max(observed) if observed else None}


def failures_for(task: dict) -> list[dict[str, Any]]:
    failures = []
    identity = task["id"]
    execution = task["execution"]
    if execution["status"] == "failed" or execution["worker_status"] == "failed":
        error = execution.get("error") or {}
        failures.append({"task_id": identity, "category": "execution",
                         "message": str(error.get("message", execution.get("reason", "recorded execution failure")))[:1200],
                         "error_type": error.get("type"), "stage": execution.get("failure_stage", "unknown")})
    for phase, checks in [("causality", {"logits": task["checks"]["causality"]}),
                          ("recovery", task["checks"]["recovery_comparisons"])]:
        for field, check in checks.items():
            for kind in ("floating", "structure", "integer"):
                state = check.get(kind)
                if state in {"outside_tolerance", "nonfinite", "unequal"}:
                    failures.append({"task_id": identity, "category": f"{phase}_{kind}", "field": field,
                                     "message": f"{state}; max_abs={check.get('max_abs')}; failed_fields={check.get('failed_fields')}",
                                     "max_abs": check.get("max_abs"), "failed_field_count": check.get("failed_field_count")})
    recovery = task["recovery"]
    if recovery["status"] == "failed":
        failures.append({"task_id": identity, "category": "recovery_execution",
                         "message": f"fresh-process recovery failed; return_code={recovery.get('return_code')}"})
    for field in ("saved_data_exact", "next_input_exact", "rng_after_exact"):
        if recovery.get(field) is False:
            failures.append({"task_id": identity, "category": "recovery_" + field, "message": f"{field}=false"})
    return failures


def build_report(round_path: Path, *, generated_unix: float | None = None) -> dict[str, Any]:
    round_path = round_path.resolve()
    generated = time.time() if generated_unix is None else generated_unix
    errors: list[dict[str, str]] = []
    scheduler_path = round_path / "scheduler.json"
    scheduler_identity_before = file_identity(scheduler_path)
    scheduler = read_record(scheduler_path, errors) or {}
    plan = read_record(round_path / "plan.json", errors) or {}
    planned = {task["id"]: task for task in plan.get("tasks", [])}
    scheduled = {task["id"]: task for task in scheduler.get("tasks", [])}
    task_ids = list(dict.fromkeys([*planned, *scheduled]))
    directory = round_path / "tasks"
    if directory.is_dir():
        task_ids.extend(path.name for path in sorted(directory.iterdir())
                        if path.is_dir() and path.name not in task_ids)
    tasks = []
    for identity in task_ids:
        path = directory / identity
        result = read_record(path / "result.json", errors) or {}
        actual_task = read_record(path / "task.json", errors) or result.get("task") or planned.get(identity) or {"id": identity}
        process = read_record(path / "process.json", errors) or {}
        queue = read_record(path / "queue.json", errors) or {}
        worker = read_record(path / "worker.json", errors) or {}
        item = task_summary(actual_task, scheduled.get(identity, {}), process, result, queue, worker,
                            scheduler.get("started_unix"), generated)
        item.update(directory=str(path), planned=identity in planned,
                    configuration_matches_plan=actual_task == planned[identity] if identity in planned else None,
                    source_record_identities={name: file_identity(path / name) for name in
                                              ("task.json", "result.json", "process.json", "queue.json", "worker.json")},
                    raw_records={"task": actual_task, "result": result or None, "process": process or None,
                                 "scheduler_task": scheduled.get(identity), "queue": queue or None, "worker": worker or None})
        tasks.append(item)
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    censored = []
    for task in tasks:
        timing = task["timing"]
        start, end = timing["worker_started_unix"], timing["worker_ended_unix"]
        if start is not None and end is None:
            end = timing["ongoing_observed_until_unix"]
            if end is not None:
                censored.append(task["id"])
        if start is not None and end is not None and end > start:
            intervals[task["backend"]].append((start, end))
            intervals["all"].append((start, end))
    concurrent = {backend: interval_peak(intervals[backend]) for backend in ("cpu", "npu", "all")}
    concurrent.update(censored_running_tasks=censored,
                      scope="observed main worker intervals including child restore; interval end is exclusive; not a contention or isolation comparison")
    by_backend = {}
    for backend in sorted({task["backend"] for task in tasks}):
        group = [task for task in tasks if task["backend"] == backend]
        by_backend[backend] = {
            "task_count": len(group), "execution_counts": dict(Counter(task["execution"]["status"] for task in group)),
            "recovery_counts": dict(Counter(task["recovery"]["status"] for task in group)),
            "training_tokens_sum": sum((task.get("training") or {}).get("tokens", 0) for task in group),
            "training_records_observed": sum(task.get("training") is not None for task in group),
            "training_records_unknown": sum(task.get("training") is None for task in group),
            "training_tokens_sum_scope": "sum of recorded training tokens; missing observations are not measured zero-token runs",
            "recorded_worker_seconds_sum": sum(task["timing"]["worker_execution_seconds"] or 0 for task in group),
            "initialization_seconds": distribution([task["timing"]["initialization_seconds"] for task in group]),
            "launch_to_worker_seconds": distribution([task["timing"]["launch_to_worker_seconds"] for task in group]),
            "device_queue_seconds": distribution([task["timing"]["device_queue_seconds"] for task in group]),
            "checkpoint_bytes_sum": sum((task["checkpoint"] or {}).get("bytes", 0) for task in group),
            "peak_simultaneous_workers": concurrent.get(backend, {}).get("peak"),
        }
    input_identity = None
    identity_path = None
    candidates = [round_path.parent / "frozen" / "identity.json"]
    if scheduler.get("source"):
        candidates.append(Path(scheduler["source"]) / "identity.json")
    for path in dict.fromkeys(candidates):
        value = read_record(path, errors)
        if value is not None:
            input_identity, identity_path = value, path
            break
    data_manifest = None
    data_identity = None
    if scheduler.get("data"):
        data_path = Path(scheduler["data"]) / "manifest.json"
        data_manifest, data_identity = read_record(data_path, errors), file_identity(data_path)
    failures = [failure for task in tasks for failure in failures_for(task)]
    no_result = [task["id"] for task in tasks if task["raw_records"]["result"] is None]
    scheduler_identity_after = file_identity(scheduler_path)
    stage_counts = {}
    for stage in ("training", "inference", "checkpoint", "parameter_updates"):
        stage_counts[stage] = {"recorded": sum(task.get(stage) is not None for task in tasks),
                               "unknown": sum(task.get(stage) is None for task in tasks)}
    return {
        "schema": "tide-pilot-round-review-v1", "generated_unix": generated,
        "generated_utc": datetime.fromtimestamp(generated, timezone.utc).isoformat(),
        "reporter_file_identity": file_identity(Path(__file__)),
        "round_directory": str(round_path), "round_id": plan.get("round_id", round_path.name),
        "scheduler_status": scheduler.get("status", "unknown"),
        "scheduler_terminal_observed": scheduler.get("status") in TERMINAL,
        "scheduler_snapshot_changed_during_read": scheduler_identity_before != scheduler_identity_after,
        "scheduler_started_unix": scheduler.get("started_unix"), "scheduler_updated_unix": scheduler.get("updated_unix"),
        "round_observed_seconds": difference(scheduler.get("updated_unix"), scheduler.get("started_unix")),
        "deadline_unix": scheduler.get("deadline_unix"),
        "all_task_processes_terminated": scheduler.get("all_task_processes_terminated"),
        "planned_task_count": len(planned), "reported_task_count": len(tasks),
        "execution_counts": dict(Counter(task["execution"]["status"] for task in tasks)),
        "recovery_counts": dict(Counter(task["recovery"]["status"] for task in tasks)),
        "failure_category_counts": dict(Counter(failure["category"] for failure in failures)),
        "execution_failure_stage_counts": dict(Counter(failure.get("stage", "unknown") for failure in failures
                                                       if failure["category"] == "execution")),
        "stage_observation_counts": stage_counts, "tasks_without_worker_result": no_result,
        "backend_summaries": by_backend, "concurrency": concurrent,
        "physical_ids_observed": sorted({physical for task in tasks for physical in
                                         (task["device_mapping"]["physical_ids"] or [])}),
        "storage": scheduler.get("storage"), "resource_slots": scheduler.get("slots"),
        "source_input_identity": input_identity,
        "source_identity_file": file_identity(identity_path) if identity_path is not None else None,
        "data_manifest": data_manifest, "data_manifest_identity": data_identity,
        "scheduler_file_identity": scheduler_identity_after,
        "plan_file_identity": file_identity(round_path / "plan.json"),
        "plan": plan, "scheduler_record": scheduler,
        "limitations": LIMITATIONS, "read_errors": errors, "failures": failures, "tasks": tasks,
    }


def fmt(value: Any, digits: int = 1) -> str:
    parsed = number(value)
    return f"{parsed:,.{digits}f}" if parsed is not None else "unknown"


def cell(value: Any) -> str:
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def mechanism_observations(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select review leads without changing execution or check classifications."""
    observations = []
    for task in tasks:
        updates = task.get("parameter_updates") or {}
        changed, total = number(updates.get("changed_tensor_count")), number(updates.get("total_tensor_count"))
        none_gradients = updates.get("none_gradient_parameters")
        if not isinstance(none_gradients, list):
            none_gradients = None
        work = task.get("work_observations") or {}
        empty = {phase: (work.get(phase) or {}).get("empty_output_positions") for phase in ("initial", "final")}
        counts = {phase: (work.get(phase) or {}).get("output_positions") for phase in ("initial", "final")}
        reasons = []
        if changed is not None and total is not None and changed < total:
            reasons.append("some_parameter_tensors_unchanged")
        if none_gradients:
            reasons.append("last_update_none_gradients")
        if any(number(value) is not None and number(value) > 0 for value in empty.values()):
            reasons.append("selected_output_positions_empty")
        if reasons:
            observations.append({"task_id": task["id"], "reasons": reasons,
                                 "changed_tensor_count": updates.get("changed_tensor_count"),
                                 "total_tensor_count": updates.get("total_tensor_count"),
                                 "none_gradient_parameters": none_gradients,
                                 "empty_selected_output_positions": empty,
                                 "selected_output_positions": counts})
    return observations


def render_markdown(report: dict[str, Any]) -> str:
    count = report["reported_task_count"]
    lines = [
        f"# 第一轮试运行报告：{report['round_id']}", "",
        f"生成时间：{report['generated_utc']}。原始记录：`{report['round_directory']}`。", "",
        f"计划 {report['planned_task_count']} 个任务，报告覆盖 {count} 个；调度器状态 `{report['scheduler_status']}`，"
        f"观察时长 {fmt(report['round_observed_seconds'])} 秒，全部任务进程已终止：`{report['all_task_processes_terminated']}`。",
        f"执行状态：`{json.dumps(report['execution_counts'], ensure_ascii=False, sort_keys=True)}`。"
        f"恢复状态：`{json.dumps(report['recovery_counts'], ensure_ascii=False, sort_keys=True)}`。", "",
        "执行完成只表示该运行的执行结果。浮点误差、结构/整数、恢复数据与学习观察仍按各自记录判断。", "",
    ]
    if not report["scheduler_terminal_observed"] or report["scheduler_snapshot_changed_during_read"]:
        lines.extend(["此报告是运行中快照；调度器尚未确认终态或读取期间发生更新。", ""])
    lines.extend(["## 资源与并发", "",
                  "| 后端 | 任务数 | 最大同时执行 worker | 累计训练 token | 累计 worker 秒 | 启动耗时中位数/最大值（秒） |",
                  "|---|---:|---:|---:|---:|---:|"])
    for backend, summary in report["backend_summaries"].items():
        init = summary["initialization_seconds"]
        lines.append(f"| {backend} | {summary['task_count']} | {summary['peak_simultaneous_workers']} | "
                     f"{fmt(summary['training_tokens_sum'], 0)} | {fmt(summary['recorded_worker_seconds_sum'])} | "
                     f"{fmt(init['median'])} / {fmt(init['max'])} |")
    lines.extend(["", f"观察到的物理 NPU：`{report['physical_ids_observed']}`。"
                  f"跨后端合计最大同时执行 worker：{report['concurrency']['all']['peak']}。",
                  "worker 区间包括训练、评估、保存和子进程恢复。设备等待、解释器启动及实际执行在逐任务 JSON 中分别保留。",
                  "本轮没有独占与并发的配对测量；能够同时执行不等于没有争用，也不证明同卡并发能力。", "",
                  "## 全部任务", "",
                  "吞吐单位为有效文本 token/s。恢复 `within_tolerance` 表示记录的恢复检查在既定容差内；逐位一致标志另存 JSON。unknown 表示未记录。", "",
                  "| 任务 | 后端/物理卡 | 参数 M | B×L | 训练步 | 训练吞吐 | Prefill | Decode | 执行 | 恢复 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|---|"])
    for task in report["tasks"]:
        training, inference = task["training"] or {}, task["inference"] or {}
        workload, mapping = task["workload"], task["device_mapping"]
        device = task["backend"] + (":" + ",".join(map(str, mapping["physical_ids"])) if mapping["physical_ids"] else "")
        parameters = number(task["actual_parameters"])
        lines.append("| " + " | ".join([
            cell(task["id"]), cell(device), fmt(parameters / 1e6 if parameters is not None else None, 3),
            f"{workload['batch_size']}×{workload['sequence_length']}", fmt(training.get("steps"), 0),
            fmt(training.get("tokens_per_second")), fmt(inference.get("prefill_tokens_per_second")),
            fmt(inference.get("decode_tokens_per_second")), cell(task["execution"]["status"]),
            cell(task["recovery"]["status"]),
        ]) + " |")
    lines.extend(["", "## 资源成本和学习探针", "",
                  "以下峰值为各任务记录范围内的峰值。主进程及恢复进程分开保存，不能简单相加为同时占用。", "",
                  "| 任务 | 初始化秒 | 排队/启动秒 | Worker 秒 | 训练峰值 GiB（NPU / RSS） | Checkpoint MiB / 保存秒 | 验证 NLL 前→后（变化） |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for task in report["tasks"]:
        timing, memory, checkpoint = task["timing"], task["memory"]["training"] or {}, task["checkpoint"] or {}
        before, after = task["validation_probe"]["initial"] or {}, task["validation_probe"]["final"] or {}
        npu, rss, size = number(memory.get("npu_peak_allocated_bytes")), number(memory.get("cpu_rss_peak_bytes")), number(checkpoint.get("bytes"))
        delta = task["validation_probe"]["nll_delta_final_minus_initial"]
        lines.append("| " + " | ".join([
            cell(task["id"]), fmt(timing["initialization_seconds"]), fmt(timing["launch_to_worker_seconds"]),
            fmt(timing["worker_execution_seconds"]),
            f"{fmt(npu / 2**30 if npu is not None else None, 3)} / {fmt(rss / 2**30 if rss is not None else None, 3)}",
            f"{fmt(size / 2**20 if size is not None else None)} / {fmt(checkpoint.get('save_seconds'))}",
            f"{fmt(before.get('nll'), 4)}→{fmt(after.get('nll'), 4)} ({fmt(delta, 4)})",
        ]) + " |")
    lines.extend(["", "验证仅使用小型固定文章探针，训练 token 曝光与完成更新数也不同；表中损失变化不用于架构质量排名。", "",
                  "## 机制覆盖观察", "",
                  "以下仅列值得后续审阅的观察，不将其自动归为 bug、不可训练或执行失败。None 梯度只反映最后一个训练更新；"
                  "未变化参数不能证明不可训练，参数变化也可能包含 AdamW decay。活动计数来自 B=1 冷状态探针，不认证训练全程或完整离散轨迹。", ""])
    lines.extend(["参数变化记录只有总数，没有逐参数未变名单；末步 None 梯度名单不能当作整轮未变参数清单。", ""])
    observations = mechanism_observations(report["tasks"])
    if observations:
        lines.extend(["| 任务 | 发生变化/总参数张量 | 最后一步 None 梯度数 | 选定输出空位（初始 / 最终） |",
                      "|---|---:|---:|---|"])
        for observation in observations:
            empty, counts = observation["empty_selected_output_positions"], observation["selected_output_positions"]
            outputs = " / ".join(f"{fmt(empty[phase], 0)} of {fmt(counts[phase], 0)}" for phase in ("initial", "final"))
            none_count = len(observation["none_gradient_parameters"]) if observation["none_gradient_parameters"] is not None else None
            lines.append(f"| {cell(observation['task_id'])} | {fmt(observation['changed_tensor_count'], 0)} / "
                         f"{fmt(observation['total_tensor_count'], 0)} | {fmt(none_count, 0)} | {outputs} |")
        for observation in observations:
            if observation["none_gradient_parameters"]:
                names = ", ".join("`" + cell(name) + "`" for name in observation["none_gradient_parameters"])
                lines.extend(["", f"`{observation['task_id']}` 的最后一步 None 梯度：{names}。"])
    else:
        lines.append("已记录字段中没有上述观察；缺测字段仍为 unknown，这不构成所有模块有效学习的证明。")
    lines.extend(["", "## 失败与未完成检查", "",
                  f"失败类别计数（可重叠）：`{json.dumps(report['failure_category_counts'], ensure_ascii=False, sort_keys=True)}`。",
                  f"阶段覆盖：`{json.dumps(report['stage_observation_counts'], ensure_ascii=False, sort_keys=True)}`。", ""])
    if report["failures"]:
        lines.extend(["| 任务 | 类别 / 字段 | 原始信息摘录 |", "|---|---|---|"])
        for failure in report["failures"][:40]:
            category = failure["category"] + (" / " + failure["field"] if failure.get("field") else "")
            if failure.get("stage"):
                category += " / " + failure["stage"]
            lines.append(f"| {cell(failure['task_id'])} | {cell(category)} | {cell(failure['message'][:500])} |")
        if len(report["failures"]) > 40:
            lines.extend(["", f"此处展示前 40 项，全部 {len(report['failures'])} 项保留在 JSON。"])
    else:
        lines.append("已读取字段中未报告上述失败；缺失检查仍为 unknown，不据此宣称全部能力通过。")
    if report["tasks_without_worker_result"]:
        lines.extend(["", "缺少 worker 结果：`" + ", ".join(report["tasks_without_worker_result"]) + "`。"])
    if report["read_errors"]:
        lines.extend(["", "记录读取错误：`" + cell(json.dumps(report["read_errors"], ensure_ascii=False)) + "`。"])
    lines.extend(["", "## 解释范围", ""])
    lines.extend("- " + text for text in report["limitations"])
    lines.extend(["", "本报告的 JSON 保留每个任务原始配置、执行与恢复记录、各字段比较、输入身份及缺测信息，供下一轮排程与诊断使用。", ""])
    return "\n".join(lines)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, required=True, dest="round_path")
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/pilot-round1-20261119.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/pilot-round1-results.md"))
    args = parser.parse_args(argv)
    report = build_report(args.round_path)
    # The repository-required guard rejects workload paths before any write.
    from scripts.review_io import write_review_json
    write_review_json(args.output, report, root=ROOT)
    markdown = args.markdown if args.markdown.is_absolute() else ROOT / args.markdown
    markdown = markdown.resolve()
    if not markdown.is_relative_to(ROOT / "docs") or markdown.suffix != ".md":
        raise ValueError("Markdown review must be an .md file within repository docs/")
    atomic_text(markdown, render_markdown(report))
    print(json.dumps({"reported_tasks": report["reported_task_count"], "scheduler_status": report["scheduler_status"],
                      "execution_counts": report["execution_counts"], "recovery_counts": report["recovery_counts"],
                      "json": str(args.output), "markdown": str(markdown)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
