"""Finite preparation admission, capacity observations and real continuation.

No stage-A/B learning is submitted here.  This program uses the ordinary
cooperative queue and leaves any scientific failure closed for later repair.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

from .io import atomic_json, sha256
from .scheduler3 import Blocked, Supervisor, freeze_json, read

MI = 2**20
PREPARATION_TOKEN_LIMIT = 64 * MI


def task_processing_bound(task: dict) -> int:
    """Two evaluations, two causal probes and all declared training updates."""
    train = int(task.get("segment_target_tokens", task["target_tokens"]))
    return train + 2 * int(task.get("eval_tokens", 512)) + 32


def claim_token_budget(path: Path, run: str, amount: int, plan_sha: str) -> dict:
    """Reserve worst-case attempts before dispatch; completed claims never reset."""
    with Path(str(path) + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read(path)
        if ledger is None:
            raise ValueError("create the campaign budget immediately before first submission")
        claims = ledger.setdefault("preparation_token_reservations", {})
        requested = {"reserved_tokens": amount, "plan_sha256": plan_sha}
        existing = claims.get(run)
        if existing is not None:
            if any(existing.get(key) != value for key, value in requested.items()):
                raise ValueError("existing preparation token reservation changed")
            return existing
        if type(amount) is not int or amount <= 0:
            raise ValueError("positive preparation token reservation required")
        reserved = sum(value["reserved_tokens"] for value in claims.values())
        reserved += sum(value.get("preparation_reserved_tokens", 0)
                        for key, value in ledger.get("segments", {}).items() if key not in claims)
        if reserved + amount > PREPARATION_TOKEN_LIMIT:
            raise Blocked("preparation_processed_token_limit")
        claims[run] = dict(requested, created_unix=time.time(), status="reserved")
        ledger["preparation_max_processed_tokens"] = PREPARATION_TOKEN_LIMIT
        atomic_json(path, ledger)
        return claims[run]


def finish_token_accounting(path: Path, run: str, results: dict, status: str):
    # On failed attempts, counters may stop before a failing forward returns.
    # Keep the original upper-bound reservation charged instead of inventing
    # precise physical-token counts or reclaiming uncertain work.
    completed = sum(int(value.get("training", {}).get("segment_tokens", 0))
                    + int(value.get("evaluation", {}).get("tokens", 0)) for value in results.values())
    with Path(str(path) + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read(path)
        claim = ledger["preparation_token_reservations"][run]
        if completed > claim["reserved_tokens"]:
            raise RuntimeError("observed preparation exposure exceeds the reserved ceiling")
        claim.update(status=status, observed_completed_loss_tokens=completed, updated_unix=time.time(),
                     accounting_scope="successful optimizer/evaluation exposure; uncertain attempts retain full reservation")
        atomic_json(path, ledger)


def build_plan(prepared: dict) -> dict:
    """CPU/meta-only planning from the exact frozen catalog."""
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise ValueError("preparation planning requires TORCH_DEVICE_BACKEND_AUTOLOAD=0")
    from .catalog3 import exact_parameter_counts, generate_subjects, model_key
    catalog = generate_subjects()
    counts = {row["id"]: exact_parameter_counts(row["model"]) for row in catalog}
    groups = ("fc", "lattice", "exchange", "multiscale_settle", "multiscale_timed")
    representatives = {}
    for group in groups:
        representatives[group] = min((row for row in catalog if row["group"] == group),
            key=lambda row: (counts[row["id"]]["whole_model_parameters"], row["id"]))

    def specific(group, memory, variant):
        return next(row for row in catalog if row["group"] == group and row["model"]["config"]["d"] == 64
                    and row["model"]["config"]["memory"] == memory and row["coverage_variant"] == variant)

    representatives.update(fc_ssd=specific("fc", "S", "base"),
        exchange_x=specific("exchange", "X", "base"),
        exchange_shannon=specific("exchange", "X", "shannon"),
        exchange_pool=specific("exchange", "X", "pool"),
        largest=max(catalog, key=lambda row: (counts[row["id"]]["whole_model_parameters"], row["id"])))
    for family in ("transformer", "moe"):
        model = dict(family=family, d=64, layers=2, heads=2, ffn_width=128, window=128)
        if family == "moe":
            model.update(experts=4, top_k=2)
        representatives[family] = {"id": "preflight-" + family + "-" + model_key(model)[:10],
                                   "group": family, "model": model, "model_key": model_key(model)}

    def task(label, role, *, batch=1, length=128, threads=4, updates=6, purpose="mechanism_admission"):
        model = representatives[role]
        tokens = batch * length * updates
        return {"id": "pf-" + label, "phase": "preflight", "purpose": purpose,
            "representative_role": role, "model_id": model["id"], "model": copy.deepcopy(model["model"]),
            "source_identity": prepared["source_identity"], "seed": 20261120, "init_recipe": "default",
            "batch_size": batch, "sequence_length": length, "stream_chunks": 4, "cpu_threads": threads,
            "precision": "fp32", "device": "npu:0", "deterministic": False,
            "target_tokens": tokens, "segment_target_tokens": tokens, "schedule_tokens": 12 * MI,
            "warmup_tokens": batch * length, "lr": 3e-4, "lr_floor_ratio": 0.1,
            "weight_decay": 0.01, "grad_clip": 1.0,
            "auxiliary_loss_weight": 0.01 if role == "moe" else 1.0,
            "eval_tokens": 512, "eval_interval_tokens": 0, "health_interval_tokens": 2**30,
            "closing_reserve_seconds": 120, "checkpoint_interval_seconds": 120,
            "record_step_fingerprints": True, "capture_failure": purpose != "capacity_calibration",
            "measurement_scope": "finite concurrent admission; source workload differs across shape cases, not equivalent-speedup evidence"}

    core = [task(role, role) for role in representatives]
    shapes = []
    for role, batch, length, threads in (
        ("fc", 8, 128, 4), ("fc", 32, 128, 4), ("fc", 32, 128, 8),
        ("exchange_x", 4, 128, 4), ("exchange_x", 8, 128, 4), ("exchange_x", 8, 128, 8),
        ("multiscale_timed", 2, 128, 4), ("multiscale_timed", 1, 256, 4),
        ("transformer", 16, 128, 4), ("transformer", 64, 128, 4), ("transformer", 64, 256, 4),
        ("moe", 8, 128, 4), ("moe", 16, 256, 4)):
        item = task(f"shape-{role}-b{batch}-l{length}-t{threads}", role, batch=batch, length=length,
                    threads=threads, updates=4, purpose="capacity_calibration")
        item["prerequisite_task"] = "pf-" + role
        shapes.append(item)
    resumes = []
    for role in ("exchange_x", "transformer", "moe"):
        item = task("resume-" + role, role, updates=4, purpose="continuation_admission")
        item["resume_parent_id"] = "pf-" + role
        item["target_tokens"] += next(value["target_tokens"] for value in core if value["id"] == item["resume_parent_id"])
        resumes.append(item)
    for item in core + shapes + resumes:
        item["budget_tokens"] = task_processing_bound(item)
    upper = 2 * sum(task_processing_bound(value) for value in core + shapes + resumes)
    if upper > PREPARATION_TOKEN_LIMIT:
        raise ValueError("preparation matrix exceeds its worst-case token budget")
    return {"schema": "pilot-r3-preflight-plan-v1", "source_identity": prepared["source_identity"],
            "catalog_model_count": len(catalog), "representatives": representatives,
            "core_tasks": core, "shape_tasks": shapes, "resume_templates": resumes,
            "task_count_upper": len(core) + len(shapes) + len(resumes),
            "processed_token_reservation": upper, "attempts_per_task_upper": 2,
            "numeric_scope": "FP32 positive-LR mechanism/load admission; no strict cross-process numerical replay claim",
            "capacity_policy": "all representative mechanisms start B1; expansion requires measured memory headroom"}


def check_result(task: dict, result: dict, *, parent: dict | None = None) -> list[str]:
    reasons = []
    if result.get("status") != "completed" or result.get("training", {}).get("tokens") != task["target_tokens"]:
        reasons.append("execution_or_exposure_not_complete")
    if result.get("source_identity") != task["source_identity"]:
        reasons.append("source_identity_mismatch")
    if result.get("health", {}).get("eligible") is not True or result.get("health", {}).get("blocking_reasons"):
        reasons.append("health_not_qualified")
    if result.get("causality", {}).get("within_tolerance") is not True:
        reasons.append("causality_not_verified")
    if result.get("recovery", {}).get("checkpoint_load", {}).get("status") != "exact":
        reasons.append("checkpoint_load_not_exact")
    work = result.get("work", [])
    if not work or any(value.get("work", {}).get("activity", {}).get("qualified") is not True for value in work):
        reasons.append("actual_activity_not_qualified")
    memory = result.get("device_memory", {})
    if not (isinstance(memory.get("npu_peak_allocated_bytes"), int) and memory["npu_peak_allocated_bytes"] > 0
            and isinstance(memory.get("npu_total_memory_bytes"), int) and memory["npu_total_memory_bytes"] > 0
            and isinstance(memory.get("npu_free_memory_bytes"), int)
            and 0 <= memory["npu_free_memory_bytes"] <= memory["npu_total_memory_bytes"]):
        reasons.append("device_capacity_observation_missing")
    checkpoint = result.get("checkpoint") or {}
    path = checkpoint.get("path")
    if not path or not Path(path).is_file() or sha256(path) != checkpoint.get("sha256"):
        reasons.append("checkpoint_missing_or_changed")
    if parent is not None:
        resume, prior = result.get("resume", {}), parent["checkpoint"]
        expected = parent["training"]["tokens"]
        if (resume.get("path") != prior["path"] or resume.get("sha256") != prior["sha256"]
                or resume.get("from_tokens") != expected or result.get("training", {}).get("from_tokens") != expected
                or resume.get("source_task_id") != parent["id"] or resume.get("state_reused") is not True
                or resume.get("checkpoint_load", {}).get("status") != "exact"
                or resume.get("state_sha256") != prior["state_sha256"]):
            reasons.append("fresh_worker_continuation_chain_not_exact")
    return reasons


def capacity_admission(task: dict, parent: dict) -> dict:
    memory = parent.get("device_memory", {})
    peak, total = memory.get("npu_peak_allocated_bytes"), memory.get("npu_total_memory_bytes")
    free, reserved = memory.get("npu_free_memory_bytes"), memory.get("npu_reserved_bytes")
    if (not isinstance(peak, int) or not isinstance(total, int) or min(peak, total) <= 0
            or not isinstance(free, int) or not isinstance(reserved, int)
            or not (0 <= free <= total and 0 <= reserved <= total)):
        return {"admitted": False, "reason": "missing_device_memory_measurement"}
    # Parent probes are B1/L128. Scaling the entire peak (including fixed
    # parameters) overestimates simple activation growth, providing headroom.
    # Free HBM is a device-wide observation while the parent still owns its
    # allocator cache. Only that cache is added back, not unmeasured context
    # memory or allocations belonging to other processes.
    usable = min(total, free + reserved)
    scale = max(1.0, task["batch_size"] * task["sequence_length"] / 128)
    estimate = math.ceil(peak * scale)
    return {"admitted": estimate <= int(0.7 * usable), "estimated_peak_bytes": estimate,
            "measured_b1_peak_bytes": peak, "device_total_bytes": total, "safety_fraction": 0.7,
            "device_free_at_parent_sample_bytes": free, "parent_allocator_reserved_bytes": reserved,
            "estimated_post_parent_release_usable_bytes": usable,
            "reason": "within_conservative_headroom" if estimate <= 0.7 * usable else "predicted_capacity_limit",
            "scope": "admission estimate; actual peak still measured and OOM remains a capacity failure"}


def _measurement(task, result, record):
    attempt = Path(record["attempts"][-1]["path"])
    rows = [json.loads(line) for line in (attempt / "steps.jsonl").read_text().splitlines()]
    plain = [row for row in rows if not row.get("sampled_diagnostics") and row.get("seconds", 0) > 0]
    quantum = task["batch_size"] * task["sequence_length"]
    throughput = quantum * len(plain) / sum(row["seconds"] for row in plain) if plain else None
    return {"id": task["id"], "model_id": task["model_id"], "role": task["representative_role"],
        "batch_size": task["batch_size"], "sequence_length": task["sequence_length"], "cpu_threads": task["cpu_threads"],
        "unprofiled_step_count": len(plain), "unprofiled_step_tokens_per_second": throughput,
        "training": result["training"], "device_memory": result["device_memory"],
        "checkpoint_bytes": result["checkpoint"]["size_bytes"], "device_seconds": record["device_seconds"],
        "actual_activity": result["work"][-1]["work"]["activity"], "result_path": record["result_path"],
        "result_sha256": record["result_sha256"], "concurrency_scope": task["measurement_scope"]}


def run(prepared: dict):
    from .prepare3 import verify_data
    from .worker3 import validate_source_identity
    validate_source_identity({"source_identity": prepared["source_identity"]}, Path(prepared["data"]))
    plan = build_plan(prepared)
    output = Path(prepared["output"])
    output.mkdir(parents=True, exist_ok=True)
    plan_path = freeze_json(output / "preflight-plan.json", plan)
    ledger_path = Path(prepared["budget_ledger"])
    ledger = read(ledger_path)
    if ledger is None:
        raise ValueError("campaign budget must exist before submitting preparation")
    claim_token_budget(ledger_path, str(output), plan["processed_token_reservation"], sha256(plan_path))
    config = {key: prepared[key] for key in ("python", "source", "foundation", "data", "run_id", "max_workers",
                                             "storage_reserve_bytes", "max_rss_bytes", "budget_ledger")}
    config.update(deadline_unix=ledger["deadline_unix"], preparation=True)
    supervisor = Supervisor(config, output)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    results, tasks, admissions, rejections, rows = {}, {}, {}, {}, []
    report = {"schema": "pilot-r3-preflight-report-v1", "source_identity": prepared["source_identity"],
              "plan_path": str(plan_path), "plan_sha256": sha256(plan_path), "status": "running",
              "stage_b_gate": "closed", "started_unix": time.time(), "qualification_scope": plan["numeric_scope"]}

    def collect(stage_tasks, observed, *, parents=None, allow_capacity=False):
        for task in stage_tasks:
            result = observed[task["id"]]
            results[task["id"]], tasks[task["id"]] = result, task
            if allow_capacity and result.get("status") == "failed" and result.get("failure_class") == "capacity_error":
                rejections[task["id"]] = {"reason": "measured_allocator_oom", "result_path": supervisor.state["tasks"][task["id"]]["result_path"]}
                continue
            reasons = check_result(task, result, parent=(parents or {}).get(task["id"]))
            if reasons:
                raise Blocked(task["id"] + ": " + ", ".join(reasons))
            record = supervisor.state["tasks"][task["id"]]
            rows.append(_measurement(task, result, record))
            if task.get("resume_checkpoint"):
                first = json.loads((Path(record["attempts"][-1]["path"]) / "steps.jsonl").read_text().splitlines()[0])
                if first.get("state_reset") is not False:
                    raise Blocked(task["id"] + ": first continuation update discarded private state")

    try:
        observed = supervisor.run("preflight-core", plan["core_tasks"])
        collect(plan["core_tasks"], observed)
        shape_tasks = []
        for task in plan["shape_tasks"]:
            admission = capacity_admission(task, results[task["prerequisite_task"]])
            admissions[task["id"]] = admission
            if admission["reason"] == "missing_device_memory_measurement":
                raise Blocked("capacity expansion lacks device memory evidence")
            if admission["admitted"]:
                shape_tasks.append(task)
        if shape_tasks:
            observed = supervisor.run("preflight-shapes", shape_tasks, allowed_failures=("capacity_error",))
            collect(shape_tasks, observed, allow_capacity=True)
        resume_tasks, parents = [], {}
        for template in plan["resume_templates"]:
            parent = results[template["resume_parent_id"]]
            task = dict(template, resume_checkpoint=parent["checkpoint"]["path"],
                        resume_checkpoint_sha256=parent["checkpoint"]["sha256"])
            resume_tasks.append(task)
            parents[task["id"]] = parent
        observed = supervisor.run("preflight-real-continuation", resume_tasks)
        collect(resume_tasks, observed, parents=parents)
        validate_source_identity({"source_identity": prepared["source_identity"]}, Path(prepared["data"]))
        data_verification = verify_data(Path(prepared["data"]))
        if supervisor.owned_members():
            raise Blocked("preparation descendants still alive")
        recommendations = {}
        for role in sorted({row["role"] for row in rows}):
            available = [row for row in rows if row["role"] == role and row["sequence_length"] == 128
                         and row["unprofiled_step_tokens_per_second"] and not row["id"].startswith("pf-resume-")]
            if available:
                best = max(row["unprofiled_step_tokens_per_second"] for row in available)
                winner = min((row for row in available if row["unprofiled_step_tokens_per_second"] >= 0.95 * best),
                             key=lambda row: (row["cpu_threads"], -row["unprofiled_step_tokens_per_second"]))
                recommendations[role] = {key: winner[key] for key in ("id", "model_id", "batch_size", "sequence_length",
                                        "cpu_threads", "unprofiled_step_tokens_per_second", "device_memory")}
        report.update(status="passed", measurements=rows, capacity_admission=admissions,
            capacity_rejections=rejections, recommendations=recommendations, recommended_common_sequence_length=128,
            recommendation_scope="exact tested model only; other candidates need their own capacity calibration; 256 not qualified across all paths",
            recovery={task["id"]: {"resume": results[task["id"]]["resume"],
                "numeric_trajectory": results[task["id"]]["recovery"]["numeric_trajectory"]} for task in resume_tasks},
            terminal_data_verification=data_verification,
            actual_peak_npu_concurrency=supervisor.state["peak_actual_npu_concurrency"],
            remaining_descendants=[], cgroup_scope="controller still owns its service; verify unit/cgroup release after exit")
        supervisor.finish("completed")
    except Exception as error:
        supervisor.stop_reason = supervisor.stop_reason or f"{type(error).__name__}: {error}"
        supervisor.stop_all()
        report.update(status="blocked", reason=supervisor.stop_reason, measurements=rows,
                      capacity_admission=admissions, capacity_rejections=rejections,
                      remaining_descendants=supervisor.state.get("remaining_descendants"),
                      actual_peak_npu_concurrency=supervisor.state.get("peak_actual_npu_concurrency", 0))
    finally:
        for identity, record in supervisor.state["tasks"].items():
            if identity not in results and record.get("result_path"):
                value = read(record["result_path"])
                if value is not None:
                    results[identity] = value
        report["finished_unix"] = time.time()
        report["device_seconds"] = supervisor.state.get("device_seconds", 0)
        report["processed_token_reservation"] = plan["processed_token_reservation"]
        report["observed_training_tokens"] = sum(value.get("training", {}).get("segment_tokens", 0) for value in results.values())
        report["observed_evaluation_tokens"] = sum(value.get("evaluation", {}).get("tokens", 0) for value in results.values())
        report["result_files"] = {identity: {"path": record.get("result_path"), "sha256": record.get("result_sha256")}
                                  for identity, record in supervisor.state["tasks"].items()}
        finish_token_accounting(ledger_path, str(output), results, report["status"])
        atomic_json(output / "preflight-report.json", report)
        supervisor.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    prepared = read(args.prepared)
    if args.plan_only:
        plan = build_plan(prepared)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run(prepared)
    except Exception as error:
        traceback.print_exc()
        path = Path(prepared["output"]) / "preflight-report.json"
        report = read(path, {})
        report.update(schema="pilot-r3-preflight-report-v1", status="blocked", stage_b_gate="closed",
                      reason=f"{type(error).__name__}: {error}", finished_unix=time.time(),
                      source_identity=prepared.get("source_identity"))
        atomic_json(path, report)
    print(json.dumps({"status": report["status"], "reason": report.get("reason"),
                      "report": str(Path(prepared["output"]) / "preflight-report.json")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
