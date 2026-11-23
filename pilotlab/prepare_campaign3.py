"""Freeze and draft the authorized campaign after passed preparation.

This command never submits a service, initializes a budget, or probes devices.
The caller supplies the currently measured device count. Qualification transfers
only when all specified numerical/training sources and foundation Python files
are unchanged; lifecycle differences remain explicit in the new plan.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from . import prepare3
from .io import sha256
from .scheduler import current_quota, storage_headroom
from .scheduler3 import freeze_json, read

GIB = 1024**3
CORE_FILES = {name + ".py" for name in ("worker3", "data3", "models3", "models", "worker", "io")}


def load(path):
    value = read(path)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object missing or invalid: {path}")
    return value


def finite_nonnegative(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def budget_remaining(path, now):
    ledger = load(path)
    if (ledger.get("max_device_seconds") != 7200 * 3600
            or ledger.get("preparation_max_device_seconds") != 48 * 3600
            or ledger.get("preparation_max_processed_tokens") != 64 * 2**20
            or not finite_nonnegative(ledger.get("started_unix"))
            or ledger.get("deadline_unix") != ledger["started_unix"] + 30 * 86400
            or not isinstance(ledger.get("segments"), dict)):
        raise ValueError("original approved budget and absolute deadline are required")
    spent = ledger.get("device_seconds")
    if not finite_nonnegative(spent):
        raise ValueError("invalid device-time ledger")
    segments = ledger["segments"]
    preparation = sum(row.get("preparation_device_seconds", row.get("device_seconds", 0)
                              if row.get("preparation") else 0) for row in segments.values())
    claims = ledger.get("preparation_token_reservations", {})
    reserved = sum(row["reserved_tokens"] for row in claims.values())
    reserved += sum(row.get("preparation_reserved_tokens", 0) for key, row in segments.items() if key not in claims)
    remaining = {"started_unix": ledger["started_unix"], "deadline_unix": ledger["deadline_unix"],
                 "max_device_seconds": ledger["max_device_seconds"],
                 "remaining_wall_seconds": ledger["deadline_unix"] - now,
                 "remaining_device_seconds": ledger["max_device_seconds"] - spent,
                 "remaining_preparation_device_seconds": 48 * 3600 - preparation,
                 "remaining_preparation_processed_tokens": 64 * 2**20 - reserved,
                 "budget_ledger": str(Path(path).resolve()), "observed_unix": now}
    if any(not finite_nonnegative(remaining[key]) or remaining[key] <= 0 for key in (
            "remaining_wall_seconds", "remaining_device_seconds", "remaining_preparation_device_seconds",
            "remaining_preparation_processed_tokens")):
        raise ValueError("original campaign or preparation budget has no remaining capacity")
    return remaining


def source_inventory(prepared):
    """Verify every frozen byte before interpreting relative qualification paths."""
    source = Path(prepared["source"]).resolve() / "pilotlab"
    foundation = Path(prepared["foundation"]).resolve() / "src/tide"
    project, base = {}, {}
    for filename, expected in prepared["source_identity"]["source_files"].items():
        path = Path(filename).resolve()
        if sha256(path) != expected:
            raise ValueError(f"frozen source changed: {path}")
        if path.is_relative_to(source):
            project[path.relative_to(source).as_posix()] = expected
        elif path.is_relative_to(foundation):
            base[path.relative_to(foundation).as_posix()] = expected
        else:
            raise ValueError("source identity contains a file outside its frozen roots")
    if not (CORE_FILES | {"catalog3.py"}) <= project.keys() or not base:
        raise ValueError("qualification source inventory is incomplete")
    if {p.relative_to(foundation).as_posix() for p in foundation.rglob("*.py")} != set(base):
        raise ValueError("frozen foundation Python inventory changed")
    return project, base


def compare_sources(old, project, base):
    old_project, old_base = old
    changed_core = [name for name in sorted(CORE_FILES) if project.get(name) != old_project.get(name)]
    if changed_core or "catalog3.py" not in project or base != old_base:
        raise ValueError(f"preflight qualification cannot transfer: core={changed_core}, foundation_changed={base != old_base}")
    return [{"file": name, "preflight_sha256": old_project.get(name), "campaign_sha256": project.get(name)}
            for name in sorted(set(project) | set(old_project)) if project.get(name) != old_project.get(name)]


def verify_preflight(prepared, report_path):
    report = load(report_path)
    if (report.get("status") != "passed" or report.get("remaining_descendants") != []
            or report.get("source_identity") != prepared["source_identity"]):
        raise ValueError("passed preflight with unchanged identity and no remaining descendants is required")
    if sha256(report["plan_path"]) != report.get("plan_sha256"):
        raise ValueError("preflight plan changed")
    state = load(Path(prepared["output"]) / "scheduler.json")
    peak = report.get("actual_peak_npu_concurrency")
    if (state.get("status") != "completed" or type(peak) is not int or peak < 1
            or state.get("peak_actual_npu_concurrency") != peak):
        raise ValueError("preflight terminal state or measured concurrency is invalid")
    resources = state.get("resources", {})
    if (not isinstance(resources.get("quota"), dict)
            or not finite_nonnegative(resources.get("rss_bytes"))
            or not finite_nonnegative(resources.get("available_storage_bytes"))
            or resources["available_storage_bytes"] <= 0):
        raise ValueError("preflight resource observations are missing")
    for entry in report.get("result_files", {}).values():
        if not entry.get("path") or sha256(entry["path"]) != entry.get("sha256"):
            raise ValueError("preflight result evidence changed")
    rows = report.get("measurements", [])
    if not rows:
        raise ValueError("preflight has no actual measurements")
    total_bytes = []
    for row in rows:
        result = load(row["result_path"])
        task_path = Path(prepared["output"]) / "tasks" / row["id"] / "task.json"
        task = load(task_path)
        if (sha256(row["result_path"]) != row.get("result_sha256")
                or result.get("task_sha256") != sha256(task_path)
                or result.get("source_identity") != prepared["source_identity"]
                or task.get("source_identity") != prepared["source_identity"]
                or result.get("status") != "completed" or result.get("health", {}).get("eligible") is not True
                or task.get("precision") != "fp32" or task.get("deterministic") is not False):
            raise ValueError("preflight measurement is not bound to qualified FP32 task evidence")
        memory = row.get("device_memory", {})
        total = memory.get("npu_total_memory_bytes")
        peak_bytes = memory.get("npu_peak_allocated_bytes")
        if (memory != result.get("device_memory") or type(total) is not int or total <= 0
                or type(peak_bytes) is not int or not 0 < peak_bytes <= total):
            raise ValueError("actual NPU memory measurement is missing or inconsistent")
        total_bytes.append(total)
    if not any(row.get("cpu_threads") == 4 and row.get("sequence_length") == 128 for row in rows):
        raise ValueError("the common 128-token / four-thread protocol lacks measured evidence")
    return report, state, min(total_bytes)


def storage_observation(output, *, reserve, snapshot_bytes):
    quota = current_quota()
    parent = Path(output).resolve().parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    available = storage_headroom(quota, free)
    if available < reserve + 2 * snapshot_bytes:
        raise ValueError("insufficient quota/filesystem headroom for frozen inputs and reserve")
    return {"observed_unix": time.time(), "quota": asdict(quota), "quota_limit_basis": "hard", "filesystem_free_bytes": free,
            "available_storage_bytes": available, "snapshot_source_bytes": snapshot_bytes,
            "storage_reserve_bytes": reserve,
            "scope": "input snapshot headroom only; campaign forecasts all resolved controls before release"}


def frozen_subjects(prepared):
    """Run the exact frozen CPU/meta catalog without vendor backend autoload."""
    environment = {**os.environ, "PYTHONPATH": prepared["environment"]["PYTHONPATH"],
                   "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run([prepared["python"], "-c",
        "import json; from pilotlab.catalog3 import generate_subjects; print(json.dumps(generate_subjects(), allow_nan=False))"],
        env=environment, cwd=prepared["source"], capture_output=True, text=True, timeout=120, check=True)
    subjects = json.loads(result.stdout)
    if (not isinstance(subjects, list) or len(subjects) != 216
            or len({row["id"] for row in subjects}) != 216
            or len({row["model_key"] for row in subjects}) != 216
            or any(row["model"]["family"] != "tide" for row in subjects)):
        raise ValueError("frozen catalog is not 216 distinct Tide subjects")
    return subjects


def prepare(repo, foundation, output, *, python, data, budget, preflight_prepared,
            card_count, max_workers=16, preflight_report=None,
            storage_reserve_bytes=24 * GIB, max_rss_bytes=512 * GIB):
    if any(type(value) is not int or value < 1 for value in (card_count, max_workers)):
        raise ValueError("measured card count and maximum workers must be positive integers")
    repo, foundation, output = Path(repo).resolve(), Path(foundation).resolve(), Path(output).resolve()
    preflight_prepared = Path(preflight_prepared).resolve()
    old = load(preflight_prepared)
    report_path = Path(preflight_report).resolve() if preflight_report else Path(old["output"]) / "preflight-report.json"
    if Path(old["budget_ledger"]).resolve() != Path(budget).resolve():
        raise ValueError("the original preflight budget ledger must be continued")
    if Path(old["python"]).absolute() != Path(python).absolute():
        raise ValueError("the qualified private Python environment must be retained")
    report, state, npu_total = verify_preflight(old, report_path)
    old_inventory = source_inventory(old)
    project = {p.name: sha256(p) for p in (repo / "pilotlab").glob("*.py")}
    base = {p.relative_to(foundation / "src/tide").as_posix(): sha256(p)
            for p in (foundation / "src/tide").rglob("*.py")}
    changes = compare_sources(old_inventory, project, base)
    remaining = budget_remaining(budget, time.time())
    source_paths = list((repo / "pilotlab").glob("*.py")) + list((foundation / "src/tide").rglob("*.py"))
    storage = storage_observation(output, reserve=storage_reserve_bytes,
                                  snapshot_bytes=sum(path.stat().st_size for path in source_paths))
    hashes = {"preflight_prepared_sha256": sha256(preflight_prepared), "preflight_report_sha256": sha256(report_path)}
    workers = min(card_count, max_workers)
    prepare3.freeze(repo, foundation, output, python=python, data=data, budget=budget,
                    max_workers=workers, storage_reserve_bytes=storage_reserve_bytes, max_rss_bytes=max_rss_bytes)
    prepared_path = output / "prepared.json"
    prepared = load(prepared_path)
    changes = compare_sources(old_inventory, *source_inventory(prepared))
    for key in ("data_manifest_sha256", "runtime_packages", "python_version"):
        if prepared["source_identity"].get(key) != old["source_identity"].get(key):
            raise ValueError(f"qualified runtime or data identity changed: {key}")
    if (sha256(preflight_prepared) != hashes["preflight_prepared_sha256"]
            or sha256(report_path) != hashes["preflight_report_sha256"]):
        raise ValueError("preflight evidence changed during campaign preparation")
    subjects = frozen_subjects(prepared)
    old_subjects = frozen_subjects(old)
    if subjects != old_subjects:
        raise ValueError("the complete 216-subject catalog changed after preflight")
    by_id = {row["id"]: row for row in subjects}
    for row in report["measurements"]:
        task = load(Path(old["output"]) / "tasks" / row["id"] / "task.json")
        if task["model"]["family"] == "tide" and (
                task["model_id"] not in by_id or task["model"] != by_id[task["model_id"]]["model"]):
            raise ValueError("qualified Tide representative differs from the final subject catalog")
    remaining = budget_remaining(budget, time.time())
    config = {key: prepared[key] for key in ("python", "source", "foundation", "data", "run_id", "max_workers",
                                             "storage_reserve_bytes", "max_rss_bytes", "budget_ledger")}
    config["deadline_unix"] = remaining["deadline_unix"]
    protocol = {"precision": "fp32", "deterministic": False, "sequence_length": 128, "cpu_threads": 4,
                "npu_total_bytes": npu_total, "admitted_parallelism": report["actual_peak_npu_concurrency"]}
    qualification = {"preflight_prepared_path": str(preflight_prepared), "preflight_report_path": str(report_path),
        **hashes, "core_files": sorted(CORE_FILES), "foundation_python_files": len(old_inventory[1]),
        "lifecycle_source_changes": [row for row in changes if row["file"] != "catalog3.py"],
        "catalog_source_changes": [row for row in changes if row["file"] == "catalog3.py"],
        "catalog_comparison": "complete frozen old/new 216-subject JSON equality plus qualified Tide task/model identity",
        "qualification_scope": report.get("qualification_scope"),
        "thread_selection": "four threads fixed from measured representative paths; no claim for every model",
        "preflight_resources": state["resources"], "measured_card_count": card_count,
        "card_count_source": "caller-provided current device-count measurement; helper performs no device probe"}
    plan = {"schema": "pilot-r3-campaign-plan-v1", "config": config, "source_identity": prepared["source_identity"],
            "protocol": protocol, "subjects": subjects, "cohort_models": 4,
            "profile_private_checkpoint_target_bytes": 32*2**20, "profile_private_parameter_factor": 2,
            "qualification_transfer": qualification, "budget_remaining_at_freeze": remaining,
            "storage_observation": storage}
    plan_path = freeze_json(output / "campaign-plan.json", plan)
    launcher = output / "campaign-launch.sh"
    command = [prepared["python"], "-u", "-m", "pilotlab.round3", "--plan", str(plan_path), "--output", prepared["output"]]
    script = "\n".join(("#!/usr/bin/env bash", "set -euo pipefail", "source /usr/share/Modules/init/bash",
        "module purge", "module load ascend/dev-workspace-9.2.0", "export TORCH_DEVICE_BACKEND_AUTOLOAD=0",
        "export PYTHONDONTWRITEBYTECODE=1", "export PYTHONHASHSEED=0", "export OPENBLAS_NUM_THREADS=1",
        "export OMP_NUM_THREADS=4", "export MKL_NUM_THREADS=4",
        "export PYTHONPATH=" + shlex.quote(prepared["environment"]["PYTHONPATH"]),
        "cd " + shlex.quote(prepared["source"]), "exec " + shlex.join(command), ""))
    with launcher.open("x") as stream:
        stream.write(script)
        stream.flush()
        os.fsync(stream.fileno())
    launcher.chmod(0o755)
    unit = "tide-pilot-r3-" + output.name + ".service"
    log = output / "campaign.log"
    runtime_seconds = math.ceil(remaining["remaining_wall_seconds"]) + 120
    argv = ["systemd-run", "--user", f"--unit={unit}", "--service-type=exec", "--slice=background.slice",
            f"--working-directory={prepared['source']}", "--property=KillMode=control-group",
            f"--property=RuntimeMaxSec={runtime_seconds}", "--property=TimeoutStopSec=120",
            f"--property=StandardOutput=append:{log}", f"--property=StandardError=append:{log}", str(launcher)]
    draft = {"schema": "pilot-r3-campaign-launch-draft-v1", "status": "not_submitted", "unit": unit,
        "prepared_path": str(prepared_path), "prepared_sha256": sha256(prepared_path),
        "plan_path": str(plan_path), "plan_sha256": sha256(plan_path),
        "preflight_report_path": str(report_path), "preflight_report_sha256": hashes["preflight_report_sha256"],
        "launcher": str(launcher), "launcher_sha256": sha256(launcher), "command_argv": command,
        "launch_argv": argv, "launch_command": shlex.join(argv), "environment": prepared["environment"],
        "output": prepared["output"], "log": str(log), "stage_b_gate": "closed",
        "budget_ledger": prepared["budget_ledger"], "started_unix": remaining["started_unix"],
        "deadline_unix": remaining["deadline_unix"], "runtime_max_seconds": runtime_seconds,
        "inspection_command": f"systemctl --user show {unit} -p ActiveState -p MainPID -p ControlGroup -p ExecMainStatus",
        "stop_command": f"systemctl --user stop {unit}",
        "resume_policy": "inspect first; retain the original absolute deadline and ledger; never auto-repair code",
        "submission_note": "draft only; final admission remains in the frozen campaign and stage B gate"}
    freeze_json(output / "campaign-launch-draft.json", draft)
    return draft


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "foundation", "output", "python", "data", "budget", "preflight-prepared"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--card-count", type=int, required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
