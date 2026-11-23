"""Bind a repaired, supplemental campaign to frozen qualification evidence.

The snapshot and independent quad probe must already exist. This command drafts
the finite campaign only; it never submits, changes a snapshot, or resets time.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

from .io import sha256
from .prepare_campaign3 import (CORE_FILES, budget_remaining, load,
                                source_inventory, storage_observation)
from .round3 import DIAGNOSTIC_SEEDS, TUNING_SEEDS
from .scheduler3 import freeze_json
from .transfer3 import verify_stage_a_transfer

GIB = 1024**3


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def terminal_review(path, *, completed):
    report = load(path)
    if (report.get("schema") != "pilot-r3-terminal-audit-v1"
            or report.get("status") != "passed" or report.get("findings")
            or report.get("lifecycle", {}).get("status") != "terminal"
            or completed and report.get("execution", {}).get("status") != "completed"):
        raise ValueError(f"required terminal qualification did not pass: {path}")
    return report


def frozen_catalog(prepared):
    environment = {**os.environ, "PYTHONPATH": prepared["environment"]["PYTHONPATH"],
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    code = ("import json; from pilotlab.catalog3 import generate_combined_subjects, "
            "subject_catalog_contract; print(json.dumps({'subjects':generate_combined_subjects(),"
            "'contract':subject_catalog_contract(include_supplemental=True)},allow_nan=False))")
    result = subprocess.run([prepared["python"], "-c", code], env=environment,
        cwd=prepared["source"], text=True, capture_output=True, check=True, timeout=180)
    catalog = json.loads(result.stdout)
    subjects, contract = catalog["subjects"], catalog["contract"]
    if (len(subjects) != 344 or len({row["id"] for row in subjects}) != 344
            or len({row["model_key"] for row in subjects}) != 344
            or contract.get("subject_count_limit") != 344
            or contract.get("control_count_limit") != 1032
            or contract.get("total_model_count_limit") != 1376):
        raise ValueError("expected the authorized 216 + 128 distinct subject catalog")
    return subjects, contract


def transferred_resources(old_round, model_ids):
    """Carry measurements, not accounting or task completion state."""
    old = load(Path(old_round) / "model-resources.json")
    result = {}
    for mid in model_ids:
        row = copy.deepcopy(old[mid])
        row["device_by_batch"] = {str(batch): "npu:0" for batch in row["batches"]}
        row["npu_peak_bytes_by_batch"], row["rss_peak_bytes_by_batch"] = {}, {}
        for entry in row["measurements"]:
            task, measurement = load(entry["task_path"]), load(entry["result_path"])
            if (measurement.get("task_sha256") != sha256(entry["task_path"])
                    or measurement.get("status") != "completed" or task["model_id"] != mid):
                raise ValueError("transferred capacity evidence is inconsistent")
            memory, batch = measurement["device_memory"], str(task["batch_size"])
            row["npu_peak_bytes_by_batch"][batch] = max(memory.get("npu_peak_allocated_bytes", 0),
                                                       memory.get("npu_peak_reserved_bytes", 0))
            row["rss_peak_bytes_by_batch"][batch] = memory.get("cpu_rss_peak_bytes", 0)
        row["transfer_scope"] = "old capacity observations; selected recipes retain NPU and original batch/LR/threads"
        result[mid] = row
    return result


def finalize(prepared, *, transfer, old_audit, sharing_audit, repair_audit,
             repair_prepared, resource_review, target_npu_devices=7, max_cpu_workers=32):
    prepared_path = Path(prepared).resolve()
    new = load(prepared_path)
    output = prepared_path.parent
    if (new["max_workers"] != 80 or not 1 <= target_npu_devices <= 16
            or not 1 <= max_cpu_workers <= 32 or len(os.sched_getaffinity(0)) < 320):
        raise ValueError("revision needs measured CPU capacity for 80 four-thread slots")
    source_inventory(new)
    transfer_ref = reference(transfer)
    transfer_report = load(transfer)
    old = load(transfer_report["old_prepared"]["path"])
    if (Path(old["budget_ledger"]).resolve() != Path(new["budget_ledger"]).resolve()
            or Path(old["python"]).absolute() != Path(new["python"]).absolute()):
        raise ValueError("revision must keep the original ledger and private interpreter")
    subjects, contract = frozen_catalog(new)
    old_plan = load(Path(old["output"]).parent / "campaign-plan.json")
    if subjects[:216] != old_plan["subjects"]:
        raise ValueError("the original 216 subjects changed")
    selected = load(transfer_report["old_selected"]["path"])
    if set(selected) != set(transfer_report["model_ids"]) or len(selected) != 24:
        raise ValueError("expected exactly the 24 fully selected quad models")
    manifest = {"source_identity": new["source_identity"], "tuning_seeds": TUNING_SEEDS,
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "subjects": [{**row, "effective_key": row["model_key"]} for row in subjects]}
    verify_stage_a_transfer(transfer_ref, manifest=manifest, selected=selected)
    old_review = terminal_review(old_audit, completed=False)
    shared_review = terminal_review(sharing_audit, completed=True)
    repaired_review = terminal_review(repair_audit, completed=True)
    repair = load(repair_prepared)
    if (old_review["round"] != old["output"] or repaired_review["round"] != repair["output"]
            or load(Path(repair["output"]) / "scheduler.json").get("status") != "completed"):
        raise ValueError("terminal review is bound to the wrong campaign or repair")
    old_scheduler_path = Path(old["output"]) / "scheduler.json"
    old_scheduler = load(old_scheduler_path)
    if sha256(old_scheduler_path) != old_review["execution"]["scheduler"]["sha256"]:
        raise ValueError("old evaluation accounting changed after terminal audit")
    prior_evaluation = {mid: sum(row.get("evaluation_reserved_tokens", 0)
        for row in old_scheduler["tasks"].values() if row.get("evaluation_budget_key") == mid)
        for mid in transfer_report["model_ids"]}
    if any(type(value) is not int or not 0 < value < 2**20 for value in prior_evaluation.values()):
        raise ValueError("transferred stage-A evaluation budget is missing or exhausted")
    repair_project, repair_base = source_inventory(repair)
    new_project, new_base = source_inventory(new)
    if (repair_base != new_base or any(repair_project.get(name) != new_project.get(name) for name in CORE_FILES)
            or any(repair["source_identity"].get(key) != new["source_identity"].get(key)
                   for key in ("runtime_packages", "python_version", "data_manifest_sha256"))):
        raise ValueError("NPU repair qualification differs from frozen training/runtime")
    measured = load(resource_review)
    if measured.get("status") != "passed":
        raise ValueError("current measured resource review must pass")
    remaining = budget_remaining(new["budget_ledger"], time.time())
    storage = storage_observation(output, reserve=new["storage_reserve_bytes"], snapshot_bytes=0)
    config = {key: new[key] for key in ("python", "source", "foundation", "data", "run_id", "max_workers",
        "storage_reserve_bytes", "max_rss_bytes", "budget_ledger")}
    config.update(deadline_unix=remaining["deadline_unix"], shared_execution=True,
        external_occupant_policy="continue_running",
        max_cpu_workers=max_cpu_workers, max_cpu_threads=320, target_npu_devices=target_npu_devices,
        npu_workers_per_device=16, shared_max_hbm_bytes=48*GIB,
        max_inflight_checkpoint_bytes=old_plan["config"].get("max_inflight_checkpoint_bytes"))
    config["prior_evaluation_reservations"] = prior_evaluation
    protocol = {**old_plan["protocol"], "admitted_parallelism": target_npu_devices}
    qualifications = {name: reference(path) for name, path in {
        "old_terminal_audit": old_audit, "sharing_terminal_audit": sharing_audit,
        "repair_terminal_audit": repair_audit, "repair_prepared": repair_prepared,
        "presubmission_resources": resource_review}.items()}
    qualifications["old_evaluation_accounting"] = reference(old_scheduler_path)
    plan = {"schema": "pilot-r3-campaign-plan-v1", "config": config,
        "source_identity": new["source_identity"], "protocol": protocol, "subjects": subjects,
        "catalog_contract": contract, "stage_a_transfer": transfer_ref,
        "qualification_transfer": qualifications, "cohort_models": 16,
        "cpu_recipe_calibration": True, "cpu_min_exclusive_rate_ratio": 1.0,
        "retire_consumed_a_parents": True, "profile_private_checkpoint_target_bytes": 32*2**20,
        "profile_private_parameter_factor": 2, "budget_remaining_at_freeze": remaining,
        "storage_observation": storage,
        "resource_policy": "Up to 80 workers and 320 compute threads, including up to 32 CPU workers. Each leased NPU admits up to 16 independent workers subject to HBM, RSS and the plan's storage settings. Unknown capacity probes run exclusively."}
    # All validation precedes writes. The old references retain their identities.
    freeze_json(Path(new["output"]) / "selected.json", selected)
    freeze_json(Path(new["output"]) / "model-resources.json",
                transferred_resources(old["output"], transfer_report["model_ids"]))
    plan_path = freeze_json(output / "campaign-plan.json", plan)
    command = [new["python"], "-u", "-m", "pilotlab.round3", "--plan", str(plan_path), "--output", new["output"]]
    launcher = output / "campaign-launch.sh"
    script = "\n".join(("#!/usr/bin/env bash", "set -euo pipefail", "source /usr/share/Modules/init/bash",
        "module purge", "module load ascend/dev-workspace-9.2.0", "export TORCH_DEVICE_BACKEND_AUTOLOAD=0",
        "export PYTHONDONTWRITEBYTECODE=1", "export PYTHONHASHSEED=0", "export OPENBLAS_NUM_THREADS=1",
        "export OMP_NUM_THREADS=4", "export MKL_NUM_THREADS=4",
        "export PYTHONPATH=" + shlex.quote(new["environment"]["PYTHONPATH"]),
        "cd " + shlex.quote(new["source"]), "exec " + shlex.join(command), ""))
    with launcher.open("x") as stream:
        stream.write(script)
        stream.flush()
        os.fsync(stream.fileno())
    launcher.chmod(0o755)
    unit, log = "tide-pilot-r3-" + output.name + ".service", output / "campaign.log"
    runtime = math.ceil(remaining["remaining_wall_seconds"]) + 120
    argv = ["systemd-run", "--user", f"--unit={unit}", "--service-type=exec", "--slice=background.slice",
        f"--working-directory={new['source']}", "--property=KillMode=control-group",
        f"--property=RuntimeMaxSec={runtime}", "--property=TimeoutStopSec=120",
        f"--property=StandardOutput=append:{log}", f"--property=StandardError=append:{log}", str(launcher)]
    draft = {"schema": "pilot-r3-campaign-launch-draft-v1", "status": "not_submitted", "unit": unit,
        "prepared_path": str(prepared_path), "prepared_sha256": sha256(prepared_path),
        "plan_path": str(plan_path), "plan_sha256": sha256(plan_path), "launcher": str(launcher),
        "launcher_sha256": sha256(launcher), "command_argv": command, "launch_argv": argv,
        "launch_command": shlex.join(argv), "environment": new["environment"], "output": new["output"],
        "log": str(log), "stage_b_gate": "closed", "budget_ledger": new["budget_ledger"],
        "started_unix": remaining["started_unix"], "deadline_unix": remaining["deadline_unix"],
        "runtime_max_seconds": runtime, "qualification_transfer": qualifications,
        "inspection_command": "python .local/pilot-r3/inspect.py",
        "stop_command": f"systemctl --user stop {unit}",
        "resume_policy": "inspect first; original budget and immutable inputs; implementation failures close B and stop"}
    freeze_json(output / "campaign-launch-draft.json", draft)
    return draft


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "transfer", "old-audit", "sharing-audit", "repair-audit", "repair-prepared", "resource-review"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-npu-devices", type=int, default=7)
    parser.add_argument("--max-cpu-workers", type=int, default=32)
    args = parser.parse_args(argv)
    print(json.dumps(finalize(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
