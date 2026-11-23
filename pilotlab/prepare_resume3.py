"""Prepare the audited r04 continuation under a separately frozen controller.

Original training modules, model/data identities and old artifacts stay fixed.
This command only writes a new snapshot's requests and launch draft; no submit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from .io import sha256
from .prepare_campaign3 import budget_remaining, source_inventory, storage_observation
from .scheduler3 import evaluate_outcome, freeze_json, read, live


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def prepare(prepared_path, old_prepared_path, reconciliation_path):
    prepared_path, old_prepared_path = Path(prepared_path).resolve(), Path(old_prepared_path).resolve()
    new, old = read(prepared_path), read(old_prepared_path)
    root, output = prepared_path.parent, Path(new["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("continuation output is not empty")
    source_inventory(new)
    source_inventory(old)
    if new["budget_ledger"] != old["budget_ledger"] or new["data"] != old["data"] or new["python"] != old["python"]:
        raise ValueError("continuation must preserve original data, environment and budget")
    review = read(reconciliation_path)
    old_state_path = Path(old["output"]) / "scheduler.json"
    old_state = read(old_state_path)
    if (review.get("schema") != "pilot-r3-r04-reconciliation-v1"
            or review.get("status") != "supplemental_review_completed"
            or review["source_records"]["scheduler"] != reference(old_state_path)
            or review["source_records"]["prepared"] != reference(old_prepared_path)):
        raise ValueError("continuation needs the bound supplementary terminal review")
    if old_state.get("status") != "a_blocked_needs_repair" or old_state.get("stage_b_gate") != "closed":
        raise ValueError("expected the closed, interrupted r04 stage A")
    identities = [old_state.get("controller_identity"), *old_state.get("all_seen_processes", {}).values()]
    if any(live(identity) for identity in identities):
        raise ValueError("old campaign still has live owned processes")
    old_plan = read(old_prepared_path.parent / "campaign-plan.json")
    if len(old_plan["subjects"]) != 344 or len(old_state["tasks"]) != 144:
        raise ValueError("this adoption is scoped to the reviewed r04 campaign")
    remaining = budget_remaining(new["budget_ledger"], time.time())
    storage = storage_observation(root, reserve=old_plan["config"]["storage_reserve_bytes"], snapshot_bytes=0)
    if len(os.sched_getaffinity(0)) < 320:
        raise ValueError("CPU/NPU host thread policy needs the measured 320-core allowance")
    config = copy.deepcopy(old_plan["config"])
    config.update(source=new["source"], foundation=new["foundation"], run_id=new["run_id"],
        worker_source=old["source"], worker_foundation=old["foundation"],
        deadline_unix=remaining["deadline_unix"], max_workers=80, max_cpu_workers=40,
        max_cpu_task_threads=160, max_cpu_threads=320, max_rss_bytes=1_000_000_000_000,
        max_npu_devices=8, target_npu_devices=8, npu_workers_per_device=16,
        external_occupant_policy="continue_running",
        shared_max_hbm_bytes=48*2**30)
    adapter = reference(Path(new["source"]) / "pilotlab/resume_adapter3.py")
    config["resume_adapter"] = adapter
    imported = {"schema": "pilot-r3-resume-import-v1", "created_unix": time.time(),
        "old_prepared": reference(old_prepared_path), "old_scheduler": reference(old_state_path),
        "reconciliation": reference(reconciliation_path), "worker_source_identity": old["source_identity"],
        "prior_retirement": reference(Path(old["output"]) / "checkpoint-retirement.json"),
        "old_budget_segment": copy.deepcopy(read(new["budget_ledger"])["segments"][old["output"]]),
        "accounting": "old segment and all original evaluation reservations retained; new segment charges only new device leases and preparation work",
        "tasks": {}}
    state = {"schema": "pilot-r3-scheduler-v1", "started_unix": time.time(), "status": "preparing",
        "stage": "audited_resume", "stage_b_gate": "closed", "tasks": {}, "shared_groups": {},
        "events": [], "all_seen_processes": {}, "peak_actual_npu_concurrency": 0, "device_seconds": 0.}
    classifications = Counter()
    for task_id, record in old_state["tasks"].items():
        task = read(record["task_path"])
        original_task = reference(record["task_path"])
        result_path = Path(record["result_path"]) if record.get("result_path") else Path(record["attempts"][-1]["path"]) / "result.json"
        result = read(result_path)
        group = read(Path(old_state["shared_groups"][record["shared_group"]]["path"]) / "group.json")
        worker = group["workers"][task_id]
        entry = {"task": original_task, "result": reference(result_path) if result else None,
            "attempt_path": record["attempts"][-1]["path"], "original_scheduler_status": record["status"],
            "group": reference(Path(old_state["shared_groups"][record["shared_group"]]["path"]) / "group.json"),
            "worker_return_code": worker.get("return_code"),
            "evaluation_reserved_tokens": record.get("evaluation_reserved_tokens", 0),
            "preparation_reserved_tokens": record.get("preparation_reserved_tokens", 0),
            "additional_evaluation_reserved_tokens": 0}
        if result and (result["task_sha256"] != original_task["sha256"] or result["source_identity"] != old["source_identity"]):
            raise ValueError("old result/source binding changed")
        if result and evaluate_outcome(task, result, worker.get("return_code"))[0] == "completed":
            if not result.get("health", {}).get("eligible") or worker.get("return_code") != 0:
                raise ValueError("a completion lacks independent worker qualification")
            entry["disposition"] = "completed"
        elif result:
            if (task["phase"] != "a1" or result["status"] != "budget_exhausted" or result.get("stop_reason") != "signal"
                    or result.get("validation_initial", {}).get("status") != "completed"
                    or result.get("validation_final", {}).get("status") == "completed"):
                raise ValueError("only reviewed A1 signal interruptions may resume")
            cp = result["checkpoint"]
            if sha256(cp["path"]) != cp["sha256"]:
                raise ValueError("resume checkpoint bytes changed")
            entry["disposition"] = "resume"
            entry["checkpoint"] = {**cp, "tokens": result["training"]["tokens"], "steps": result["training"]["steps"]}
            entry["additional_evaluation_reserved_tokens"] = 32 + result.get("validation_final", {}).get("tokens", 0)
            receipt = {"schema": "pilot-r3-task-resume-receipt-v1", "task_id": task_id,
                "original_task": original_task, "result": entry["result"], "checkpoint": entry["checkpoint"],
                "source_identity": task["source_identity"], "original_initial_evaluation": result["validation_initial"],
                "evaluation_additional_reservation_tokens": entry["additional_evaluation_reserved_tokens"]}
            receipt_path = freeze_json(root / "resume-receipts" / (task_id+".json"), receipt)
            entry["resume_receipt"] = reference(receipt_path)
        else:
            if worker.get("status") != "not_started" or worker.get("pid") is not None:
                raise ValueError("missing result without proved non-start")
            entry["disposition"] = "not_started"
        imported["tasks"][task_id] = entry
        classifications[entry["disposition"]] += 1
        # Preserve exact bytes so copied task paths bind the same old result.
        task_path = output / "tasks" / task_id / "task.json"
        task_path.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(record["task_path"], task_path)
        if sha256(task_path) != original_task["sha256"]:
            raise ValueError("task changed during adoption")
        new_record = {"status": "completed" if entry["disposition"] == "completed" else "pending",
            "stage": record["stage"], "model_id": task["model_id"], "device": task.get("device", "npu:0"),
            "preparation": task["phase"] == "preflight", "task_path": str(task_path),
            "attempts": [], "prior_attempts": copy.deepcopy(record["attempts"]), "device_seconds": 0.,
            "imported_from": str(old_state_path), "imported_completion": entry["disposition"] == "completed"}
        for key in ("evaluation_reserved_tokens", "evaluation_budget_key"):
            if key in record:
                new_record[key] = record[key]
        if entry["disposition"] == "completed":
            new_record.update(result_path=str(result_path), result_sha256=sha256(result_path))
        elif entry["disposition"] == "resume":
            new_record.update(resume_receipt=entry["resume_receipt"], retry_checkpoint=entry["checkpoint"]["path"],
                              retry_checkpoint_sha256=entry["checkpoint"]["sha256"])
        state["tasks"][task_id] = new_record
    if classifications != {"completed": 127, "resume": 16, "not_started": 1}:
        raise ValueError(f"reviewed recovery set changed: {classifications}")
    additional = sum(row["additional_evaluation_reserved_tokens"] for row in imported["tasks"].values())
    if additional != 6912:
        raise ValueError("reviewed evaluation continuation reservation changed")
    imported["summary"] = {**dict(classifications), "additional_evaluation_reserved_tokens": additional}
    import_path = freeze_json(root / "resume-import.json", imported)
    config["resume_import"] = reference(import_path)
    state["config"] = config
    state["events"].append({"kind": "audited_cross_controller_resume", "import": config["resume_import"],
                            "summary": imported["summary"], "time_unix": time.time()})
    state["counts"] = dict(Counter(row["status"] for row in state["tasks"].values()))
    freeze_json(output / "scheduler.json", state)
    for name in ("selected.json", "model-resources.json"):
        shutil.copyfile(Path(old["output"]) / name, output / name)
    plan = {**old_plan, "config": config, "source_identity": old["source_identity"],
        "controller_source_identity": new["source_identity"], "worker_prepared": reference(old_prepared_path),
        "pipeline_stage_a": True, "pipeline_model_limit": 32,
        "resume_import": config["resume_import"], "budget_remaining_at_freeze": remaining,
        "storage_observation": storage,
        "resource_policy": "Up to 40 four-thread CPU workers, 160 CPU compute threads, 320 host threads and 80 workers overall. Up to 8 queued or running NPU owners and 16 workers per card. Admission uses measured HBM, a 1 TB process-tree RSS limit and the inherited storage settings."}
    plan_path = freeze_json(root / "campaign-plan.json", plan)
    campaign_prepared = {**new, "worker_source_identity": old["source_identity"], "worker_prepared": reference(old_prepared_path),
        "resume_import": config["resume_import"], "status": "frozen_resume_not_submitted"}
    campaign_prepared_path = freeze_json(root / "campaign-prepared.json", campaign_prepared)
    command = [new["python"], "-u", "-m", "pilotlab.round3", "--plan", str(plan_path), "--output", str(output)]
    launcher = root / "campaign-launch.sh"
    script = "\n".join(("#!/usr/bin/env bash", "set -euo pipefail", "source /usr/share/Modules/init/bash", "module purge",
        "module load ascend/dev-workspace-9.2.0", "export TORCH_DEVICE_BACKEND_AUTOLOAD=0", "export PYTHONDONTWRITEBYTECODE=1",
        "export PYTHONHASHSEED=0", "export OPENBLAS_NUM_THREADS=1", "export OMP_NUM_THREADS=4", "export MKL_NUM_THREADS=4",
        "export PYTHONPATH="+shlex.quote(new["environment"]["PYTHONPATH"]), "cd "+shlex.quote(new["source"]),
        "exec "+shlex.join(command), ""))
    with launcher.open("x") as stream:
        stream.write(script)
        stream.flush()
        os.fsync(stream.fileno())
    launcher.chmod(0o755)
    unit, log = "tide-pilot-r3-"+root.name+".service", root/"campaign.log"
    argv = ["systemd-run", "--user", f"--unit={unit}", "--service-type=exec", "--slice=background.slice",
        f"--working-directory={new['source']}", "--property=KillMode=control-group",
        f"--property=RuntimeMaxSec={math.ceil(remaining['remaining_wall_seconds'])+120}", "--property=TimeoutStopSec=120",
        "--property=MemoryAccounting=yes", "--property=MemoryMax=1000000000000",
        f"--property=StandardOutput=append:{log}", f"--property=StandardError=append:{log}", str(launcher)]
    draft = {"schema": "pilot-r3-campaign-launch-draft-v1", "status": "not_submitted", "unit": unit,
        "prepared_path": str(campaign_prepared_path), "prepared_sha256": sha256(campaign_prepared_path),
        "plan_path": str(plan_path), "plan_sha256": sha256(plan_path), "launcher": str(launcher), "launcher_sha256": sha256(launcher),
        "command_argv": command, "launch_argv": argv, "launch_command": shlex.join(argv), "environment": new["environment"],
        "output": str(output), "log": str(log), "stage_b_gate": "closed", "budget_ledger": new["budget_ledger"],
        "started_unix": remaining["started_unix"], "deadline_unix": remaining["deadline_unix"],
        "resume_import": config["resume_import"], "summary": imported["summary"], "subject_count": 344,
        "inspection_command": "python .local/pilot-r3/inspect.py", "stop_command": f"systemctl --user stop {unit}",
        "resume_policy": "fixed r04 training modules; new audited controller; original budget/deadline; no automatic code repair"}
    freeze_json(root / "campaign-launch-draft.json", draft)
    return draft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--old-prepared", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prepared, args.old_prepared, args.reconciliation), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
