"""Draft a finite, audited controller relay; never stop, submit or change a worker.

The destination is a fresh prepare3.freeze snapshot. Its controller can change;
all task bytes, training sources, dependencies and old charges remain bound.
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
from .prepare_resume3 import reference
from .resume3 import bound, imported_attempt, imported_entry, retirement_references
from .resume_adapter3 import build_relay_receipt
from .scheduler3 import evaluate_outcome, evaluation_bound, freeze_json, live, read


def validate_terminal(old_path, old, state, review):
    scheduler = Path(old["output"])/"scheduler.json"
    if (review.get("schema") != "pilot-r3-terminal-audit-v1" or review.get("status") != "passed"
            or review.get("findings") or review.get("prepared", {}).get("sha256") != sha256(old_path)
            or review.get("execution", {}).get("scheduler", {}).get("sha256") != sha256(scheduler)
            or Path(review.get("round", "")).resolve() != Path(old["output"]).resolve()):
        raise ValueError("a content-bound passed terminal audit is required")
    lifecycle = review.get("lifecycle", {})
    if (lifecycle.get("status") != "terminal"
            or any(lifecycle.get(key, {}).get("released") is not True for key in ("processes", "cgroup", "queue"))
            or lifecycle.get("systemd", {}).get("properties", {}).get("MainPID") != "0"
            or state.get("status") not in {"interrupted", "stopped_budget", "a_blocked_needs_repair", "failed"}
            or state.get("stage_b_gate", "closed") != "closed"):
        raise ValueError("relay requires a released, interrupted stage A")
    if any(live(identity) for identity in [state.get("controller_identity"), *state.get("all_seen_processes", {}).values()]):
        raise ValueError("old campaign still has live recorded processes")


def normalized_record(old_output, task_id, record, review):
    """Bind even never-dispatched requests without mutating the old scheduler."""
    result = copy.deepcopy(record)
    task_path = Path(result.get("task_path") or Path(old_output)/"tasks"/task_id/"task.json").resolve()
    task_ref = reference(task_path)
    observations = [row for row in review.get("artifacts", {}).get("files", [])
                    if row.get("kind") == "task" and row.get("task_id") == task_id]
    if (len(observations) != 1 or observations[0].get("sha256") != task_ref["sha256"]
            or Path(observations[0].get("path", "")).resolve() != task_path
            or observations[0].get("status") != "passed"):
        raise ValueError("terminal audit does not bind the immutable task: "+task_id)
    if result.get("task_sha256", task_ref["sha256"]) != task_ref["sha256"] or read(task_path).get("id") != task_id:
        raise ValueError("scheduler task identity changed: "+task_id)
    result["task_path"] = str(task_path)
    return result


def hard_stop_a3_parent(state, task):
    """Resolve the original A3 start against the qualified A2 result."""
    parent_id = task["id"].replace("-a3-", "-a2-")
    record = state["tasks"].get(parent_id, {})
    if (state.get("blocking_reason") != "user_signal" or parent_id == task["id"]
            or record.get("status") != "completed" or not record.get("attempts")
            or record["attempts"][-1].get("return_code") != 0):
        raise ValueError("forced A3 replay requires its completed original A2 parent")
    task_ref = reference(record["task_path"])
    result_ref = {"path": record["result_path"], "sha256": record["result_sha256"]}
    parent, result = bound(task_ref), bound(result_ref)
    cp = result.get("checkpoint") or {}
    segment_fields = {"id", "phase", "target_tokens", "evaluation_start_tokens", "resume_checkpoint", "resume_checkpoint_sha256"}
    if (parent.get("id") != parent_id or parent.get("phase") != "a2"
            or {k:v for k,v in parent.items() if k not in segment_fields} != {k:v for k,v in task.items() if k not in segment_fields}
            or result.get("id") != parent_id or result.get("task_sha256") != task_ref["sha256"]
            or result.get("source_identity") != task["source_identity"] or result.get("health", {}).get("eligible") is not True
            or evaluate_outcome(parent, result, record["attempts"][-1].get("return_code"))[0] != "completed"
            or cp.get("path") != task.get("resume_checkpoint") or not cp.get("path")
            or cp.get("sha256") != task.get("resume_checkpoint_sha256") or sha256(cp["path"]) != cp["sha256"]
            or cp.get("tokens") != parent["target_tokens"] or cp["tokens"] != task["evaluation_start_tokens"]
            or type(cp.get("steps")) is not int or cp["steps"] <= 0):
        raise ValueError("forced A3 replay parent identity, exposure or checkpoint changed")
    return {"task": task_ref, "result": result_ref, "checkpoint": cp}


def hard_stop_retry_proof(state, task, record, directory, group, worker):
    """One replay from the original stage start after a verified shutdown."""
    phase = task.get("phase")
    if (phase not in {"a1", "a3"} or phase == "a1" and task.get("resume_checkpoint") or record.get("prior_attempts")
            or state.get("status") != "a_blocked_needs_repair" or not state.get("blocking_reason")
            or group.get("status") != "failed" or group.get("reason") != "signal_15"
            or worker.get("return_code") != -9 or worker.get("cleanup_verified") is not True
            or worker.get("descendant_cleanup_verified") is not True or worker.get("remaining_descendants") != []
            or live(worker.get("process")) or live(group.get("owner"))
            or (directory/"failure.pt").exists()):
        raise ValueError("incomplete stage lacks a single verified forced shutdown")
    result = read(directory/"result.json")
    if (result.get("status") != "running" or result.get("failure_class") or result.get("execution_adapter")
            or result.get("health", {}).get("blocking_reasons") != ["not_evaluated"]):
        raise ValueError("forced shutdown result contains a completed/scientific verdict")
    paths = {key: directory/name for key,name in (("steps","steps.jsonl"),("health","health.jsonl"),
        ("heartbeat","heartbeat.json"),("checkpoint_index","checkpoint.json"))}
    steps = [json.loads(line) for line in paths["steps"].read_text().splitlines()]
    health = [json.loads(line) for line in paths["health"].read_text().splitlines()]
    if not steps or not health or any(row.get("observed_finite") is not True for row in health):
        raise ValueError("forced shutdown has no qualified bounded diagnostics")
    parent = hard_stop_a3_parent(state, task) if phase == "a3" else None
    start = parent["checkpoint"] if parent else {"steps": 0, "tokens": 0}
    previous_tokens = start["tokens"]
    for i,row in enumerate(steps,start["steps"]+1):
        if (row.get("step") != i or type(row.get("tokens")) is not int
                or not previous_tokens < row["tokens"] <= task["target_tokens"]
                or any(isinstance(row.get(key),bool) or not isinstance(row.get(key),(int,float))
                       or not math.isfinite(row[key]) for key in ("loss","objective","grad_norm","lr"))):
            raise ValueError("forced shutdown journal has invalid exposure or numerical values")
        previous_tokens = row["tokens"]
    heartbeat = read(paths["heartbeat"])
    # worker3 flushes the step journal before publishing the atomic heartbeat.
    tails = steps[-2:] if phase == "a3" else steps[-1:]
    if (heartbeat.get("task_id") != task["id"] or heartbeat.get("pid") != worker.get("pid")
            or not any(heartbeat.get("tokens") == row["tokens"] and heartbeat.get("steps") == row["step"] for row in tails)):
        raise ValueError("forced shutdown heartbeat/journal identity disagrees")
    index = read(paths["checkpoint_index"])
    cp = index.get("current") or {}
    if (index.get("schema") != "pilot-r3-checkpoint-generations-v1"
            or not Path(cp.get("path", "")).resolve().is_relative_to(directory.resolve()/"checkpoints")
            or not any(cp.get("tokens") == row["tokens"] and cp.get("steps") == row["step"] for row in steps)
            or sha256(cp["path"]) != cp.get("sha256")):
        raise ValueError("forced shutdown lacks its bound decisive checkpoint")
    return {"schema":"pilot-r3-forced-stop-retry-v1", "evidence":{key:reference(path) for key,path in paths.items()},
            "original_parent":parent, "replay_from_tokens":start["tokens"],
            "retained_checkpoint":cp, "observed_training_tokens":previous_tokens,
            "prior_segment_observed_tokens":previous_tokens-start["tokens"],
            "prior_training_upper_bound_tokens":task["target_tokens"]-start["tokens"],
            "reason":f"SIGKILL after controller SIGTERM; initial evaluation/result not published; one {phase.upper()} replay from its original start, no recovery claim"}


def attempt_evidence(state, task_id, record):
    """Keep each attempt's own result and return code through repeated relays."""
    prior = copy.deepcopy(record.get("prior_attempts", []))
    current = copy.deepcopy(record.get("attempts", []))
    evidence, paths = [], set()
    for attempt in prior:
        item, _, _ = imported_attempt(state["config"], task_id, attempt["path"])
        evidence.append(copy.deepcopy(item))
    for attempt in current:
        directory = Path(attempt["path"])
        path = directory/"result.json"
        item = {"attempt_path": str(directory), "task": reference(record["task_path"]),
                "result": reference(path) if path.exists() else None,
                "worker_return_code": attempt.get("return_code"), "disposition": "not_started"}
        if attempt.get("shared_group"):
            entry = state["shared_groups"][attempt["shared_group"]]
            group_path = Path(entry["path"])/"group.json"
            group = read(group_path)
            if sha256(entry["manifest_path"]) != entry["manifest_sha256"] or group["manifest_sha256"] != entry["manifest_sha256"]:
                raise ValueError("group manifest identity changed")
            worker = group["workers"][task_id]
            item.update(group=reference(group_path), worker_return_code=worker.get("return_code"),
                        worker_status=worker["status"], worker_pid=worker.get("pid"),
                        group_status=group.get("status"), group_return_code=entry.get("return_code"),
                        group_manifest=reference(entry["manifest_path"]))
            if path.exists() and worker.get("result_sha256") != sha256(path):
                raise ValueError("group result publication is not bound")
            if path.exists() and read(path).get("status") == "running" and worker.get("return_code") == -9:
                item["forced_stop_retry"] = hard_stop_retry_proof(state, read(record["task_path"]), record,
                    directory, group, worker)
        if path.exists():
            item["disposition"] = read(path)["status"]
        elif item.get("worker_status") != "not_started" or item.get("worker_pid") is not None:
            item["disposition"] = "missing_result_after_start"
        evidence.append(item)
    for item in evidence:
        path = str(Path(item["attempt_path"]).resolve())
        if path in paths:
            raise ValueError("duplicate attempt in relay ancestry")
        paths.add(path)
        if bound(item["task"]) != read(record["task_path"]):
            raise ValueError("historical attempt changed its immutable task")
        if item.get("result"):
            result = bound(item["result"])
            if result.get("task_sha256") != item["task"]["sha256"] or result.get("id") != task_id:
                raise ValueError("historical result binding changed")
    return prior+current, evidence


def continuation_results(evidence):
    """Select the actual chain; abandoned retries retain full old reservations."""
    refs = {str(Path(row["result"]["path"]).resolve()): row["result"] for row in evidence if row.get("result")}
    current = evidence[-1]["result"]
    chain, seen = [], set()
    while current:
        key = str(Path(current["path"]).resolve())
        if key in seen or len(seen) >= 64 or refs.get(key) != current:
            raise ValueError("unbound or cyclic continuation attempt ancestry")
        seen.add(key); chain.append(current)
        current = bound(current).get("execution_adapter", {}).get("original_result")
    return list(reversed(chain)), len(refs)-len(chain)


def classify(task, record, evidence, identity):
    original = reference(record["task_path"])
    if task.get("source_identity") != identity or task.get("phase") not in {"preflight", "init", "a1", "a2", "a3"}:
        raise ValueError("relay is limited to unchanged stage-A worker tasks")
    last = evidence[-1] if evidence else {}
    result = bound(last["result"]) if last.get("result") else None
    entry = {"relay_version": 2, "task": original, "result": last.get("result"),
             "worker_return_code": last.get("worker_return_code"), "attempt_path": last.get("attempt_path"),
             "attempt_evidence": evidence, "original_scheduler_status": record["status"],
             "evaluation_reserved_tokens": record.get("evaluation_reserved_tokens", 0),
             "preparation_reserved_tokens": record.get("preparation_reserved_tokens", 0),
             "additional_evaluation_reserved_tokens": 0}
    if result and (result.get("source_identity") != identity or result.get("task_sha256") != original["sha256"]):
        raise ValueError("latest worker result identity changed")
    if result and evaluate_outcome(task, result, last.get("worker_return_code"))[0] == "completed":
        if last.get("worker_return_code") != 0 or result.get("health", {}).get("eligible") is not True:
            raise ValueError("completed worker lacks qualification")
        entry["disposition"] = "completed"
    elif result and evaluate_outcome(task, result, last.get("worker_return_code"))[0] == "recipe_rejected":
        entry["disposition"] = "recipe_rejected"
    elif result and task.get("purpose") == "capacity_calibration" and result.get("failure_class") == "capacity_error":
        entry["disposition"] = "capacity_rejected"
    elif (task.get("phase") == "preflight" and task.get("purpose") == "capacity_calibration"
          and (result and result.get("status") == "budget_exhausted" and result.get("stop_reason") == "signal"
               or not result and evidence and last.get("disposition") != "not_started")):
        permitted = {"declared_training_exposure_not_reached", "evaluation_incomplete",
                     "checkpoint_load_not_verified", "causality_not_verified"}
        if result and (result.get("failure_class")
                or set(result.get("health", {}).get("blocking_reasons", [])) - permitted
                or result.get("causality") and result["causality"].get("within_tolerance") is not True):
            raise ValueError("capacity interruption contains a scientific failure: "+task["id"])
        entry.update(disposition="retry", additional_evaluation_reserved_tokens=evaluation_bound(task),
                     retry_scope="fresh complete capacity measurement; prior uncertain exposure/reservation retained, no recovery claim")
    elif result and result.get("status") == "budget_exhausted" and result.get("stop_reason") == "signal" and result.get("checkpoint"):
        refs, abandoned = continuation_results(evidence)
        abandoned_bound = abandoned * evaluation_bound(task)
        receipt = build_relay_receipt(original, refs, max(0,entry["evaluation_reserved_tokens"]-abandoned_bound))
        entry.update(disposition="resume", checkpoint=receipt["checkpoint"], receipt_value=receipt,
                     additional_evaluation_reserved_tokens=receipt["evaluation_additional_reservation_tokens"],
                     abandoned_attempt_evaluation_upper_bound_tokens=abandoned_bound)
    elif result and result.get("status") == "running" and last.get("forced_stop_retry"):
        if len(evidence) != 1 or task["phase"] not in {"a1", "a3"}:
            raise ValueError("forced stage replay is limited to one reviewed original attempt")
        entry.update(disposition="retry", additional_evaluation_reserved_tokens=evaluation_bound(task),
                     retry_scope=last["forced_stop_retry"]["reason"], forced_stop_retry=last["forced_stop_retry"])
    elif (not evidence or not result and last.get("disposition") == "not_started"):
        entry.update(disposition="not_started", additional_evaluation_reserved_tokens=max(0,
            evaluation_bound(task)-entry["evaluation_reserved_tokens"]))
    else:
        raise ValueError("unclassified interruption or scientific failure: "+task["id"])
    return entry


def verify_resume_payloads(config, entries):
    """Decode through the original CPU-loaded worker without creating owners."""
    rows = [{"task": row["task"], "checkpoint": row["checkpoint"]} for row in entries if row["disposition"] == "resume"]
    if not rows:
        return
    script = """
import json,sys
from pathlib import Path
from pilotlab import worker3 as w
for row in json.load(sys.stdin):
 task=json.loads(Path(row['task']['path']).read_text()); cp=row['checkpoint']
 w.validate_source_identity(task,Path(sys.argv[1]))
 payload=w._read_checkpoint(Path(cp['path']),cp['sha256'])
 w.validate_checkpoint(payload,task,task.get('device','npu:0'))
 if payload['task_id']!=task['id'] or any(payload[k]!=cp[k] for k in ('tokens','steps','state_sha256')):
  raise ValueError('relay payload does not match same-task checkpoint')
"""
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=config["worker_source"]+":"+config["worker_foundation"]+"/src")
    subprocess.run([config["python"], "-c", script, config["data"]], input=json.dumps(rows), text=True,
                   env=env, cwd=config["worker_source"], check=True, timeout=300)


def prepare(prepared_path, old_prepared_path, terminal_audit_path, *, resource_limited=False):
    prepared_path, old_prepared_path = Path(prepared_path).resolve(), Path(old_prepared_path).resolve()
    new, old = read(prepared_path), read(old_prepared_path)
    root, output = prepared_path.parent, Path(new["output"])
    old_state_path = Path(old["output"])/"scheduler.json"
    old_state, review = read(old_state_path), read(terminal_audit_path)
    validate_terminal(old_prepared_path, old, old_state, review)
    old_state_ref, ledger_before = reference(old_state_path), reference(new["budget_ledger"])
    if root == old_prepared_path.parent or output.is_relative_to(Path(old["output"])):
        raise ValueError("relay must use a separate snapshot and output")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("relay output is not empty")
    for key in ("python", "data", "budget_ledger"):
        if new[key] != old[key]:
            raise ValueError("relay changed environment, data or budget")
    source_inventory(new); source_inventory(old)
    old_plan_path = old_prepared_path.parent/"campaign-plan.json"
    old_plan = read(old_plan_path)
    submission = old_prepared_path.parent/"submission.json"
    draft = read(submission if submission.is_file() else old_prepared_path.parent/"campaign-launch-draft.json")
    if draft["plan_sha256"] != sha256(old_plan_path) or draft["prepared_sha256"] != sha256(old_prepared_path):
        raise ValueError("old launch does not bind the plan/prepared inputs")
    identity = old_plan["source_identity"]
    config = copy.deepcopy(old_state["config"])
    config.update(source=new["source"], foundation=new["foundation"], run_id=new["run_id"],
                  worker_source=config.get("worker_source", old["source"]),
                  worker_foundation=config.get("worker_foundation", old["foundation"]),
                  dynamic_shared_admission=True, dynamic_probe_owners=1,
                  dynamic_owner_max_total_workers=256, dynamic_owner_max_commands=257)
    if resource_limited:
        cpu_budget = config["max_cpu_task_threads"]
        allowed = sorted(os.sched_getaffinity(0))
        count = min(cpu_budget, len(allowed))
        config.update(max_workers=None, max_cpu_threads=None,
                      dynamic_probe_owners=old_state["config"].get("dynamic_probe_owners", 1),
                      host_cpu_affinity=[allowed[i * len(allowed) // count] for i in range(count)])
    source_inventory({"source": config["worker_source"], "foundation": config["worker_foundation"], "source_identity": identity})
    for key in ("data_manifest_sha256", "runtime_packages", "python_version"):
        if new["source_identity"][key] != identity[key]:
            raise ValueError("worker runtime/data identity changed")
    remaining = budget_remaining(new["budget_ledger"], time.time())
    if config["deadline_unix"] != remaining["deadline_unix"]:
        raise ValueError("relay must keep the original absolute deadline")
    storage = storage_observation(root, reserve=config["storage_reserve_bytes"], snapshot_bytes=0)
    config["resume_adapter"] = reference(Path(new["source"])/"pilotlab/resume_adapter3.py")
    parent_imports = [old_state["config"]["resume_import"]] if old_state["config"].get("resume_import") else []
    retirements = retirement_references({"prior_imports": parent_imports})
    retirement = Path(old["output"])/"checkpoint-retirement.json"
    if retirement.exists():
        retirements.append(reference(retirement))
    imported = {"schema": "pilot-r3-resume-import-v1", "relay_version": 2, "created_unix": time.time(),
        "old_prepared": reference(old_prepared_path), "old_scheduler": old_state_ref,
        "reconciliation": reference(terminal_audit_path), "worker_source_identity": identity,
        "prior_imports": parent_imports, "prior_retirements": retirements,
        "old_budget_segments": copy.deepcopy(read(new["budget_ledger"])["segments"]),
        "accounting": "all old segments and evaluation reservations retained; new segment charges only new leases/preparation",
        "tasks": {}}
    state = {"schema": "pilot-r3-scheduler-v1", "started_unix": time.time(), "status": "preparing",
        "stage": "audited_relay", "stage_b_gate": "closed", "tasks": {}, "shared_groups": {},
        "events": [], "all_seen_processes": {}, "peak_actual_npu_concurrency": 0, "device_seconds": 0.}
    for tid, old_record in old_state["tasks"].items():
        record = normalized_record(old["output"], tid, old_record, review)
        task = read(record["task_path"])
        history, evidence = attempt_evidence(old_state, tid, record)
        entry = classify(task, record, evidence, identity)
        if record.get("imported_completion"):
            previous, original, result = imported_entry(old_state["config"], tid)
            if previous["disposition"] != "completed" or original != task or entry["result"] != previous["result"]:
                raise ValueError("recursive completed evidence changed")
            entry["inherited_from"] = {"import": old_state["config"]["resume_import"], "task_id": tid}
        imported["tasks"][tid] = entry
        new_record = {"status": entry["disposition"] if entry["disposition"] in {"completed", "recipe_rejected"} else
                     "failed" if entry["disposition"] == "capacity_rejected" else "pending",
            "stage": record["stage"], "model_id": task["model_id"], "device": task.get("device", "npu:0"),
            "preparation": task["phase"] == "preflight", "task_path": str(output/"tasks"/tid/"task.json"),
            "attempts": [], "prior_attempts": history, "device_seconds": 0.,
            "imported_from": str(old_state_path), "imported_completion": entry["disposition"] == "completed"}
        for key in ("evaluation_reserved_tokens", "evaluation_budget_key"):
            if key in record:
                new_record[key] = record[key]
        if entry["disposition"] in {"completed", "capacity_rejected", "recipe_rejected"}:
            new_record.update(result_path=entry["result"]["path"], result_sha256=entry["result"]["sha256"])
        state["tasks"][tid] = new_record
    verify_resume_payloads(config, imported["tasks"].values())
    # Refuse insufficient per-model budget before writing any relay requests.
    used, extra = Counter(config.get("prior_evaluation_reservations", {})), Counter()
    for tid, row in imported["tasks"].items():
        task = bound(row["task"]); key = task.get("evaluation_budget_key")
        if key:
            used[key] += row["evaluation_reserved_tokens"]
            extra[key] += row["additional_evaluation_reserved_tokens"]
    for row in imported["tasks"].values():
        task = bound(row["task"]); key = task.get("evaluation_budget_key")
        if key and used[key]+extra[key] > task["evaluation_token_limit"]:
            raise ValueError("relay evaluation increment exceeds model limit: "+key)
    if reference(old_state_path) != old_state_ref or reference(new["budget_ledger"]) != ledger_before:
        raise ValueError("old scheduler or budget changed during relay preparation")
    for tid, row in imported["tasks"].items():
        target = Path(state["tasks"][tid]["task_path"])
        target.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(row["task"]["path"], target)
        if sha256(target) != row["task"]["sha256"]:
            raise ValueError("task changed while copying exact bytes")
        if row["disposition"] == "resume":
            receipt = row.pop("receipt_value")
            path = freeze_json(root/"resume-receipts"/(tid+".json"), receipt)
            row["resume_receipt"] = reference(path)
            state["tasks"][tid].update(resume_receipt=row["resume_receipt"],
                retry_checkpoint=row["checkpoint"]["path"], retry_checkpoint_sha256=row["checkpoint"]["sha256"])
    imported["summary"] = {**dict(Counter(row["disposition"] for row in imported["tasks"].values())),
        "additional_evaluation_reserved_tokens": sum(extra.values()), "additional_evaluation_by_model": dict(extra)}
    config["resume_import"] = reference(freeze_json(root/"resume-import.json", imported))
    state["config"] = config
    state["counts"] = dict(Counter(row["status"] for row in state["tasks"].values()))
    state["events"] = [{"kind": "audited_controller_relay", "import": config["resume_import"], "summary": imported["summary"]}]
    freeze_json(output/"scheduler.json", state)
    for name in ("selected.json", "model-resources.json"):
        if (Path(old["output"])/name).exists():
            shutil.copyfile(Path(old["output"])/name, output/name)
    plan = {**old_plan, "config": config, "source_identity": identity, "controller_source_identity": new["source_identity"],
        "worker_prepared": old_plan.get("worker_prepared", reference(old_prepared_path)),
        "resume_import": config["resume_import"], "budget_remaining_at_freeze": remaining, "storage_observation": storage}
    if resource_limited:
        plan["resource_policy"] = (f'{config["max_npu_devices"]} NPU leases, up to {config["npu_workers_per_device"]} '
            f'workers per device subject to measured HBM/throughput; CPU pool {cpu_budget} threads, '
            f'host affinity {len(config["host_cpu_affinity"])} CPUs; RSS {config["max_rss_bytes"]} bytes')
    module = "pilotlab.round3"
    if "stage_a_revision" in plan:
        from .stage_a_revision3 import verify_scope
        verify_scope(plan)
        module = "pilotlab.stage_a_revision3"
    plan_path = freeze_json(root/"campaign-plan.json", plan)
    campaign_prepared = {**new, "worker_source_identity": identity, "worker_prepared": plan["worker_prepared"],
                         "resume_import": config["resume_import"], "status": "frozen_relay_not_submitted",
                         "campaign_plan": reference(plan_path)}
    campaign_path = freeze_json(root/"campaign-prepared.json", campaign_prepared)
    return write_campaign_draft(root, new, plan, module, remaining,
        {"resume_import":config["resume_import"], "summary":imported["summary"],
         "resume_policy":"original worker, task dependencies, budget and deadline; audited controller relay only"})


def write_campaign_draft(root, new, plan, module, remaining, provenance):
    config, output = plan["config"], Path(new["output"])
    plan_path, campaign_path = root/"campaign-plan.json", root/"campaign-prepared.json"
    command = [new["python"], "-u", "-m", module, "--plan", str(plan_path), "--output", str(output)]
    if config.get("host_cpu_affinity"):
        command = ["taskset", "--cpu-list", ",".join(map(str, config["host_cpu_affinity"])), *command]
    launcher = root/"campaign-launch.sh"
    script = "\n".join(("#!/usr/bin/env bash", "set -euo pipefail", "source /usr/share/Modules/init/bash", "module purge",
        "module load ascend/dev-workspace-9.2.0", "export TORCH_DEVICE_BACKEND_AUTOLOAD=0", "export PYTHONDONTWRITEBYTECODE=1",
        "export PYTHONHASHSEED=0", "export OPENBLAS_NUM_THREADS=1", "export OMP_NUM_THREADS=4", "export MKL_NUM_THREADS=4",
        "export PYTHONPATH="+shlex.quote(new["environment"]["PYTHONPATH"]), "cd "+shlex.quote(new["source"]),
        "exec "+shlex.join(command), ""))
    with launcher.open("x") as handle:
        handle.write(script); handle.flush(); os.fsync(handle.fileno())
    launcher.chmod(0o755)
    unit, log = "tide-pilot-r3-"+root.name+".service", root/"campaign.log"
    argv = ["systemd-run", "--user", "--unit="+unit, "--service-type=exec", "--slice=background.slice",
        "--working-directory="+new["source"], "--property=KillMode=control-group",
        "--property=RuntimeMaxSec="+str(math.ceil(remaining["remaining_wall_seconds"])+120), "--property=TimeoutStopSec=120",
        "--property=MemoryAccounting=yes",
        *([] if config.get("runq_root") else ["--property=MemoryMax="+str(config["max_rss_bytes"])]),
        "--property=StandardOutput=append:"+str(log), "--property=StandardError=append:"+str(log), str(launcher)]
    draft = {"schema": "pilot-r3-campaign-launch-draft-v1", "status": "not_submitted", "unit": unit,
        "prepared_path": str(campaign_path), "prepared_sha256": sha256(campaign_path),
        "plan_path": str(plan_path), "plan_sha256": sha256(plan_path), "launcher": str(launcher), "launcher_sha256": sha256(launcher),
        "command_argv": command, "launch_argv": argv, "launch_command": shlex.join(argv), "environment": new["environment"],
        "output": str(output), "log": str(log), "budget_ledger": new["budget_ledger"], "stage_b_gate": "closed",
        "started_unix": remaining["started_unix"], "deadline_unix": remaining["deadline_unix"],
        **provenance, "subject_count": len(plan["subjects"]),
        "inspection_command": "python .local/pilot-r3/inspect.py", "stop_command": "systemctl --user stop "+unit,
        }
    freeze_json(root/"campaign-launch-draft.json", draft)
    return draft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "old-prepared", "terminal-audit"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--resource-limited", action="store_true",
                        help="use the CPU and NPU pool capacities instead of a separate worker/host-thread reservation cap")
    args = parser.parse_args()
    print(json.dumps(prepare(args.prepared, args.old_prepared, args.terminal_audit,
                             resource_limited=args.resource_limited), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
