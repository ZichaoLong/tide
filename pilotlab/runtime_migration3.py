"""Admit immutable R3 trajectories on the two qualified Python runtimes.

The submitted plan binds the original task and byte-exact pre-cloud results.
Each run also binds the image from its own maflow allocation and the packages
observed in the worker. Checkpoint and training code see the original task.
This module is also shipped as a standalone file beside tide_portable.py.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SOURCE = {"python": "3.12.13", "torch": "2.12.0+cpu", "torch-npu": "2.12.0.post3", "numpy": "2.5.2"}
TARGET = {"python": "3.12.3", "torch": "2.10.0", "torch-npu": "2.10.0.post4", "numpy": "1.26.4"}
IMAGE = "swr.cn-southwest-2.myhuaweicloud.com/lingxi_blue/zlong-ascend-dev:torch2.10-cann9.1-r1"
IMAGE_DIGEST = "sha256:99442d9e494fd9b7c7dd0c666c2598bcd076cf3084429a5f0967d4b6b1472c7f"
REGION = "cn-southwest-2"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_runtime(task):
    identity = task["source_identity"]
    return {"python": identity["python_version"], **identity["runtime_packages"]}


def valid_allocation(manifest, name):
    allocation = {k: manifest.get(k) for k in ("name", "region", "image", "image_digest")}
    require(name and allocation == dict(name=name, region=REGION, image=IMAGE, image_digest=IMAGE_DIGEST),
            "runtime migration allocation does not match the submitted job and approved image")
    return allocation


def build_runtime_plan(task_path, *, prior_results=(), reserved_evaluation_tokens=0):
    """Freeze one task and its ordered signal interruptions, oldest first.

    prior_results accepts result.json paths or {path,sha256} receipts. Its bytes
    are embedded once so the chain survives storage relocation and cold release.
    """
    task_bytes = Path(task_path).read_bytes()
    task = json.loads(task_bytes)
    parents = []
    for reference in prior_results:
        path = Path(reference["path"] if isinstance(reference, dict) else reference)
        data = path.read_bytes()
        sha = digest(data)
        if isinstance(reference, dict):
            require(reference.get("sha256") == sha, "declared predecessor SHA-256 differs")
        parents.append({"path": str(path.absolute()), "sha256": sha, "json": data.decode("utf-8")})
    plan = {"schema": "tide-runtime-migration/v1", "task": task["id"], "fingerprint": digest(task_bytes),
            "backend": task["device"].split(":")[0], "source_runtime": source_runtime(task),
            "target_runtime": dict(TARGET), "image": IMAGE, "image_digest": IMAGE_DIGEST,
            "prior_results": parents, "reserved_evaluation_tokens": reserved_evaluation_tokens}
    history = history_for(plan, task, plan["fingerprint"])
    plan["evaluation"] = evaluation_plan(task, history, reserved_evaluation_tokens)
    return plan


def exact_link(task, current, previous):
    cp = previous["checkpoint"]
    train, resume = current["training"], current.get("resume") or {}
    require(resume.get("runtime_override") is True and resume.get("fallback_used") is False
            and resume.get("checkpoint_load", {}).get("status") == "exact"
            and resume.get("source_task_id") == task["id"]
            and resume.get("sha256") == cp["sha256"] and resume.get("state_sha256") == cp["state_sha256"]
            and resume.get("from_steps") == train["from_steps"] == cp["steps"]
            and resume.get("from_tokens") == train["from_tokens"] == cp["tokens"],
            "runtime attempt does not restore its exact predecessor")
    require(train["steps"] - train["from_steps"] == train["segment_steps"] >= 0
            and train["tokens"] - train["from_tokens"] == train["segment_tokens"] >= 0
            and train["tokens"] == train["steps"] * task["batch_size"] * task["sequence_length"]
            and train["tokens"] <= task["target_tokens"], "runtime continuation exposure differs")


def original_stage_start(task, result):
    train, resume = result["training"], result.get("resume") or {}
    initial = task.get("evaluation_start_tokens", 0)
    require(train.get("from_tokens") == initial and not resume.get("runtime_override"),
            "missing first attempt at the original stage start")
    if task.get("resume_checkpoint"):
        require(resume.get("sha256") == task["resume_checkpoint_sha256"] and resume.get("fallback_used") is False
                and resume.get("from_tokens") == initial and resume.get("checkpoint_load", {}).get("status") == "exact",
                "original stage parent differs from its declared checkpoint")
    else:
        require(not resume and initial == 0, "unexpected stage-start checkpoint")


def history_for(plan, task, fingerprint):
    require(plan["schema"] == "tide-runtime-migration/v1" and (plan["task"], plan["fingerprint"]) == (task["id"], fingerprint)
            and plan["backend"] == task["device"].split(":")[0] and plan["backend"] in ("cpu", "npu")
            and plan["source_runtime"] == source_runtime(task) == SOURCE and plan["target_runtime"] == TARGET
            and (plan["image"], plan["image_digest"]) == (IMAGE, IMAGE_DIGEST), "unapproved runtime plan or task identity")
    require(type(plan["reserved_evaluation_tokens"]) is int and plan["reserved_evaluation_tokens"] >= 0,
            "invalid original evaluation reservation")
    history, seen = [], set()
    previous_hashes = []
    allowed = {"declared_training_exposure_not_reached", "evaluation_incomplete", "checkpoint_load_not_verified",
               "causality_not_verified"}
    for reference in plan["prior_results"]:
        sha = digest(reference["json"].encode("utf-8"))
        require(sha == reference["sha256"] and sha not in seen, "predecessor result content or order differs")
        seen.add(sha)
        rec = json.loads(reference["json"])
        require(rec.get("schema") == "pilot-r3-worker-result-v1" and rec["task_sha256"] == fingerprint
                and all(rec.get(k) == task[k] for k in ("id", "model_id", "phase", "source_identity"))
                and (rec["status"], rec["stop_reason"]) == ("budget_exhausted", "signal")
                and not rec.get("failure_class") and not (set(rec["health"]["blocking_reasons"]) - allowed),
                "predecessor is not a signal-only interruption of this task")
        cp, train = rec["checkpoint"], rec["training"]
        require(cp.get("sha256") and cp.get("state_sha256") and 0 < cp.get("size_bytes", 0)
                and cp["steps"] == train["steps"] and cp["tokens"] == train["tokens"]
                and cp["tokens"] == cp["steps"] * task["batch_size"] * task["sequence_length"]
                and 0 <= cp["tokens"] <= task["target_tokens"], "predecessor checkpoint exposure differs")
        require(type(rec["evaluation"]["tokens"]) is int and rec["evaluation"]["tokens"] >= 0,
                "invalid observed predecessor evaluation")
        if train.get("segment_steps") or rec.get("causality"):
            require(rec.get("causality", {}).get("within_tolerance") is True, "unqualified predecessor causality")
        if history:
            exact_link(task, rec, history[-1])
        else:
            original_stage_start(task, rec)
        if "execution_runtime" in rec:
            verify_execution(rec, plan, previous_hashes)
            prior = sum(item["evaluation"]["tokens"] for item in history)
            require(rec["evaluation"]["prior_attempt_tokens"] == prior and
                    rec["evaluation"]["cumulative_task_observed_tokens"] == prior + rec["evaluation"]["tokens"],
                    "cloud predecessor accounting disagrees with its ancestry")
        history.append(rec)
        previous_hashes.append(sha)
    return history


def evaluation_plan(task, history, reserved):
    """Reconcile the original reservation, including the existing 32-token causality allowance."""
    count = task.get("eval_tokens", 16384)
    start = task.get("evaluation_start_tokens", 0)
    initial = initial_index = None
    if history:
        for index, rec in enumerate(history):
            obs = rec.get("validation_initial", {})
            if (rec["training"]["from_tokens"] != start or obs.get("status") != "completed"
                    or "inherited_from" in obs):
                continue
            nll = obs.get("nll")
            require(obs.get("split") == "selection" and obs.get("tokens") == obs.get("requested_tokens") == count
                    and isinstance(nll, (float, int)) and not isinstance(nll, bool) and math.isfinite(nll),
                    "original stage evaluation is incomplete or invalid")
            initial, initial_index = obs, index
            break
        start = history[-1]["checkpoint"]["tokens"]
        require(initial is not None or start == task.get("evaluation_start_tokens", 0),
                "cannot re-evaluate the original selection observation at a later exposure")
    interval = task.get("eval_interval_tokens", 1048576)
    middle = max(0, (task["target_tokens"]-1)//interval-start//interval) if interval > 0 else 0
    future = 32 + (1 + middle + (initial is None)) * count
    if task.get("allow_confirmation"):
        future += task.get("confirmation_tokens", 131072)
    prior = sum(r["evaluation"]["tokens"] for r in history)
    return {"initial": initial, "initial_index": initial_index, "prior_observed_tokens": prior, "future_upper_bound": future,
            "additional_reservation": max(0, prior + 32 * len(history) + future - reserved)}


def read_plan(path, expected_sha, task, fingerprint):
    payload = Path(path).read_bytes()
    require(digest(payload) == expected_sha, "runtime plan SHA-256 differs")
    plan = json.loads(payload)
    history = history_for(plan, task, fingerprint)
    require(plan["evaluation"] == evaluation_plan(task, history, plan["reserved_evaluation_tokens"]),
            "runtime plan evaluation accounting differs")
    return plan, history


def inherited_evaluation(plan, task):
    observation = plan["evaluation"]["initial"]
    ref = plan["prior_results"][plan["evaluation"]["initial_index"]]
    return {**observation, "tokens": 0, "seconds": 0.0,
            "original_observed_tokens": observation["tokens"], "original_observed_seconds": observation.get("seconds"),
            "exposure": task.get("evaluation_start_tokens", 0),
            "inherited_from": {"path": ref["path"], "sha256": ref["sha256"], "field": "validation_initial",
                               "task_id": task["id"], "exposure": task.get("evaluation_start_tokens", 0)}}


def verify_execution(result, plan, prior_hashes):
    runtime = result["execution_runtime"]
    require(runtime["schema"] == "tide-runtime-attempt/v1" and runtime["prior_result_hashes"] == prior_hashes
            and runtime["observed"] == plan["target_runtime"] and runtime["declared"] == plan["source_runtime"]
            and result["device"].split(":")[0] == plan["backend"], "runtime execution receipt differs from plan")
    valid_allocation(runtime["allocation"], runtime["run"])


def verify_result(task, result, plan, history, plan_sha):
    verify_execution(result, plan, [row["sha256"] for row in plan["prior_results"]])
    require(result["execution_runtime"]["runtime_plan_sha256"] == plan_sha and result["task_sha256"] == plan["fingerprint"]
            and all(result.get(k) == task[k] for k in ("id", "model_id", "phase", "source_identity")),
            "cloud result belongs to a different task or runtime plan")
    require(type(result["evaluation"]["tokens"]) is int and 0 <= result["evaluation"]["tokens"] <= plan["evaluation"]["future_upper_bound"] - 32
            and result["evaluation"]["prior_attempt_tokens"] == plan["evaluation"]["prior_observed_tokens"]
            and result["evaluation"]["cumulative_task_observed_tokens"] ==
            plan["evaluation"]["prior_observed_tokens"] + result["evaluation"]["tokens"],
            "migrated evaluation accounting differs")
    if history:
        exact_link(task, result, history[-1])
        if plan["evaluation"]["initial"] is not None:
            require(result["validation_initial"] == inherited_evaluation(plan, task),
                    "migrated attempt changed the stage's original selection loss")
    else:
        original_stage_start(task, result)


def reviewed_result(task, result, entry):
    from .continuation3 import CheckedAttemptResult
    ref = entry["runtime_plan"]
    plan, history = read_plan(ref["path"], ref["sha256"], task, result["task_sha256"])
    verify_result(task, result, plan, history, ref["sha256"])
    checked = CheckedAttemptResult(result)
    origin = history[0] if history else result
    if task.get("resume_checkpoint"):
        origin = {**origin, "resume": {**origin["resume"], "path": task["resume_checkpoint"]}}
    checked.origin = origin
    if entry.get("checkpoint_path"):
        checked.relocated_checkpoint = entry["checkpoint_path"]
    return checked
