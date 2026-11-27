"""Verify interrupted attempts back to the unchanged stage's actual start.

Attempt counters remain physical counters. A stage origin is accepted only
through immutable task/result/adapter receipts and exact checkpoint loads.
No training code or checkpoint tensors are imported by this reader.
"""
from pathlib import Path
import json

from .io import sha256


def _bound(reference, *, document=True):
    if not isinstance(reference, dict) or not reference.get("path") or not reference.get("sha256"):
        raise ValueError("missing continuation reference")
    path = Path(reference["path"])
    if sha256(path) != reference["sha256"]:
        raise ValueError("continuation reference changed")
    if not document:
        return path
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("continuation reference is not an object")
    return value


def restored_private_state(resume, checkpoint):
    """The exact-load digest includes private state even when that state is None."""
    digest = checkpoint.get("state_sha256")
    if not digest or resume.get("state_sha256") != digest:
        return False
    reused = resume.get("state_reused")
    return reused is True or (
        reused is False and all(type(checkpoint.get(k)) is int and checkpoint[k] == 0
                                for k in ("steps", "tokens"))
        and resume.get("checkpoint_load", {}).get("status") == "exact"
    )


def validate_attempt_segment(task, current, previous, checkpoint, *, same_path):
    """Validate a same-backend exact checkpoint and the exposure on both sides."""
    if (previous.get("status") != "budget_exhausted" or previous.get("stop_reason") != "signal"
            or previous.get("failure_class")):
        raise ValueError("continuation predecessor was not a signal interruption")
    permitted = {"declared_training_exposure_not_reached", "evaluation_incomplete",
                 "checkpoint_load_not_verified", "causality_not_verified"}
    if set(previous.get("health", {}).get("blocking_reasons", [])) - permitted:
        raise ValueError("continuation predecessor has a scientific failure")
    if previous.get("causality") and previous["causality"].get("within_tolerance") is not True:
        raise ValueError("continuation predecessor failed causality")
    resume, training = current.get("resume", {}), current.get("training", {})
    prior_training = previous.get("training", {})
    tokens, steps = checkpoint.get("tokens"), checkpoint.get("steps")
    if (type(tokens) is not int or type(steps) is not int
            or not 0 <= tokens <= task["target_tokens"] or steps < 0
            or tokens != prior_training.get("tokens") or steps != prior_training.get("steps")
            or (same_path and resume.get("path") != checkpoint.get("path")) or resume.get("sha256") != checkpoint.get("sha256")
            or resume.get("from_tokens") != tokens or training.get("from_tokens") != tokens
            or resume.get("from_steps") != steps or training.get("from_steps") != steps
            or not restored_private_state(resume, checkpoint) or resume.get("fallback_used") is not False
            or resume.get("checkpoint_load", {}).get("status") != "exact"):
        raise ValueError("continuation does not establish the exact contiguous checkpoint load")
    end, end_steps = training.get("tokens"), training.get("steps")
    if (type(end) is not int or not tokens <= end <= task["target_tokens"]
            or type(end_steps) is not int or end_steps < steps
            or training.get("segment_tokens") != end - tokens
            or training.get("segment_steps") != end_steps - steps):
        raise ValueError("continuation segment exposure is inconsistent")


class RelocatedCheckpointResult(dict):
    """Original result fields plus the reader's hash-verified local checkpoint path."""


class CheckedAttemptResult(RelocatedCheckpointResult):
    """In-memory result view with a separately verified cross-location stage origin."""


def read_attempt_chain(task, result, entry):
    """Reconstruct the stage origin from raw attempts at their recovered locations."""
    records = entry.get("portable_attempts")
    if not isinstance(records, list) or not records:
        raise ValueError("missing portable attempt records")
    prior = origin = None
    seen = set()
    for item in records:
        receipt = _bound(item["receipt"])
        raw = _bound(item["result"])
        cp = raw.get("checkpoint") or {}
        checkpoint_ref = item["checkpoint"]
        _bound(checkpoint_ref, document=False)
        identity = (receipt.get("batch"), receipt.get("project"), receipt.get("plan_sha256"))
        if (receipt.get("schema") != "maflow-task-result/v1" or receipt.get("task") != task["id"]
                or receipt.get("fingerprint") != entry["task_sha256"] or receipt.get("attempt") in seen
                or any(raw.get(k) != task.get(k) for k in ("id","model_id","phase","source_identity"))
                or raw.get("task_sha256") != entry["task_sha256"]
                or any(ref["sha256"] != receipt.get("artifacts",{}).get(role,{}).get("sha256")
                       for role,ref in (("checkpoint",checkpoint_ref),("result",item["result"])))
                or cp.get("sha256") != checkpoint_ref["sha256"]):
            raise ValueError("portable attempt does not bind its task, raw result and checkpoint")
        seen.add(receipt["attempt"])
        if prior is None:
            if raw.get("resume",{}).get("runtime_override") or receipt.get("resume_from"):
                raise ValueError("portable evidence lacks its first stage attempt")
            origin = raw
            batch = identity
        else:
            resume = receipt.get("resume_from") or {}
            old_checkpoint = prior["checkpoint"]
            if (identity != batch or not raw.get("resume",{}).get("runtime_override")
                    or resume.get("parent",resume.get("attempt")) != last_receipt["attempt"]
                    or resume.get("sha256") != old_checkpoint["sha256"]
                    or resume.get("size") != last_receipt["artifacts"]["checkpoint"]["size"]
                    or resume.get("step") != old_checkpoint["steps"] or last_receipt["status"] != "interrupted"):
                raise ValueError("portable attempt predecessor differs")
            validate_attempt_segment(task,raw,prior,old_checkpoint,same_path=False)
        prior, last_receipt = raw, receipt
    if prior != result or last_receipt["status"] != "completed" or result["status"] != "completed":
        raise ValueError("portable chain does not terminate at the selected completion")
    if task.get("resume_checkpoint"):
        initial = entry.get("initial_checkpoint") or {}
        if (initial.get("original") != task["resume_checkpoint"]
                or initial.get("sha256") != task.get("resume_checkpoint_sha256")
                or initial.get("sha256") != origin.get("resume",{}).get("sha256")
                or origin["resume"].get("fallback_used") is not False):
            raise ValueError("original stage checkpoint cannot be matched to its relocated contents")
        _bound(initial, document=False)
        origin = {**origin, "resume":{**origin["resume"], "path":initial["original"]}}
    checked = CheckedAttemptResult(result)
    checked.origin = origin
    checked.relocated_checkpoint = entry["checkpoint_path"]
    return checked


def continuation_origin(task, result):
    """Return the first attempt without rewriting any attempt's exposure."""
    if isinstance(result, CheckedAttemptResult):
        return result.origin
    current, seen = result, set()
    for _ in range(64):
        adapter = current.get("execution_adapter")
        if not adapter:
            if current.get("resume", {}).get("runtime_override"):
                raise ValueError("same-task runtime resume has no bound adapter ancestry")
            return current
        if adapter.get("schema") == "pilot-r3-source-migration-adapter-v1":
            from .source_migration3 import migration_origin
            return migration_origin(task, current)
        if adapter.get("schema") != "pilot-r3-execution-adapter-v1":
            raise ValueError("unknown continuation adapter contract")
        original = adapter.get("original_task")
        if _bound(original) != task:
            raise ValueError("continuation task or recipe changed")
        _bound(adapter.get("adapter"), document=False)
        worker = adapter.get("worker") or {}
        if (task.get("source_identity", {}).get("source_files", {}).get(worker.get("path")) != worker.get("sha256")
                or not worker.get("sha256")):
            raise ValueError("continuation worker is outside the frozen training identity")
        _bound(worker, document=False)
        receipt = _bound(adapter.get("receipt"))
        reference = adapter.get("original_result")
        if (receipt.get("schema") not in {"pilot-r3-task-resume-receipt-v1", "pilot-r3-task-resume-receipt-v2"}
                or receipt.get("original_task") != original or receipt.get("result") != reference
                or receipt.get("task_id") != task.get("id")
                or receipt.get("source_identity") != task.get("source_identity")):
            raise ValueError("continuation receipt does not bind this task and predecessor")
        marker = str(Path(reference["path"]).resolve())
        if marker in seen:
            raise ValueError("cyclic continuation provenance")
        seen.add(marker)
        previous = _bound(reference)
        for value in (current, previous):
            if (value.get("task_sha256") != original["sha256"]
                    or any(value.get(key) != task.get(key) for key in ("id", "model_id", "phase", "source_identity"))):
                raise ValueError("continuation result identity changed")
        checkpoint = receipt.get("checkpoint")
        if checkpoint != previous.get("checkpoint") or adapter.get("checkpoint") != checkpoint:
            raise ValueError("continuation checkpoint does not bind the predecessor")
        _bound(checkpoint, document=False)
        validate_attempt_segment(task, current, previous, checkpoint, same_path=True)
        current = previous
    raise ValueError("continuation provenance exceeds the finite depth limit")
