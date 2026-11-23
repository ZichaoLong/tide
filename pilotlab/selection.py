"""Finite, predeclared learning-rate selection for a continuation stage.

This selects within one model/workload using complete seed groups at equal
exposure. It does not compare architectures or reinterpret failed candidates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics


CONTRACT_FIELDS = (
    "model", "backend", "batch_size", "sequence_length", "stream_chunks",
    "precision", "cpu_threads", "warmup_steps", "weight_decay", "grad_clip",
    "eval_batches", "decode_steps", "resume_check", "resume_verify_steps",
    "measurement_warmup_steps", "checkpoint_interval_steps", "checkpoint_identity",
)


def _read(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value, hashlib.sha256(payload).hexdigest()


def resolve_selection(task: dict, records: dict, root: Path) -> tuple[dict, dict]:
    """Resolve one bounded continuation from already terminal candidates.

    ``resume_select`` declares candidates, seeds and exact ``updates``. The
    scheduler owns dependency readiness, checkpoint verification and launch.
    Only ``lr`` and ``resume_task`` are derived; resources and budgets remain
    those declared by the continuation template.
    """
    spec = task["resume_select"]
    if spec.get("rule") != "lowest_mean_validation_nll":
        raise ValueError("unsupported selection rule")
    ids, seeds, updates = spec["candidates"], spec["seeds"], spec["updates"]
    if (not ids or len(ids) != len(set(ids)) or not seeds
            or len(seeds) != len(set(seeds)) or type(updates) is not int or updates < 1):
        raise ValueError("invalid finite candidate/seed/update declaration")
    if set(ids) != set(task.get("depends_on", [])):
        raise ValueError("selection candidates must equal declared dependencies")
    if task["seed"] not in seeds:
        raise ValueError("continuation seed is outside declared selection seeds")
    grouped: dict[float, list[dict]] = {}
    observations = []
    for identity in ids:
        if not isinstance(identity, str) or Path(identity).name != identity or identity in (".", ".."):
            raise ValueError("unsafe candidate id")
        state = records[identity]
        if state.get("status") in ("running", "queued", "not_started"):
            raise ValueError("selection requires terminal candidate records")
        directory = Path(root) / "tasks" / identity
        candidate, task_sha = _read(directory / "task.json")
        if candidate.get("id") != identity:
            raise ValueError("candidate identity mismatch")
        if any(candidate.get(key) != task.get(key) for key in CONTRACT_FIELDS):
            raise ValueError(f"candidate {identity} has a different comparison contract")
        if candidate.get("seed") not in seeds:
            raise ValueError("candidate seed outside the declared set")
        lr = candidate.get("lr")
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError("invalid candidate learning rate")
        result_path = directory / "result.json"
        result, result_sha = _read(result_path) if result_path.exists() else ({}, None)
        nll = result.get("validation_final", {}).get("nll")
        training = result.get("training", {})
        expected_tokens = updates * task["batch_size"] * task["sequence_length"]
        reasons = []
        if state.get("status") != "completed" or result.get("status") != "completed":
            reasons.append("execution_not_completed")
        if result.get("recovery", {}).get("status") != "within_tolerance":
            reasons.append("recovery_not_within_original_tolerance")
        if training.get("steps") != updates or training.get("tokens") != expected_tokens:
            reasons.append("declared_training_exposure_not_reached")
        if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
            reasons.append("missing_or_invalid_validation_nll")
        observation = {
            "id": identity, "lr": float(lr), "seed": candidate["seed"],
            "status": state.get("status"), "nll": nll,
            "updates": training.get("steps"), "tokens": training.get("tokens"),
            "eligible": not reasons, "ineligible_reasons": reasons,
            "task_sha256": task_sha, "result_sha256": result_sha,
        }
        observations.append(observation)
        grouped.setdefault(float(lr), []).append(observation)
    groups = []
    for lr, group in sorted(grouped.items()):
        actual_seeds = [row["seed"] for row in group]
        if len(actual_seeds) != len(set(actual_seeds)):
            raise ValueError("duplicate seed within a learning-rate group")
        complete = set(actual_seeds) == set(seeds)
        eligible = complete and all(row["eligible"] for row in group)
        groups.append({"lr": lr, "candidate_ids": [row["id"] for row in group],
                       "complete_seed_group": complete, "eligible": eligible,
                       "mean_validation_nll": statistics.mean(row["nll"] for row in group) if eligible else None,
                       "seed_nll_range": [min(row["nll"] for row in group), max(row["nll"] for row in group)] if eligible else None})
    eligible_groups = [group for group in groups if group["eligible"]]
    if not eligible_groups:
        raise ValueError("no complete, qualified seed group for continuation")
    winner = min(eligible_groups, key=lambda group: (group["mean_validation_nll"], group["lr"]))
    parent = next(row for row in grouped[winner["lr"]] if row["seed"] == task["seed"])
    resolved = copy.deepcopy(task)
    del resolved["resume_select"]
    resolved.update(lr=winner["lr"], resume_task=parent["id"])
    record = {"schema": "pilot-lr-selection-v1", "rule": spec["rule"],
              "selection_scope": "within one frozen model/recipe at equal token exposure; exploratory, not a quality ranking",
              "requested_task_id": task["id"], "seeds": seeds, "updates_per_seed": updates,
              "candidates": observations, "groups": groups, "winner": winner,
              "selected_parent": parent["id"], "selected_seed": task["seed"],
              "tie_break": "lowest numerical learning rate"}
    return resolved, record
