"""Content-bound machine receipts for consumed successful A1/A2 checkpoints.

Creation requires every selected training checkpoint to exist and hash-match.
Later absence is accepted only when a separate immutable retirement manifest
names that exact consumed parent. All task/result files and A3 checkpoints
remain required. The helper never unlinks a file or grants an experiment gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .io import sha256
from .scheduler3 import freeze_json

SCHEMA = "pilot-r3-model-chain-receipt-v1"
RETIREMENT_SCHEMA = "pilot-r3-parent-checkpoint-retirement-v1"
PROTOCOL_FIELDS = {"chain_receipt", "parent_checkpoint_retirement"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def _read_reference(entry):
    from .gate3 import load
    if not isinstance(entry, dict) or not entry.get("path") or entry != _reference(entry["path"]):
        raise ValueError("chain receipt/retirement artifact changed or is not content-bound")
    return load(entry["path"])


def _checkpoint(result, *, phase, seed, allowed_missing):
    record = result.get("checkpoint") or {}
    if not record.get("path") or not record.get("sha256"):
        raise ValueError("selected chain checkpoint record is incomplete")
    path = Path(record["path"]).resolve()
    expected = allowed_missing.get(str(path))
    if path.is_file():
        if sha256(path) != record["sha256"]:
            raise ValueError("selected chain checkpoint is corrupt")
        size = path.stat().st_size
    elif expected and phase in {"a1", "a2"} and expected["sha256"] == record["sha256"]:
        size = expected["size_bytes"]
    else:
        raise ValueError("selected chain checkpoint is missing without an authorized consumed-parent retirement")
    if type(size) is not int or size <= 0 or record.get("size_bytes", size) != size:
        raise ValueError("selected chain checkpoint size is invalid")
    return {"path": str(path), "sha256": record["sha256"], "size_bytes": size, "phase": phase, "seed": seed}


def _inspect_chain(*, model, choice, source_identity, tuning_seeds, diagnostic_seeds, allowed_missing=None):
    from .catalog3 import model_key
    from .gate3 import MI, combine_activity, continuation_matches, eligible, last_activity, read_evidence, verify_activity_sources
    if not isinstance(source_identity, dict) or not source_identity:
        raise ValueError("chain receipt requires the immutable source identity")
    tuning, diagnostics = sorted(tuning_seeds), sorted(diagnostic_seeds)
    if (len(tuning) != 2 or len(set(tuning)) != 2 or len(diagnostics) != 3 or len(set(diagnostics)) != 3
            or not set(tuning) <= set(diagnostics) or any(type(seed) is not int or seed < 0 for seed in diagnostics)):
        raise ValueError("chain receipt requires the fixed two learning/three diagnostic seeds")
    mid, spec = model["id"], model["model"]
    key = model_key(spec)
    if model.get("model_key", key) != key or model.get("effective_key", key) != key:
        raise ValueError("chain receipt model identity is not its executable effective identity")
    allowed_missing = allowed_missing or {}
    evidence, checkpoints, continuations, groups, activities = [], [], [], {}, []
    for field, phase, target in (("selection_a1", "a1", MI//4), ("selection_a2", "a2", MI),
                                  ("confirmation", "a3", 2*MI), ("diagnostics", "init", None)):
        group = {}
        rows = choice.get(field)
        expected_seeds = diagnostics if phase == "init" else tuning
        if not isinstance(rows, list) or len(rows) != len(expected_seeds):
            raise ValueError(f"incomplete selected {phase} chain")
        for entry in rows:
            task, result, item = read_evidence(entry)
            seed = task.get("seed")
            if (type(seed) is not int or seed in group or seed not in expected_seeds
                    or task.get("model_id") != mid or task.get("model") != spec
                    or task.get("phase") != phase or task.get("source_identity") != source_identity
                    or target is not None and task.get("target_tokens") != target
                    or task.get("init_recipe") != choice.get("init_recipe")
                    or phase != "init" and task.get("recipe_id") != choice.get("recipe_id")):
                raise ValueError("selected chain task identity, recipe, phase or seed mismatch")
            reasons = eligible(task, result, require_resume=phase in {"a2", "a3"})
            if reasons:
                raise ValueError(f"selected chain evidence is not qualified: {reasons}")
            if phase == "a1" and task.get("resume_checkpoint"):
                raise ValueError("selected A1 must begin at its own fixed initialization")
            if phase in {"a2", "a3"}:
                parent = groups["a1" if phase == "a2" else "a2"].get(seed)
                if parent is None or not continuation_matches(parent, task, result):
                    raise ValueError("selected chain does not prove the actual exact parent continuation")
                parent_task, parent_result = parent
                continuations.append({"seed": seed, "parent_task_id": parent_task["id"], "child_task_id": task["id"],
                    "from_tokens": parent_task["target_tokens"], "parent_checkpoint_path": parent_result["checkpoint"]["path"],
                    "parent_checkpoint_sha256": parent_result["checkpoint"]["sha256"],
                    "checkpoint_load_status": result["recovery"]["checkpoint_load"]["status"]})
            if phase != "init":
                cp = _checkpoint(result, phase=phase, seed=seed, allowed_missing=allowed_missing)
                checkpoints.append(cp)
                item.update(checkpoint_path=cp["path"], checkpoint_sha256=cp["sha256"], checkpoint_size_bytes=cp["size_bytes"])
            if phase == "a3":
                activities.append(last_activity(result, task))
            item.update(phase=phase, seed=seed, task_id=task["id"])
            evidence.append(item)
            group[seed] = task, result
        if sorted(group) != expected_seeds:
            raise ValueError("selected chain seed set changed")
        groups[phase] = group
    if len({cp["path"] for cp in checkpoints}) != 6:
        raise ValueError("selected chain checkpoint paths are not six distinct payloads")
    clean_choice = {name: value for name, value in choice.items() if name not in PROTOCOL_FIELDS}
    measured = combine_activity(activities)
    verify_activity_sources(choice, measured)
    return {"schema": SCHEMA, "source_identity": copy.deepcopy(source_identity),
            "model": {"id": mid, "model": copy.deepcopy(spec), "model_key": key},
            "choice_sha256": _digest(clean_choice), "tuning_seeds": tuning, "diagnostic_seeds": diagnostics,
            "evidence": evidence, "continuations": continuations,
            "retirable_checkpoints": [cp for cp in checkpoints if cp["phase"] in {"a1", "a2"}],
            "retained_checkpoints": [cp for cp in checkpoints if cp["phase"] == "a3"],
            "activity": measured,
            "checkpoint_verification": "all six selected payloads read in full and SHA256/size verified before receipt creation",
            "scope": "machine continuation/retirement evidence; scientific admission still requires the complete stage-A gate"}


def write_model_chain_receipt(path, *, model, choice, source_identity, tuning_seeds, diagnostic_seeds):
    """Verify every live training payload, then write a once-only receipt."""
    receipt = _inspect_chain(model=model, choice=choice, source_identity=source_identity,
                             tuning_seeds=tuning_seeds, diagnostic_seeds=diagnostic_seeds)
    freeze_json(path, receipt)
    return _reference(path)


def _retirement_records(receipt_ref, receipt, choice):
    reference = choice.get("parent_checkpoint_retirement")
    if reference is None:
        return {}
    retirement = _read_reference(reference)
    if retirement.get("schema") != RETIREMENT_SCHEMA or retirement.get("receipt") != receipt_ref:
        raise ValueError("parent retirement manifest does not bind this chain receipt")
    raw_root = retirement.get("output_root")
    if not isinstance(raw_root, str) or not raw_root or str(Path(raw_root).resolve()) != raw_root:
        raise ValueError("parent retirement requires the canonical new-round output root")
    root = Path(raw_root)
    declared = retirement.get("checkpoints")
    allowed = {row["path"]: row for row in receipt["retirable_checkpoints"]}
    if not isinstance(declared, list) or not declared:
        raise ValueError("parent retirement list is empty or missing")
    result = {}
    for row in declared:
        if (not isinstance(row, dict) or row.get("path") not in allowed or row != allowed[row["path"]]
                or row["path"] in result or not Path(row["path"]).is_relative_to(root)):
            raise ValueError("retirement includes an unconsumed, duplicate or out-of-round checkpoint")
        result[row["path"]] = row
    for row in receipt["evidence"]:
        if not all(Path(row[field]).resolve().is_relative_to(root) for field in ("task_path", "result_path")):
            raise ValueError("retirement cannot consume evidence from an older or external round")
    return result


def verify_model_chain_receipt(receipt_ref, *, model, choice, source_identity, tuning_seeds, diagnostic_seeds):
    """Re-read all metadata and every remaining payload; check declared absences."""
    if choice.get("chain_receipt") != receipt_ref:
        raise ValueError("selection does not bind the requested chain receipt")
    receipt = _read_reference(receipt_ref)
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported model-chain receipt")
    allowed = _retirement_records(receipt_ref, receipt, choice)
    rebuilt = _inspect_chain(model=model, choice=choice, source_identity=source_identity,
        tuning_seeds=tuning_seeds, diagnostic_seeds=diagnostic_seeds, allowed_missing=allowed)
    if rebuilt != receipt:
        raise ValueError("model-chain receipt disagrees with independently reconstructed task/result/continuation evidence")
    # Pin the reference bytes again after re-reading the possibly large payloads.
    _read_reference(receipt_ref)
    if choice.get("parent_checkpoint_retirement"):
        _read_reference(choice["parent_checkpoint_retirement"])
    return {"receipt": dict(receipt_ref), "authorized_retirements": allowed,
            "retired_checkpoints": [row for path, row in allowed.items() if not Path(path).exists()],
            "retained_checkpoints": receipt["retained_checkpoints"]}


def retire_consumed_parents(receipt_ref, choice, output_root):
    """Return exact successful consumed-parent records; never delete files."""
    receipt = _read_reference(receipt_ref)
    verified = verify_model_chain_receipt(receipt_ref, model=receipt["model"], choice=choice,
        source_identity=receipt["source_identity"], tuning_seeds=receipt["tuning_seeds"],
        diagnostic_seeds=receipt["diagnostic_seeds"])
    root = Path(output_root).resolve()
    if not root.is_dir():
        raise ValueError("new-round output root does not exist")
    for row in receipt["evidence"]:
        if not all(Path(row[field]).resolve().is_relative_to(root) for field in ("task_path", "result_path")):
            raise ValueError("retirement evidence is outside this new round")
    records = receipt["retirable_checkpoints"]
    if len(records) != 4 or any(not Path(row["path"]).resolve().is_relative_to(root) for row in records):
        raise ValueError("retirement paths are not exactly the four consumed parents in this round")
    retained = {row["path"] for row in verified["retained_checkpoints"]}
    if any(row["path"] in retained for row in records):
        raise ValueError("retirement would remove an A3 checkpoint")
    return copy.deepcopy(records)
