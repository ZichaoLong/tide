"""Fail-closed scientific admission for the finite round-three campaign."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from .io import sha256
from .scheduler3 import freeze_json

MI = 2**20
RECIPE_FIELDS = ("model_id", "model", "recipe_id", "init_recipe", "lr", "batch_size", "device",
                 "sequence_length", "stream_chunks", "cpu_threads", "precision", "deterministic",
                 "schedule_tokens", "warmup_tokens", "lr_floor_ratio", "weight_decay", "grad_clip",
                 "auxiliary_loss_weight")


def load(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_evidence(entry, *, checkpoint=False, retired_checkpoints=None):
    task, result = load(entry["task_path"]), load(entry["result_path"])
    task_hash = sha256(entry["task_path"])
    if result.get("task_sha256") != task_hash:
        raise ValueError("worker result does not bind the frozen task")
    evidence = {"task_path": entry["task_path"], "task_sha256": task_hash,
                "result_path": entry["result_path"], "result_sha256": sha256(entry["result_path"])}
    if checkpoint:
        cp = result.get("checkpoint") or {}
        if not cp.get("path") or not cp.get("sha256"):
            raise ValueError("selected checkpoint missing or corrupt")
        path = Path(cp["path"])
        retired = (retired_checkpoints or {}).get(str(path.resolve()))
        if path.exists():
            if sha256(path) != cp["sha256"]:
                raise ValueError("selected checkpoint missing or corrupt")
        elif retired and retired.get("sha256") == cp["sha256"] and retired.get("phase") in {"a1", "a2"}:
            evidence["checkpoint_state"] = "retired_after_verified_continuation"
        else:
            raise ValueError("selected checkpoint missing or corrupt")
        evidence.update(checkpoint_path=cp["path"], checkpoint_sha256=cp["sha256"])
    return task, result, evidence


def activity_observation(result, task=None):
    if task and ((result.get("id") is not None and task.get("id") != result["id"])
                 or result.get("source_identity") != task.get("source_identity")):
        raise ValueError("activity belongs to a different task or source")
    rows = result.get("work")
    if rows == [] and task and task.get("stage_a_migration"):
        from .source_migration3 import inherited_activity
        activity = inherited_activity(task, result)
    else:
        if not isinstance(rows, list) or not rows or not isinstance(rows[-1], dict):
            raise ValueError("missing final measured activity")
        work = rows[-1].get("work")
        if not isinstance(work, dict) or not isinstance(work.get("activity"), dict):
            raise ValueError("missing final measured activity")
        activity = work["activity"]
        if "measurement_origin" in activity:
            raise ValueError("unverified activity origin in a direct measurement")
    total = activity.get("backbone_total_parameters")
    active = activity.get("mean_active_backbone_parameters")
    tokens = activity.get("measured_external_tokens")
    if (activity.get("qualified") is not True or type(total) is not int or total <= 0
            or type(tokens) is not int or tokens <= 0
            or isinstance(active, bool) or not isinstance(active, (int, float))
            or not math.isfinite(active) or not 0 < active <= total):
        raise ValueError("invalid final measured activity")
    return activity


def last_activity(result, task=None):
    row = activity_observation(result, task)
    return {k:row[k] for k in ("backbone_total_parameters", "mean_active_backbone_parameters",
                               "measured_external_tokens", "measurement_origin") if k in row}


def combine_activity(rows):
    if not rows or len({row["backbone_total_parameters"] for row in rows}) != 1:
        raise ValueError("activity parameter totals disagree across confirmation seeds")
    tokens = sum(row["measured_external_tokens"] for row in rows)
    active = sum(row["mean_active_backbone_parameters"] * row["measured_external_tokens"]
                 for row in rows) / tokens
    total = rows[0]["backbone_total_parameters"]
    sources = [row["measurement_origin"] for row in rows if "measurement_origin" in row]
    return {"backbone_total_parameters": total, "mean_active_backbone_parameters": active,
            "active_ratio": active / total, "measured_external_tokens": tokens,
            **({"inherited_samples":sources} if sources else {})}


def verify_activity_sources(choice, measured):
    recorded = choice.get("activity", {})
    if "inherited_samples" in measured or "inherited_samples" in recorded:
        if any(recorded.get(k) != v for k,v in measured.items()) or recorded.get("inherited_samples") != measured.get("inherited_samples"):
            raise ValueError("selected activity differs from the measured seed evidence")


def eligible(task, result, *, require_resume=False):
    reasons = []
    if result.get("status") != "completed":
        reasons.append("execution_not_completed")
    if result.get("source_identity") != task.get("source_identity") or not task.get("source_identity"):
        reasons.append("source_identity_mismatch")
    if result.get("training", {}).get("tokens") != task["target_tokens"]:
        reasons.append("token_target_not_reached")
    health = result.get("health", {})
    if health.get("eligible") is not True or health.get("blocking_reasons"):
        reasons.append("health_not_qualified")
    if task["phase"] not in {"init", "preflight"}:
        value = result.get("validation_final", {}).get("nll")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            reasons.append("missing_or_invalid_selection_nll")
    if require_resume and result.get("recovery", {}).get("checkpoint_load", {}).get("status") != "exact":
        reasons.append("state_restore_not_exact")
    if require_resume:
        expected_start = {"a2": MI//4, "a3": MI}.get(task["phase"])
        try:
            from .continuation3 import continuation_origin
            origin = continuation_origin(task, result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            reasons.append("continuation_provenance_invalid:" + str(error))
            origin = {}
        if (not task.get("resume_checkpoint") or expected_start is not None and
                (origin.get("resume", {}).get("from_tokens") != expected_start
                 or origin.get("training", {}).get("from_tokens") != expected_start)):
            reasons.append("declared_real_continuation_not_established")
    return reasons


def choose_recipes(tasks, results, seeds, *, keep):
    """Within-model selection at equal exposure with complete seed groups."""
    if len({task["model_id"] for task in tasks}) != 1:
        raise ValueError("selection mixes model identities")
    if len({task["target_tokens"] for task in tasks}) != 1:
        raise ValueError("selection mixes token exposures")
    groups = {}
    observations = []
    for task in tasks:
        result = results.get(task["id"], {})
        reasons = eligible(task, result, require_resume=task["phase"] in {"a2", "a3"})
        row = {"id": task["id"], "seed": task["seed"], "recipe_id": task["recipe_id"],
               "eligible": not reasons, "reasons": reasons,
               "nll": result.get("validation_final", {}).get("nll")}
        observations.append(row)
        groups.setdefault(task["recipe_id"], []).append(row)
    ranked = []
    for recipe, rows in groups.items():
        if sorted(row["seed"] for row in rows) != sorted(seeds):
            continue
        if all(row["eligible"] for row in rows):
            ranked.append({"recipe_id": recipe, "mean_nll": statistics.mean(row["nll"] for row in rows),
                           "seed_nll": [row["nll"] for row in rows], "tasks": [row["id"] for row in rows]})
    ranked.sort(key=lambda row: (row["mean_nll"], row["recipe_id"]))
    return {"observations": observations, "ranked": ranked, "selected": ranked[:keep],
            "rule": "complete fixed seed group, equal-token mean selection NLL; recipe id tie break"}


def catalog_limits(manifest):
    """Keep the legacy 216 ceiling unless the exact expanded contract is bound."""
    if "catalog_contract" not in manifest:
        return 216, 648
    from .catalog3 import subject_catalog_contract
    contract = manifest["catalog_contract"]
    if not isinstance(contract, dict) or contract.get("subject_count_limit") not in (216, 344):
        raise ValueError("invalid finite subject catalog contract")
    expected = subject_catalog_contract(include_supplemental=contract["subject_count_limit"] == 344)
    if contract != expected:
        raise ValueError("manifest catalog contract differs from the authorized finite catalog")
    return expected["subject_count_limit"], expected["control_count_limit"]


def audit_stage_a(manifest, selected, matches, *, remaining_resources_ok):
    """Re-read decisive artifacts; summaries and exit codes are not sufficient."""
    identity = manifest["source_identity"]
    expected_seeds = sorted(manifest["tuning_seeds"])
    expected_diag = sorted(manifest["diagnostic_seeds"])
    subjects = {m["id"] for m in manifest["subjects"]}
    model_by_id = {m["id"]: m for m in manifest["models"]}
    model_ids = set(model_by_id)
    blockers, evidence, eligible_models, measured_activity = [], [], [], {}
    chain_receipts, retired_checkpoint_count = {}, 0
    transferred_identities = {}
    transfer = None
    try:
        subject_limit, control_limit = catalog_limits(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        subject_limit, control_limit = 216, 648
        blockers.append(f"invalid_catalog_contract:{exc}")
    if "stage_a_transfer" in manifest:
        try:
            from .transfer3 import verify_stage_a_transfer
            transfer = verify_stage_a_transfer(manifest["stage_a_transfer"], manifest=manifest, selected=selected)
            transferred_identities = transfer["model_source_identities"]
        except (KeyError, ValueError, OSError, TypeError, SyntaxError) as exc:
            blockers.append(f"stage_a_transfer_evidence_error:{type(exc).__name__}:{exc}")
    for label, seeds, count in (("tuning", expected_seeds, 2), ("diagnostic", expected_diag, 3)):
        if len(seeds) != count or len(set(seeds)) != count or any(type(s) is not int or s < 0 for s in seeds):
            blockers.append(f"invalid_{label}_seed_contract")
    if not set(expected_seeds) <= set(expected_diag):
        blockers.append("learning_seeds_not_predeclared_in_diagnostics")
    if (len(manifest["subjects"]) != subject_limit or len(subjects) != subject_limit
            or len({m["effective_key"] for m in manifest["subjects"]}) != subject_limit):
        blockers.append(f"subject_catalog_not_{subject_limit}_unique_effective_models")
    if not subjects <= model_ids:
        blockers.append("subject_catalog_not_covered_by_model_manifest")
    if any(m.get("model", {}).get("family") != "tide"
           or m.get("model") != model_by_id.get(m["id"], {}).get("model") for m in manifest["subjects"]):
        blockers.append("subject_catalog_model_identity_mismatch")
    if len(model_ids) != len(manifest["models"]) or len(model_ids-subjects) > control_limit:
        blockers.append("control_catalog_invalid")
    if set(selected) != model_ids:
        blockers.append("selection_does_not_cover_model_manifest")
    for model_id in sorted(model_ids):
        choice = selected.get(model_id)
        if choice is None:
            continue
        stage_a_identity = transferred_identities.get(model_id, identity)
        local = []
        retired_checkpoints = {}
        if "chain_receipt" in choice:
            try:
                if model_id in transferred_identities:
                    raise ValueError("transferred old stage-A checkpoints must remain available")
                from .chain_receipt3 import verify_model_chain_receipt
                receipt = verify_model_chain_receipt(choice["chain_receipt"], model=model_by_id[model_id], choice=choice,
                    source_identity=stage_a_identity, tuning_seeds=expected_seeds, diagnostic_seeds=expected_diag)
                retired_checkpoints = receipt["authorized_retirements"]
                chain_receipts[model_id] = receipt["receipt"]
                retired_checkpoint_count += len(receipt["retired_checkpoints"])
            except (KeyError, ValueError, OSError, TypeError) as exc:
                local.append(f"chain_receipt_evidence_error:{type(exc).__name__}:{exc}")
        elif "parent_checkpoint_retirement" in choice:
            local.append("checkpoint_retirement_without_bound_chain_receipt")
        observed_seeds = []
        activity_rows = []
        parents = {}
        for field, phase, target in (("selection_a1", "a1", MI//4), ("selection_a2", "a2", MI)):
            group = {}
            for entry in choice.get(field, []):
                try:
                    task, result, item = read_evidence(entry, checkpoint=True, retired_checkpoints=retired_checkpoints)
                    if (task["model_id"] != model_id or task["phase"] != phase or task["target_tokens"] != target
                            or task["model"] != model_by_id[model_id]["model"]
                            or task["init_recipe"] != choice["init_recipe"]
                            or task["source_identity"] != stage_a_identity or task["recipe_id"] != choice["recipe_id"]):
                        local.append("invalid_selected_learning_chain")
                    local.extend(eligible(task, result, require_resume=phase == "a2"))
                    if task["seed"] in group:
                        local.append("duplicate_selected_chain_seed")
                    if phase == "a2":
                        previous = parents.get("a1", {}).get(task["seed"])
                        if previous is None or not continuation_matches(previous, task, result):
                            local.append("a2_not_from_selected_a1_checkpoint")
                    group[task["seed"]] = (task, result)
                    evidence.append(item)
                except (KeyError, ValueError, OSError, TypeError) as exc:
                    local.append(f"learning_chain_evidence_error:{type(exc).__name__}:{exc}")
            if sorted(group) != expected_seeds:
                local.append(f"incomplete_selected_{phase}_seed_group")
            parents[phase] = group
        for entry in choice.get("confirmation", []):
            try:
                task, result, item = read_evidence(entry, checkpoint=True)
                if task["model_id"] != model_id or task["phase"] != "a3" or task["target_tokens"] != 2*MI:
                    local.append("invalid_confirmation_task")
                if (task["source_identity"] != stage_a_identity or task["recipe_id"] != choice["recipe_id"]
                        or task["model"] != model_by_id[model_id]["model"]
                        or task["init_recipe"] != choice["init_recipe"]):
                    local.append("confirmation_identity_mismatch")
                local.extend(eligible(task, result, require_resume=True))
                previous = parents.get("a2", {}).get(task["seed"])
                if previous is None or not continuation_matches(previous, task, result):
                    local.append("a3_not_from_selected_a2_checkpoint")
                observed_seeds.append(task["seed"])
                activity_rows.append(last_activity(result, task))
                evidence.append(item)
            except (KeyError, ValueError, OSError, TypeError) as exc:
                local.append(f"confirmation_evidence_error:{type(exc).__name__}:{exc}")
        if sorted(observed_seeds) != expected_seeds:
            local.append("incomplete_confirmation_seed_group")
        observed_diag = []
        for entry in choice.get("diagnostics", []):
            try:
                task, result, item = read_evidence(entry)
                # Successful initialization payloads have no continuation
                # consumer and may already be retired by the existing policy.
                # Any still-present payload is checked; A1/A2/A3 stay mandatory.
                if (result.get("checkpoint") or {}).get("path") and Path(result["checkpoint"]["path"]).exists():
                    task, result, item = read_evidence(entry, checkpoint=True)
                if task["model_id"] != model_id or task["phase"] != "init" or task["init_recipe"] != choice["init_recipe"]:
                    local.append("invalid_initialization_diagnostic")
                if task["source_identity"] != stage_a_identity or task["model"] != model_by_id[model_id]["model"]:
                    local.append("diagnostic_identity_mismatch")
                local.extend(eligible(task, result))
                observed_diag.append(task["seed"])
                evidence.append(item)
            except (KeyError, ValueError, OSError, TypeError) as exc:
                local.append(f"diagnostic_evidence_error:{type(exc).__name__}:{exc}")
        if sorted(observed_diag) != expected_diag:
            local.append("incomplete_initialization_seed_group")
        try:
            measured_activity[model_id] = combine_activity(activity_rows)
            verify_activity_sources(choice, measured_activity[model_id])
        except ValueError as exc:
            local.append(str(exc))
        if local:
            blockers.append({"model_id": model_id, "reasons": sorted(set(local))})
        else:
            eligible_models.append(model_id)
    if len(subjects.intersection(eligible_models)) < 200:
        blockers.append("fewer_than_200_qualified_subjects")
    match_keys, reviewed_matches = set(), []
    for match in matches:
        reviewed = dict(match)
        reviewed_matches.append(reviewed)
        key = (match.get("subject_id"), match.get("kind"))
        if key in match_keys:
            blockers.append("duplicate_control_match")
        match_keys.add(key)
        if match.get("qualified") is not True or match.get("control_id") not in eligible_models:
            blockers.append({"invalid_control_match": match})
        if match.get("subject_id") not in subjects:
            blockers.append("control_matches_unknown_subject")
        control = model_by_id.get(match.get("control_id"), {})
        required_family = "moe" if match.get("kind") == "moe_joint" else "transformer"
        if match.get("control_id") in subjects or control.get("model", {}).get("family") != required_family:
            blockers.append("control_model_family_mismatch")
        # A frozen matcher reports independent relative parameter errors and
        # an absolute activity-ratio error. Missing values cannot pass.
        fields = {"dense_total": ["total_relative_error"],
                  "dense_active": ["active_relative_error"],
                  "moe_joint": ["total_relative_error", "active_relative_error"]}.get(match.get("kind"))
        if fields is None:
            blockers.append("unknown_control_match_kind")
            continue
        for field in fields:
            value = match.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 0.10:
                blockers.append(f"control_match_out_of_tolerance:{key}:{field}")
        if match["kind"] == "moe_joint":
            value = match.get("ratio_absolute_error")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 0.02:
                blockers.append(f"control_ratio_out_of_tolerance:{key}")
        subject_activity = measured_activity.get(match.get("subject_id"))
        control_activity = measured_activity.get(match.get("control_id"))
        if subject_activity is None or control_activity is None:
            blockers.append(f"missing_measured_control_activity:{key}")
            continue
        errors = {
            "total_relative_error": abs(control_activity["backbone_total_parameters"] -
                                         subject_activity["backbone_total_parameters"]) /
                                    subject_activity["backbone_total_parameters"],
            "active_relative_error": abs(control_activity["mean_active_backbone_parameters"] -
                                          subject_activity["mean_active_backbone_parameters"]) /
                                     subject_activity["mean_active_backbone_parameters"],
            # Compare the ratio difference before normalizing. Subtracting
            # rounded ratios can turn an exact .02 boundary into .02000...018.
            "ratio_absolute_error": abs(control_activity["mean_active_backbone_parameters"] *
                                         subject_activity["backbone_total_parameters"] -
                                         subject_activity["mean_active_backbone_parameters"] *
                                         control_activity["backbone_total_parameters"]) /
                                    (control_activity["backbone_total_parameters"] *
                                     subject_activity["backbone_total_parameters"]),
        }
        reviewed["measured_errors"] = errors
        if any(errors[field] > 0.10 for field in fields):
            blockers.append(f"measured_control_match_out_of_tolerance:{key}")
        if match["kind"] == "moe_joint" and errors["ratio_absolute_error"] > 0.02:
            blockers.append(f"measured_control_ratio_out_of_tolerance:{key}")
    if match_keys != {(s, k) for s in subjects for k in ("dense_total", "dense_active", "moe_joint")}:
        blockers.append("incomplete_control_matching")
    if not remaining_resources_ok:
        blockers.append("insufficient_remaining_resources")
    return {"schema": "pilot-r3-stage-a-audit-v1", "status": "passed" if not blockers else "blocked",
            "blocking_reasons": blockers, "source_identity": identity, "eligible_models": eligible_models,
            "eligible_subject_count": len(subjects.intersection(eligible_models)), "evidence": evidence,
            "matches": reviewed_matches, "measured_activity": measured_activity,
            "manifest_content_sha256": manifest_digest(manifest), "remaining_resources_ok": remaining_resources_ok,
            "selected": selected,
            "stage_a_transfer": transfer["report"] if transfer else None,
            "transferred_model_ids": transfer["model_ids"] if transfer else [],
            "chain_receipts": chain_receipts, "retired_parent_checkpoint_count": retired_checkpoint_count}


def continuation_matches(parent, task, result):
    previous_task, previous_result = parent
    checkpoint = previous_result.get("checkpoint") or {}
    try:
        from .continuation3 import continuation_origin
        origin = continuation_origin(task, result)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    resume = origin.get("resume") or {}
    return (all(previous_task.get(key) == task.get(key) for key in (*RECIPE_FIELDS, "seed", "source_identity"))
            and task.get("resume_checkpoint") == checkpoint.get("path")
            and task.get("resume_checkpoint_sha256") == checkpoint.get("sha256")
            and resume.get("path") == checkpoint.get("path")
            and resume.get("sha256") == checkpoint.get("sha256")
            and resume.get("from_tokens") == previous_task["target_tokens"])


def create_gate(path, *, manifest_path, report_path, task_paths, gate_kind="formal"):
    manifest, report = load(manifest_path), load(report_path)
    if report.get("status") != "passed" or report.get("blocking_reasons"):
        raise ValueError("stage A has not passed")
    if report.get("eligible_subject_count", 0) < 200:
        raise ValueError("insufficient distinct qualified subjects")
    if report.get("source_identity") != manifest.get("source_identity") or not manifest.get("source_identity"):
        raise ValueError("stage source identities differ")
    if report.get("manifest_content_sha256") != manifest_digest(manifest):
        raise ValueError("stage A report belongs to a different manifest")
    subject_limit, control_limit = catalog_limits(manifest)
    subject_ids = {model["id"] for model in manifest["subjects"]}
    model_ids = {model["id"] for model in manifest["models"]}
    if (len(manifest["subjects"]) != subject_limit or len(subject_ids) != subject_limit
            or len({model["effective_key"] for model in manifest["subjects"]}) != subject_limit
            or len(model_ids) != len(manifest["models"]) or not subject_ids <= model_ids
            or len(model_ids-subject_ids) > control_limit):
        raise ValueError("stage B manifest exceeds or violates the authorized finite catalog")
    transferred = set()
    if "stage_a_transfer" in manifest:
        from .transfer3 import verify_stage_a_transfer
        transfer = verify_stage_a_transfer(manifest["stage_a_transfer"], manifest=manifest, selected=report["selected"])
        if report.get("stage_a_transfer") != transfer["report"] or report.get("transferred_model_ids") != transfer["model_ids"]:
            raise ValueError("stage A report does not bind the independently verified transfer")
        transferred = set(transfer["model_ids"])
    elif report.get("stage_a_transfer") or report.get("transferred_model_ids"):
        raise ValueError("stage A report claims an undeclared transfer")
    eligible_ids = set(report.get("eligible_models", []))
    if len(subject_ids & eligible_ids) != report["eligible_subject_count"]:
        raise ValueError("stage A qualified-subject count disagrees with its model identities")
    if not report.get("evidence"):
        raise ValueError("stage A report has no decisive evidence")
    retired_checkpoints, chain_receipts = {}, {}
    models = {row["id"]: row for row in manifest["models"]}
    for model_id, choice in report.get("selected", {}).items():
        if "chain_receipt" not in choice:
            if "parent_checkpoint_retirement" in choice:
                raise ValueError("checkpoint retirement has no bound chain receipt")
            continue
        if model_id in transferred:
            raise ValueError("transferred old stage-A checkpoints must remain available")
        from .chain_receipt3 import verify_model_chain_receipt
        receipt = verify_model_chain_receipt(choice["chain_receipt"], model=models[model_id], choice=choice,
            source_identity=manifest["source_identity"], tuning_seeds=manifest["tuning_seeds"],
            diagnostic_seeds=manifest["diagnostic_seeds"])
        retired_checkpoints.update(receipt["authorized_retirements"])
        chain_receipts[model_id] = receipt["receipt"]
    if report.get("chain_receipts", {}) != chain_receipts:
        raise ValueError("stage A report does not bind the independently verified chain receipts")
    for entry in report["evidence"]:
        for field in ("task", "result", "checkpoint"):
            path_key, hash_key = f"{field}_path", f"{field}_sha256"
            if field == "checkpoint" and path_key not in entry:
                continue
            retired = retired_checkpoints.get(str(Path(entry[path_key]).resolve())) if entry.get(path_key) else None
            if field == "checkpoint" and retired and not Path(entry[path_key]).exists():
                if retired["sha256"] != entry.get(hash_key):
                    raise ValueError("stage A retired checkpoint digest changed after audit")
                continue
            if not entry.get(path_key) or sha256(entry[path_key]) != entry.get(hash_key):
                raise ValueError(f"stage A {field} evidence changed after audit")
    hashes = {}
    pairs = set()
    actual_models = []
    for task_path in task_paths:
        task = load(task_path)
        if task["id"] in hashes or not task["phase"].startswith("b"):
            raise ValueError("invalid or duplicate stage B task")
        if task["source_identity"] != manifest["source_identity"] or task["target_tokens"] != 12*MI:
            raise ValueError("stage B source or token contract differs")
        if task["model_id"] not in report["eligible_models"]:
            raise ValueError("stage B contains an unqualified model")
        choice = report.get("selected", {}).get(task["model_id"])
        if not choice or not choice.get("confirmation"):
            raise ValueError("stage B lacks selected recipe evidence")
        selected_task = load(choice["confirmation"][0]["task_path"])
        if any(task.get(key) != selected_task.get(key) for key in RECIPE_FIELDS):
            raise ValueError("stage B changed the selected model or training recipe")
        pair = (task["model_id"], task["seed"])
        if pair in pairs:
            raise ValueError("duplicated model/seed in stage B")
        pairs.add(pair)
        actual_models.append(task["model_id"])
        if gate_kind == "formal":
            if (task["phase"] != "b" or task["seed"] != manifest.get("formal_seed")
                    or task["seed"] in manifest["diagnostic_seeds"] or task.get("resume_checkpoint")
                    or task.get("allow_confirmation")):
                raise ValueError("formal run must start fresh with the predeclared unseen seed")
        elif gate_kind == "repeat":
            if task["phase"] != "b_repeat" or task["seed"] not in manifest.get("repeat_seeds", []) or task.get("allow_confirmation"):
                raise ValueError("invalid repeat seed or premature confirmation")
        elif gate_kind == "confirmation":
            if task["phase"] != "b_confirmation" or not task.get("resume_checkpoint") or not task.get("allow_confirmation"):
                raise ValueError("confirmation requires a completed formal checkpoint")
        else:
            raise ValueError("unknown gate kind")
        hashes[task["id"]] = sha256(task_path)
    if not hashes:
        raise ValueError("no stage B tasks")
    if gate_kind == "formal" and (set(actual_models) != set(report["eligible_models"]) or len(actual_models) != len(set(actual_models))):
        raise ValueError("formal task list must cover each eligible model exactly once")
    if gate_kind == "repeat" and (len(hashes) > 48 or len(set(actual_models)) > 24):
        raise ValueError("repeat budget exceeded")
    gate = {"schema": "pilot-stage-b-gate-v1", "status": "passed",
            "source_identity": manifest["source_identity"], "gate_kind": gate_kind,
            "manifest_path": str(Path(manifest_path).resolve()), "manifest_sha256": sha256(manifest_path),
            "report_path": str(Path(report_path).resolve()), "report_sha256": sha256(report_path),
            "allowed_task_ids": sorted(hashes), "task_sha256": hashes}
    freeze_json(path, gate)
    return gate
