"""Bind the executable Stage B policy to each model's reviewed Stage A provenance."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .io import sha256
from .source_migration3 import read_bound, revised_model


def stage_source_contract(manifest, selected):
    """Resolve A evidence and B execution sources independently from the registered plans."""
    baseline = manifest["source_identity"]
    models = {m["id"]: {"a_source": baseline, "b_source": baseline, "a_model": m["model"], "b_model": m["model"]}
              for m in manifest["models"]}
    if "stage_a_transfer" in manifest:
        from .transfer3 import verify_stage_a_transfer
        transfer = verify_stage_a_transfer(manifest["stage_a_transfer"], manifest=manifest, selected=selected)
        for mid, source in transfer["model_source_identities"].items():
            models[mid]["a_source"] = source
    else:
        transfer = None
    fresh_ids = set()
    if "fresh_shannon" in manifest:
        from .shannon_recipes3 import cohort_subjects, verify_cohort
        rollout = read_bound(manifest["fresh_shannon"])
        fresh_ids = set(rollout["model_ids"])
        if (cohort_subjects(manifest["subjects"], rollout) != manifest["subjects"]
                or transfer and fresh_ids.intersection(transfer["model_ids"])):
            raise ValueError("fresh Shannon cohort and subject or transfer declarations disagree")
        source = verify_cohort(rollout, baseline)
        for mid in fresh_ids:
            models[mid].update(a_source=source, b_source=source)
    contract = {"models": models, "transfer": transfer, "primary_history": {}, "original_choices": {},
                "mixed_ids": set(), "repairs": {}, "pool_floor": False, "carry": set()}
    if "stage_a_dispositions" not in manifest:
        return contract
    review = read_bound(manifest["stage_a_dispositions"])
    primary = read_bound(review["primary_plan"])
    original = read_bound(review["original_inputs"]["original_plan"])
    historic = read_bound(review["original_inputs"]["r07_selected"])
    catalog = {m["id"]: m for m in original["subjects"]}
    active = {m["id"]: m for m in primary["subjects"]}
    subjects = {m["id"]: m for m in manifest["subjects"]}
    rows = review["subjects"]
    if (review["schema"] != "pilot-r3-early-subject-dispositions-v1" or len(catalog) != 344
            or len(active) != 270 or set(rows) != set(catalog) - set(active) or set(catalog) != set(subjects)
            or primary["source_identity"] != baseline or primary.get("fresh_shannon") != manifest.get("fresh_shannon")
            or transfer or set(rows).intersection(fresh_ids)
            or sorted(manifest["tuning_seeds"]) != sorted(review["seeds"]["tuning"])
            or sorted(manifest["diagnostic_seeds"]) != sorted(review["seeds"]["initialization_diagnostics"])
            or Counter(row["mode"] for row in rows.values()) != review["modes"]
            or review["modes"] != dict(carry_recipe=4, compare_finalists=49, confirm_historical_winner=2,
                                       six_recipe_selection=19)):
        raise ValueError("dispositions do not cover this campaign, sources or original fixed seed groups")
    migration = read_bound(primary["stage_a_migration"])
    contract.update(primary_history=migration["completed"], original_choices=read_bound(migration["old_selected"]),
                    mixed_ids=set(migration["exposed_subject_ids"]) - fresh_ids,
                    pool_floor=any(m["model"]["config"].get("pool_support_policy") == "positive-floor-v1"
                                   for m in active.values()),
                    repairs={tid: read_bound(ref) for tid, ref in primary.get("qualified_attempts", {}).items()})
    for mid, declared in active.items():
        if subjects[mid]["model"] != declared["model"] or subjects[mid]["effective_key"] != declared["model_key"]:
            raise ValueError("primary model differs from frozen policy: " + mid)
        if (mid in contract["original_choices"] and mid in selected and mid not in fresh_ids
                and selected[mid] != contract["original_choices"][mid]):
            raise ValueError("frozen primary selection changed: " + mid)
    sources = {arm: read_bound(meta["identity"]) for arm, meta in review["sources"].items()}
    if sources["reference"] != baseline:
        raise ValueError("review baseline differs from the primary source")
    for mid, row in rows.items():
        old = catalog[mid]
        model = old["model"]
        target = {**model, "config": {**model["config"], **row["model_policy"]}}
        if (old["model_key"] != row["historical_model_key"] or subjects[mid]["effective_key"] != old["model_key"]
                or row["target_arm"] not in sources or models[mid]["b_model"] != subjects[mid]["model"]
                or subjects[mid]["model"] != target):
            raise ValueError("reviewed subject model, key or target arm changed: " + mid)
        item = models[mid]
        item.update(a_model=target, a_source=sources[row["target_arm"]], b_source=sources[row["target_arm"]])
        if mid not in selected:
            continue
        choice = selected[mid]
        if choice["recipe_id"] not in row["recipes"]:
            raise ValueError("the selected recipe was not admitted by the reviewed shortlist: " + mid)
        if row["mode"] != "carry_recipe":
            continue
        if (choice != historic[mid] or [choice["recipe_id"]] != row["recipes"]
                or choice["recipe_id"] != row["old_recipe"]):
            raise ValueError("carried winner differs from its original selection: " + mid)
        observed = {}
        for entry in choice["confirmation"]:
            task_path, result_path = entry["task_path"], entry["result_path"]
            from .gate3 import load
            task, result = load(task_path), load(result_path)
            observed[task["seed"]] = (sha256(task_path), sha256(result_path), result["checkpoint"]["sha256"])
        expected = {r["seed"]: (r["task"]["sha256"], r["result"]["sha256"], r["checkpoint"]["sha256"])
                    for r in row["old_a3"]}
        if observed != expected or sorted(expected) != sorted(manifest["tuning_seeds"]):
            raise ValueError("old A3 seeds or artifacts differ from the frozen disposition: " + mid)
        item.update(a_model=model, a_source=original["source_identity"])
        contract["carry"].add(mid)
    return contract


def matches_primary_evidence(contract, task, evidence):
    """Authorize formerly exposed stages by exact bytes and the admitted change in model policy."""
    if task["model_id"] not in contract["mixed_ids"]:
        return False
    row = contract["primary_history"].get(task["id"], {})
    if (row.get("task", {}).get("sha256") != evidence["task_sha256"]
            or row.get("result", {}).get("sha256") != evidence["result_sha256"]):
        return False
    return revised_model(task["model"], pool_floor=contract["pool_floor"]) == contract["models"][task["model_id"]]["a_model"]


def matches_stage_source(contract, task, evidence):
    approved = contract["models"][task["model_id"]]
    if task["model"] == approved["a_model"] and task["source_identity"] == approved["a_source"]:
        return True
    parent_id = task["id"].replace("-a3-", "-a2-") if task["phase"] == "a3" else task["id"]
    repaired = contract.get("repairs", {}).get(parent_id)
    if repaired and repaired["model_id"] == task["model_id"] and repaired["source_identity"] == task["source_identity"]:
        if task["phase"] == "a2":
            return task == repaired
        if task["phase"] == "a3" and task["model"] == approved["a_model"]:
            from .source_migration3 import verify_recipe
            verify_recipe(repaired, task, parent=True)
            return True
    return matches_primary_evidence(contract, task, evidence)


def chain_model(contract, model, choice):
    mid = model["id"]
    approved = contract["models"][mid]
    if mid in contract["mixed_ids"] and choice == contract["original_choices"].get(mid):
        from .gate3 import read_evidence
        task, _, evidence = read_evidence(choice["selection_a1"][0])
        if not matches_primary_evidence(contract, task, evidence):
            raise ValueError("historic chain receipt lacks the original reviewed task and output")
        return read_bound(choice["chain_receipt"])["model"], task["source_identity"]
    return {**model, "model": approved["a_model"]}, approved["a_source"]


def gate_path_for_source(base_gate, source_identity, baseline_identity):
    from .gate3 import manifest_digest
    path = Path(base_gate)
    return path if source_identity == baseline_identity else path.with_name(
        path.stem + "-" + manifest_digest(source_identity)[:20] + path.suffix)
