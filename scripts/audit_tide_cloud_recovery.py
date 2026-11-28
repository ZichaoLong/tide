"""Replay the stopped R3 dispatch and potential stage parents from archived payload receipts."""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import errno
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from tide_cloud_allocation import REPO
from tide_cloud_controller import normalized
from tide_cloud_dispatch import descriptor, objects_needed
from tide_portable import read, require, sha
from pilotlab.checkpoint_archive3 import archive_record, checkpoint_reads
from pilotlab.cloud_campaign3 import CloudCampaign, advance
from pilotlab.gate3 import audit_stage_a, choose_recipes, continuation_matches, eligible, manifest_digest, read_evidence
from pilotlab.prepare_resume3 import reference
from pilotlab.round3 import TUNING_SEEDS
from pilotlab.runtime_migration3 import read_plan
from pilotlab.scheduler3 import Blocked


def prohibit_local_checkpoints():
    """Simulate cache eviction for old .pt artifacts throughout the unmodified production entry points."""
    original_open, original_stat = Path.open, Path.stat
    checked = Counter()

    def absent(path):
        return path.suffix == ".pt"

    def guarded_open(path, *args, **kwargs):
        if absent(path):
            checked["payload_open"] += 1
            raise FileNotFoundError(errno.ENOENT, "archive-only recovery proof cannot open the local payload", str(path))
        return original_open(path, *args, **kwargs)

    def guarded_stat(path, *args, **kwargs):
        if absent(path):
            checked["payload_stat"] += 1
            raise FileNotFoundError(errno.ENOENT, "checkpoint is evicted for archive-only review", str(path))
        return original_stat(path, *args, **kwargs)

    stack = ExitStack()
    stack.enter_context(patch.object(Path, "open", guarded_open))
    stack.enter_context(patch.object(Path, "stat", guarded_stat))
    return stack, checked


def trace_candidates(campaign, provenance, mids, count):
    """Admit raw evidence, replay complete seed groups and retain any parent that could still be chosen."""
    groups = defaultdict(lambda: defaultdict(list))
    touched, skipped, diagnostics = {}, {}, []
    archived_diagnostics, unused_diagnostics = [], []
    for item in campaign.records.values():
        task = read(item["task_path"])
        if task["model_id"] not in mids:
            continue
        _, result, evidence = read_evidence(item)
        phase = task["phase"]
        reasons = eligible(task, result, require_resume=phase in ("a2", "a3"))
        require(not reasons, f"stored unfinished-subject observation is unqualified: {task['id']}, {reasons}")
        count["observations_" + phase] += 1
        if phase in ("a1", "a2", "a3"):
            groups[task["model_id"]][phase].append((task, result, evidence))
        else:
            diagnostics.append((task["id"], evidence["task_sha256"], evidence["result_sha256"]))
            cp = result.get("checkpoint") or {}
            if cp.get("path") and archive_record(cp["path"], cp["sha256"], cp.get("size_bytes")):
                archived_diagnostics.append(cp["sha256"])
            elif cp.get("path"):
                # These completed probes have no downstream checkpoint consumer. Their
                # measured results and task hashes above remain mandatory for admission.
                unused_diagnostics.append(task["id"])

    def parent(t, r, evidence):
        cp = r["checkpoint"]
        obj = archive_record(cp["path"], cp["sha256"], cp.get("size_bytes"))
        require(obj, "future selection can require a non-archived generation: " + cp["path"])
        _, _, verified = read_evidence(evidence, checkpoint=True)
        require(verified.get("checkpoint_state") == "original_verified_in_archive", "future parent required local bytes")
        touched[t["id"]] = dict(path=cp["path"], sha256=cp["sha256"], size=obj["size"])

    for model in sorted(mids):
        waves = groups[model]
        review = campaign.review["subjects"].get(model, {})
        recipes = set(review.get("recipes", ["r0", "r1", "r2", "r3", "r4", "r5"]))
        allowed = {"a1": recipes, "a2": None, "a3": None}
        by_seed = {}
        for phase, next_phase in (("a1", "a2"), ("a2", "a3"), ("a3", None)):
            rows = waves.get(phase, [])
            if not rows:
                continue
            ids = allowed[phase]
            require(ids and {t["recipe_id"] for t, _, _ in rows} <= ids,
                    f"{model} contains an unadmitted {phase} recipe or its previous wave is incomplete")
            seen = {(t["recipe_id"], t["seed"]) for t, _, _ in rows}
            require(len(seen) == len(rows) and all(seed in TUNING_SEEDS for _, seed in seen), "invalid recipe seed group")
            current = {}
            for task, result, evidence in rows:
                key = (task["recipe_id"], task["seed"])
                if phase != "a1":
                    require(key in by_seed and continuation_matches(by_seed[key][:2], task, result,
                            mixed=model in provenance["mixed_ids"]), "candidate did not consume its selected parent")
                current[key] = (task, result, evidence)
            complete = seen == {(recipe, seed) for recipe in ids for seed in TUNING_SEEDS}
            keep = min(2, len(ids)) if phase == "a1" else 1
            if complete and next_phase:
                choice = choose_recipes([t for t, _, _ in rows], {t["id"]: r for t, r, _ in rows}, TUNING_SEEDS, keep=keep)
                require(len(choice["selected"]) == keep, "ranking lacks its prescribed complete groups")
                winners = {v["recipe_id"] for v in choice["selected"]}
                allowed[next_phase] = winners
                count["replayed_ranking_" + phase] += 1
            else:
                winners = ids
                if next_phase:
                    require(not waves.get(next_phase), "unfinished ranking has downstream observations")
            for task, result, evidence in rows:
                if task["recipe_id"] in winners:
                    if phase == "a1" and complete and waves.get("a2"):
                        # When the entire A2 comparison is already known, the A1 of
                        # its losing finalist can lawfully have been consumed and retired.
                        a2 = waves["a2"]
                        if len(a2) == keep * len(TUNING_SEEDS):
                            final = choose_recipes([t for t, _, _ in a2], {t["id"]:r for t,r,_ in a2}, TUNING_SEEDS, keep=1)
                            if task["recipe_id"] != final["selected"][0]["recipe_id"]:
                                skipped[task["id"]] = "A2 winner already determined"
                                continue
                    parent(task, result, evidence)
                else:
                    skipped[task["id"]] = phase + " ranking excludes recipe"
            by_seed = current
    return dict(potential_parent_tasks=len(touched), excluded_by_replayed_ranking=len(skipped),
                diagnostic_or_capacity_observations=len(diagnostics), diagnostic_payloads_archived=len(archived_diagnostics),
                diagnostics_without_payload_and_without_continuation_consumer=len(unused_diagnostics),
                candidate_receipts_sha256=manifest_digest(touched),
                diagnostics_sha256=manifest_digest(diagnostics), excluded_tasks_sha256=manifest_digest(skipped))


def dispatch_frontier(area, rows, output):
    spec = read(area / "payload/production.json")
    subjects, objects, inputs, source_digests = set(), {}, {}, set()
    records = []
    counts = Counter()
    for i, row in enumerate(rows):
        task = normalized(dict(row))
        original = read(task["path"])
        d, planned = descriptor(area, task, spec["jobs"][i % len(spec["jobs"])], f"dry-{i}", spec)
        file = output / (d["runtime_plan_sha256"] + ".json")
        file.write_bytes(base64.b64decode(d["runtime_plan_bytes"], validate=True))
        exact, history = read_plan(file, d["runtime_plan_sha256"], original, d["fingerprint"])
        require(exact == planned and hashlib.sha256(base64.b64decode(d["task_bytes"], validate=True)).hexdigest()
                == task["fingerprint"], "descriptor or runtime-plan identity changed")
        original_path = task["path"]
        require(sha(original_path) == d["fingerprint"] and task["threads"] > 0 and task["ram_mib"] > 0
                and (task["backend"] == "cpu" or task["hbm_mib"] > 0), "invalid task or reservation")
        if d["resume"]:
            old = history[-1]["checkpoint"]
            require((old["sha256"], old["size_bytes"], old["steps"]) ==
                    (d["resume"]["sha256"], d["resume"]["size"], d["resume"]["steps"]), "interruption SHA or progress changed")
            counts["paused"] += 1
        for digest, (size, path) in objects_needed(area, d).items():
            if path is None:
                path = history[-1]["checkpoint"]["path"]
            if Path(path).suffix == ".pt":
                proof = archive_record(path, digest, size)
                require(proof, "descriptor requires an unarchived generation: " + path)
            else:
                proof = inputs.get(path)
                if not proof:
                    require(Path(path).stat().st_size == size and sha(path) == digest, "frozen data or auxiliary input differs")
                    inputs[path] = digest
                require(inputs[path] == digest, "different checksums for the same frozen data input")
            objects[digest] = size
        base = d.get("base_manifest") or read(area / "payload/bases" / d["source"] / "base.json")
        require(base["identity"] == d["source"] == manifest_digest(original["source_identity"]), "frozen source tree changed")
        if d["source"] not in source_digests:
            for path, digest in original["source_identity"]["source_files"].items():
                name = base["paths"][path]
                source = area / "payload/bases" / d["source"] / name
                require(base["files"][name]["sha256"] == digest == sha(source), "frozen archive implementation differs")
            require(base["files"]["data/manifest.json"]["sha256"] == original["source_identity"]["data_manifest_sha256"],
                    "dataset does not match task identity")
            source_digests.add(d["source"])
        subjects.add(original["model_id"])
        counts[task["backend"]] += 1
        counts[task["phase"]] += 1
        counts["retained_queue" if row.get("legacy_id") else "newly_released"] += 1
        records.append(dict(path=original_path, fingerprint=d["fingerprint"], runtime_plan_sha256=d["runtime_plan_sha256"],
                            source=d["source"], backend=d["backend"], start_tokens=d["start_tokens"]))
    return dict(tasks=len(records), subjects=len(subjects), counts=dict(counts), source_identities=sorted(source_digests),
                required_objects=len(objects), data_or_auxiliary_files=len(inputs),
                required_object_hashes_and_sizes_sha256=manifest_digest(objects), descriptors_sha256=manifest_digest(records))


def audit(area):
    from pilotlab.stage_sources3 import stage_source_contract
    with tempfile.TemporaryDirectory(prefix="tide-offsite-audit-") as temp:
        root = Path(temp)
        evicted, counts = prohibit_local_checkpoints()
        with evicted:
            inventory = area / "initial-tasks.json"
            result = advance(root, inventory, {})
            require(result["phase"] == "stage-a-subjects" and not result["blocked"], "original stopped DAG is blocked")
            campaign = CloudCampaign(root / "campaign", read(inventory), {})
            provenance = stage_source_contract(campaign.manifest(), campaign.selected)
            require(len(campaign.archives) == 1 and Path(campaign.archives[0]["path"]).name == "remaining-checkpoints-proof.json",
                    "a complete, verified East checkpoint inventory is required")
            with checkpoint_reads(campaign.archives, authoritative=True):
                selected = audit_stage_a(campaign.manifest(), campaign.selected, [], remaining_resources_ok=False)
                model_errors = [reason for reason in selected["blocking_reasons"] if isinstance(reason, dict)]
                require(selected["eligible_subject_count"] == len(campaign.selected) and not model_errors,
                        f"previous selection failed after eviction: {model_errors[:2]}")
                mids = {m["id"] for m in campaign.subjects} - set(campaign.selected)
                observations = Counter()
                candidates = trace_candidates(campaign, provenance, mids, observations)
                requests = dispatch_frontier(area, result["tasks"], root)
                try:
                    campaign.controls()
                except Blocked as error:
                    control_gate = str(error)
                else:
                    raise ValueError("control catalog was constructed before all subject A3 pairs")
                old_controls = [r["task_path"] for r in campaign.records.values()
                                if read(r["task_path"])["model"]["family"] != "tide"]
                require(not old_controls and len(mids) == 76, "unexpected prior control or changed original frontier")
            require(counts["payload_open"] == 0 and requests["tasks"] == 281 and requests["counts"]["paused"] == 69
                    and requests["counts"]["retained_queue"] == 261, "evicted dispatch lost tasks or required local payloads")
            return dict(schema="tide-r3-candidate-archive-frontier/v1", observed_at=datetime.now(timezone.utc).isoformat(),
                archive=campaign.archives[0], archive_inventory_sha256=read(campaign.archives[0]["path"])["sha256"],
                science_inputs=campaign.inputs, initial_inventory=reference(inventory), selected_subjects=len(campaign.selected),
                previously_selected_admitted=selected["eligible_subject_count"], outstanding_subjects=len(mids),
                archived_candidate_observations=dict(observations), candidate_parents=candidates,
                dispatch=requests, controls=dict(historical_control_tasks=len(old_controls), original_barrier=control_gate,
                    matching_input="A3 activity aggregates for all 344 subjects", next_phase="fresh matched-control capacity and Stage A"),
                rejected_attempts_to_open_local_checkpoint=counts["payload_open"], simulated_missing_file_stats=counts["payload_stat"],
                stage_b_gate="closed", selected_audit_scope="No matched controls or resource approval supplied.",
                method="Simulated eviction of every local .pt at pathlib stat/open. Ran the production DAG, frozen descriptor and plan validators; independently replayed complete candidate rankings and all possible next-stage parent loads.")


def main():
    from scripts.review_io import write_review_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--area", type=Path, default=REPO / ".local/modelarts-r3/control")
    p.add_argument("--evidence", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.area.resolve())
    write_review_json(args.evidence, result)
    print(json.dumps({k: result[k] for k in ("selected_subjects", "outstanding_subjects", "candidate_parents", "dispatch")}))


if __name__ == "__main__":
    main()
