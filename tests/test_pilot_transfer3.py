"""Synthetic content/identity witnesses; no training or numerical evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pilotlab import gate3 as g, transfer3 as t
from pilotlab.catalog3 import model_key
from pilotlab.prepare_campaign3 import CORE_FILES
from test_pilot_gate3 import formal_inputs, make_campaign, model_row, write_json


OLD_ALLOCATION = '''
def _quad(x):
    return x+1

def allocate(x, kind="quad"):
    if kind == "shannon":
        return x*2
    return _quad(x)

def pool(x):
    return x+3
'''
NEW_ALLOCATION = '''
def _quad(x):
    return x+1

def _shannon_solve(x):
    return x*4

def allocate(x, kind="quad"):
    if kind == "shannon":
        return _shannon_solve(x)
    return _quad(x)

def pool(x):
    return x+3
'''


def snapshot(root, allocation, data):
    source, foundation = root / "source", root / "foundation"
    files = {}
    for relative, content in [(f"source/pilotlab/{name}", "# same training source\n")
                              for name in sorted(CORE_FILES | {"catalog3.py"})] + [
                                  ("foundation/src/tide/allocation.py", allocation),
                                  ("foundation/src/tide/__init__.py", "# same foundation\n")]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        files[str(path)] = g.sha256(path)
    identity = {"schema": "pilot-r3-source-v1", "source_files": files,
                "data_manifest_sha256": g.sha256(data / "manifest.json"),
                "runtime_packages": {"torch": "test", "torch-npu": "test", "numpy": "test"},
                "python_version": "3.test"}
    return write_json(root / "prepared.json", {"source": str(source), "foundation": str(foundation),
                                              "data": str(data), "source_identity": identity})


def single_old_selection(root, identity, model):
    choice = {"recipe_id": "selected", "init_recipe": "baseline", "selection_a1": [],
              "selection_a2": [], "confirmation": [], "diagnostics": []}
    for seed in (101, 202, 303):
        previous = None
        phases = ("init", "a1", "a2", "a3") if seed != 303 else ("init",)
        for phase in phases:
            target = {"init": 128, "a1": g.MI//4, "a2": g.MI, "a3": 2*g.MI}[phase]
            task = {"id": f"{model['id']}-{phase}-{seed}", "model_id": model["id"], "model": model["model"],
                    "source_identity": identity, "phase": phase, "seed": seed, "init_recipe": "baseline",
                    "recipe_id": "selected", "target_tokens": target, "lr": .001, "batch_size": 2,
                    "sequence_length": 128, "cpu_threads": 1, "device": "cpu"}
            resume, start = {}, 0
            if phase in {"a2", "a3"}:
                task.update(resume_checkpoint=previous["checkpoint"]["path"],
                            resume_checkpoint_sha256=previous["checkpoint"]["sha256"])
                start = previous["training"]["tokens"]
                resume = {"path": previous["checkpoint"]["path"], "sha256": previous["checkpoint"]["sha256"],
                          "from_tokens": start}
            directory = root / str(seed) / phase
            task_path = write_json(directory / "task.json", task)
            checkpoint = directory / "checkpoint.pt"
            checkpoint.write_bytes(task["id"].encode())
            result = {"status": "completed", "source_identity": identity, "task_sha256": g.sha256(task_path),
                      "training": {"tokens": target, "from_tokens": start},
                      "health": {"eligible": True, "blocking_reasons": []}, "validation_final": {"nll": 3.},
                      "resume": resume, "recovery": {"checkpoint_load": {"status": "exact"}},
                      "checkpoint": {"path": str(checkpoint), "sha256": g.sha256(checkpoint)},
                      "work": [{"work": {"activity": {"qualified": True, "backbone_total_parameters": 1000,
                                                        "mean_active_backbone_parameters": 500,
                                                        "measured_external_tokens": 128}}}]}
            result_path = write_json(directory / "result.json", result)
            field = {"init": "diagnostics", "a1": "selection_a1", "a2": "selection_a2", "a3": "confirmation"}[phase]
            choice[field].append({"task_path": task_path, "result_path": result_path})
            previous = result
    return {model["id"]: choice}


@pytest.fixture
def transfer_inputs(tmp_path):
    data = tmp_path / "data"
    write_json(data / "manifest.json", {"frozen": "data"})
    old_path = snapshot(tmp_path / "old", OLD_ALLOCATION, data)
    new_path = snapshot(tmp_path / "new", NEW_ALLOCATION, data)
    old, new = g.load(old_path), g.load(new_path)
    model = model_row(0)
    model["model"]["config"]["allocation"] = "quad"
    model["model_key"] = model["effective_key"] = model_key(model["model"])
    selected = single_old_selection(tmp_path / "old-evidence", old["source_identity"], model)
    selected_path = write_json(tmp_path / "selected.json", selected)
    review = {"schema": t.REVIEW_SCHEMA, "status": "passed", "changed_nonshannon_paths": False,
              "quad_pool_contract": "unchanged",
              "old_allocation_sha256": g.sha256(Path(old["foundation"]) / "src/tide/allocation.py"),
              "new_allocation_sha256": g.sha256(Path(new["foundation"]) / "src/tide/allocation.py")}
    digest = hashlib.sha256(b"synthetic equality fixture, not measured training").hexdigest()
    test = {"schema": t.TEST_SCHEMA, "status": "passed", "exact": True,
            "old_source_identity": old["source_identity"], "new_source_identity": new["source_identity"],
            "model_ids": [model["id"]], "tested_paths": list(t.TEST_PATHS), "steps_per_model": 3,
            "observations": {model["id"]: {path: {"old_sha256": digest, "new_sha256": digest} for path in t.TEST_PATHS}}}
    return {"old_prepared_path": old_path, "new_prepared_path": new_path, "old_selected_path": selected_path,
            "model_ids": [model["id"]], "allocation_review_path": write_json(tmp_path / "review.json", review),
            "quad_test_evidence_path": write_json(tmp_path / "validation.json", test)}


def manifest_for(report):
    subjects = [{"id": mid, "model": row["model"], "model_key": row["model_key"], "effective_key": row["model_key"]}
                for mid, row in report["models"].items()]
    return {"source_identity": report["new_source_identity"], "subjects": subjects, "models": subjects,
            "tuning_seeds": report["tuning_seeds"], "diagnostic_seeds": report["diagnostic_seeds"], "formal_seed": 404}


def test_transfer_rechecks_the_complete_old_chain_and_preserves_its_identity(transfer_inputs, tmp_path):
    report = t.build_stage_a_transfer(**transfer_inputs)
    manifest = manifest_for(report)
    selected = g.load(transfer_inputs["old_selected_path"])
    report_path = write_json(tmp_path / "transfer.json", report)
    entry = {"path": report_path, "sha256": g.sha256(report_path)}
    validated = t.verify_stage_a_transfer(entry, manifest=manifest, selected=selected)
    assert validated["model_source_identities"] == {"subject-000": report["old_source_identity"]}
    assert len(report["evidence"]) == 9 and all("checkpoint_path" in row for row in report["evidence"])
    assert all(g.load(row["task_path"])["source_identity"] == report["old_source_identity"] for row in report["evidence"])
    assert report["source_comparison"]["allocation_contract"]["quad_pool_contract"] == "unchanged"
    manifest["stage_a_transfer"] = entry
    audit = g.audit_stage_a(manifest, selected, [], remaining_resources_ok=True)
    assert "subject-000" in audit["eligible_models"]
    assert audit["transferred_model_ids"] == ["subject-000"]
    assert not any(isinstance(reason, dict) and reason.get("model_id") == "subject-000" for reason in audit["blocking_reasons"])


@pytest.mark.parametrize("damage", ["training_source", "foundation_source", "data", "runtime", "quad_body"])
def test_transfer_rejects_changes_outside_the_reviewed_shannon_path(transfer_inputs, damage):
    path = transfer_inputs["new_prepared_path"]
    prepared = g.load(path)
    identity = prepared["source_identity"]
    if damage == "data":
        identity["data_manifest_sha256"] = "f"*64
    elif damage == "runtime":
        identity["runtime_packages"]["torch"] = "different"
    else:
        relative = {"training_source": "source/pilotlab/worker3.py",
                    "foundation_source": "foundation/src/tide/__init__.py",
                    "quad_body": "foundation/src/tide/allocation.py"}[damage]
        changed = Path(path).parent / relative
        content = changed.read_text()
        changed.write_text(content.replace("return x+1", "return x+2") if damage == "quad_body" else content+"# changed\n")
        identity["source_files"][str(changed)] = g.sha256(changed)
    write_json(path, prepared)
    with pytest.raises(ValueError):
        t.build_stage_a_transfer(**transfer_inputs)


@pytest.mark.parametrize("damage", ["status_only", "wrong_identity", "wrong_models", "short_test", "missing_backward", "different_state"])
def test_status_passed_cannot_substitute_for_complete_independent_training_evidence(transfer_inputs, damage):
    path = transfer_inputs["quad_test_evidence_path"]
    test = g.load(path)
    if damage == "status_only":
        test = {"schema": t.TEST_SCHEMA, "status": "passed"}
    elif damage == "wrong_identity":
        test["new_source_identity"] = test["old_source_identity"]
    elif damage == "wrong_models":
        test["model_ids"] = ["another-subject"]
    elif damage == "short_test":
        test["steps_per_model"] = 2
    elif damage == "missing_backward":
        test["observations"]["subject-000"].pop("backward")
    else:
        test["observations"]["subject-000"]["state"]["new_sha256"] = "f"*64
    write_json(path, test)
    with pytest.raises(ValueError):
        t.build_stage_a_transfer(**transfer_inputs)


@pytest.mark.parametrize("damage", ["incomplete_a3", "broken_continuation", "wrong_result_identity", "corrupt_init_checkpoint"])
def test_transferred_selection_still_requires_all_old_tasks_results_and_checkpoints(transfer_inputs, damage):
    selected = g.load(transfer_inputs["old_selected_path"])
    choice = selected["subject-000"]
    if damage == "incomplete_a3":
        choice["confirmation"].pop()
        write_json(transfer_inputs["old_selected_path"], selected)
    elif damage == "corrupt_init_checkpoint":
        result = g.load(choice["diagnostics"][0]["result_path"])
        Path(result["checkpoint"]["path"]).write_bytes(b"corrupt")
    else:
        path = choice["confirmation"][0]["result_path"]
        result = g.load(path)
        if damage == "broken_continuation":
            result["resume"]["from_tokens"] = 0
        else:
            result["source_identity"] = g.load(transfer_inputs["new_prepared_path"])["source_identity"]
        write_json(path, result)
    with pytest.raises(ValueError):
        t.build_stage_a_transfer(**transfer_inputs)


@pytest.mark.parametrize("damage", ["report_identity", "old_selected_file", "new_selection", "review_hash", "not_subject"])
def test_bound_transfer_cannot_be_relabelled_or_mutated_after_review(transfer_inputs, tmp_path, damage):
    report = t.build_stage_a_transfer(**transfer_inputs)
    manifest = manifest_for(report)
    selected = g.load(transfer_inputs["old_selected_path"])
    if damage == "report_identity":
        report["old_source_identity"] = report["new_source_identity"]
    elif damage == "old_selected_file":
        Path(transfer_inputs["old_selected_path"]).write_text(json.dumps(selected, indent=2))
    elif damage == "new_selection":
        selected["subject-000"]["recipe_id"] = "changed"
    elif damage == "review_hash":
        report["allocation_review"]["sha256"] = "f"*64
    else:
        manifest["subjects"] = []
    path = write_json(tmp_path / "transfer.json", report)
    with pytest.raises(ValueError):
        t.verify_stage_a_transfer({"path": path, "sha256": g.sha256(path)}, manifest=manifest, selected=selected)


def test_formal_gate_accepts_old_quad_recipe_evidence_only_with_fresh_new_source_tasks(transfer_inputs, tmp_path):
    report = t.build_stage_a_transfer(**transfer_inputs)
    transferred = manifest_for(report)["subjects"][0]
    subjects = [transferred] + [model_row(i) for i in range(1, 216)]
    campaign = make_campaign(tmp_path / "new-a", identity=report["new_source_identity"], subjects=subjects)
    manifest, selected, matches = campaign
    selected.update(g.load(transfer_inputs["old_selected_path"]))
    path = write_json(tmp_path / "transfer.json", report)
    manifest["stage_a_transfer"] = {"path": path, "sha256": g.sha256(path)}
    original_hashes = {row["task_path"]: g.sha256(row["task_path"]) for row in report["evidence"]}
    manifest_path, report_path, tasks = formal_inputs(campaign, tmp_path)
    gate = g.create_gate(tmp_path / "gate.json", manifest_path=manifest_path, report_path=report_path, task_paths=tasks)
    assert gate["source_identity"] == report["new_source_identity"] and len(gate["allowed_task_ids"]) == 219
    assert original_hashes == {path: g.sha256(path) for path in original_hashes}
    formal_path = next(path for path in tasks if g.load(path)["model_id"] == "subject-000")
    original = g.load(formal_path)
    for damage in ("old_source", "old_checkpoint"):
        task = deepcopy(original)
        if damage == "old_source":
            task["source_identity"] = report["old_source_identity"]
        else:
            task["resume_checkpoint"] = report["evidence"][0]["checkpoint_path"]
        write_json(formal_path, task)
        with pytest.raises(ValueError):
            g.create_gate(tmp_path / f"{damage}.json", manifest_path=manifest_path, report_path=report_path, task_paths=tasks)
