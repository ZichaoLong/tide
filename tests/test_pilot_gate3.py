"""Small destructive witnesses against round-three scientific admission.

Artifacts are synthetic JSON/state bytes, never trained models. A complete
216-subject fixture establishes a passing control before individual evidence,
continuation, or formal-manifest contracts are broken.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pilotlab import gate3 as g


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return str(path)


def model_row(index):
    spec = {"family": "tide", "config": {"d": 64, "memory": "K",
            "topology": {"kind": "fc", "n": 4, "layers": index + 1}}}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return {"id": f"subject-{index:03}", "model": spec, "effective_key": key, "model_key": key}


def make_campaign(root, *, subject_count=216, identity=None, subjects=None):
    identity = identity or {"source": "frozen-project", "foundation": "frozen-foundation", "data": "frozen-data"}
    subjects = subjects or [model_row(i) for i in range(subject_count)]
    controls = [{"id": "dense-total", "model": {"family": "transformer", "d": 64, "layers": 4},
                 "effective_key": "dense-total"},
                {"id": "dense-active", "model": {"family": "transformer", "d": 64, "layers": 2},
                 "effective_key": "dense-active"},
                {"id": "moe-joint", "model": {"family": "moe", "d": 64, "layers": 2,
                                                "experts": 4, "top_k": 2},
                 "effective_key": "moe-joint"}]
    models = subjects + controls
    manifest = {"source_identity": identity, "subjects": subjects, "models": models,
                "tuning_seeds": [101, 202], "diagnostic_seeds": [101, 202, 303], "formal_seed": 404}
    selected = {}

    def artifact(model, phase, seed, previous=None):
        target = {"init": 1024, "a1": g.MI // 4, "a2": g.MI, "a3": 2 * g.MI}[phase]
        request = {"id": f'{model["id"]}-{phase}-{seed}', "model_id": model["id"],
                   "model": model["model"], "phase": phase, "seed": seed,
                   "recipe_id": "recipe-0", "init_recipe": "baseline", "target_tokens": target,
                   "source_identity": identity, "lr": .001, "batch_size": 2, "sequence_length": 128,
                   "cpu_threads": 1, "device": "cpu"}
        from_tokens = 0
        resume = {}
        if previous:
            parent = g.load(previous["result_path"])
            request.update(resume_checkpoint=parent["checkpoint"]["path"],
                           resume_checkpoint_sha256=parent["checkpoint"]["sha256"])
            from_tokens = parent["training"]["tokens"]
            resume = {"path": parent["checkpoint"]["path"], "sha256": parent["checkpoint"]["sha256"],
                      "from_tokens": from_tokens, "state_reused": True, "fallback_used": False}
        directory = root / model["id"] / f"{phase}-{seed}"
        task_path = write_json(directory / "task.json", request)
        checkpoint = directory / "checkpoint.pt"
        checkpoint.write_bytes(json.dumps({"task": request["id"], "tokens": target}).encode())
        total = 500 if model["id"] == "dense-active" else 1000
        active = 1000 if model["id"] == "dense-total" else 500
        answer = {"status": "completed", "source_identity": identity,
                  "task_sha256": g.sha256(task_path),
                  "training": {"tokens": target, "from_tokens": from_tokens,
                               "new_tokens": target - from_tokens},
                  "health": {"eligible": True, "blocking_reasons": []},
                  "validation_final": {"nll": 3.}, "resume": resume,
                  "work": [{"work": {"activity": {"qualified": True,
                            "backbone_total_parameters": total,
                            "mean_active_backbone_parameters": active,
                            "measured_external_tokens": 128}}}],
                  "initialization": {"seed": seed, "recipe": "baseline"},
                  "recovery": {"checkpoint_load": {"status": "exact", "scope": "end_checkpoint_readback"}},
                  "checkpoint": {"path": str(checkpoint), "sha256": g.sha256(checkpoint)}}
        return {"task_path": task_path, "result_path": write_json(directory / "result.json", answer)}

    for model in models:
        choice = {"recipe_id": "recipe-0", "init_recipe": "baseline",
                  "diagnostics": [artifact(model, "init", seed) for seed in manifest["diagnostic_seeds"]],
                  "selection_a1": [], "selection_a2": [], "confirmation": []}
        for seed in manifest["tuning_seeds"]:
            a1 = artifact(model, "a1", seed)
            a2 = artifact(model, "a2", seed, a1)
            a3 = artifact(model, "a3", seed, a2)
            choice["selection_a1"].append(a1)
            choice["selection_a2"].append(a2)
            choice["confirmation"].append(a3)
        selected[model["id"]] = choice
    matches = []
    for subject in subjects:
        for kind, control in (("dense_total", "dense-total"), ("dense_active", "dense-active"),
                              ("moe_joint", "moe-joint")):
            matches.append({"subject_id": subject["id"], "kind": kind, "control_id": control,
                            "qualified": True, "total_relative_error": 0.,
                            "active_relative_error": 0., "ratio_absolute_error": 0.})
    return manifest, selected, matches


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    return make_campaign(tmp_path_factory.mktemp("gate3-campaign"))


@pytest.fixture(scope="module")
def expanded_campaign(tmp_path_factory):
    from pilotlab.catalog3 import subject_catalog_contract
    result = make_campaign(tmp_path_factory.mktemp("gate3-expanded-campaign"), subject_count=344)
    result[0]["catalog_contract"] = subject_catalog_contract(include_supplemental=True)
    return result


def audit(campaign):
    manifest, selected, matches = campaign
    return g.audit_stage_a(manifest, selected, matches, remaining_resources_ok=True)


def overlay(entry, tmp_path, field, edit):
    """Change one artifact without mutating the shared passing fixture."""
    value = g.load(entry[field])
    edit(value)
    entry[field] = write_json(tmp_path / Path(entry[field]).name, value)


def test_complete_chain_is_a_passing_control(campaign):
    report = audit(campaign)
    assert report["status"] == "passed", report["blocking_reasons"]
    assert report["eligible_subject_count"] == 216
    assert len(report["eligible_models"]) == 219


def test_expanded_catalog_requires_the_explicit_finite_contract(expanded_campaign):
    report = audit(expanded_campaign)
    assert report["status"] == "passed", report["blocking_reasons"]
    assert report["eligible_subject_count"] == 344 and len(report["matches"]) == 1032
    without_contract = deepcopy(expanded_campaign)
    without_contract[0].pop("catalog_contract")
    assert "subject_catalog_not_216_unique_effective_models" in audit(without_contract)["blocking_reasons"]


@pytest.mark.parametrize("field,value", [("subject_count_limit", 345), ("control_count_limit", 1033),
                                         ("primary_whole_parameter_limit", 2_000_000_000)])
def test_catalog_contract_cannot_expand_counts_or_relabel_the_original_cap(expanded_campaign, field, value):
    damaged = deepcopy(expanded_campaign)
    damaged[0]["catalog_contract"][field] = value
    report = audit(damaged)
    assert report["status"] == "blocked"
    assert any(isinstance(reason, str) and reason.startswith("invalid_catalog_contract:")
               for reason in report["blocking_reasons"])


@pytest.mark.parametrize("damage", ["empty", "duplicate"])
def test_seed_lists_cannot_make_missing_evidence_vacuously_qualified(campaign, damage):
    damaged = deepcopy(campaign)
    manifest, selected, _ = damaged
    if damage == "empty":
        manifest["tuning_seeds"] = []
        manifest["diagnostic_seeds"] = []
        for choice in selected.values():
            for key in ("selection_a1", "selection_a2", "confirmation", "diagnostics"):
                choice[key] = []
    else:
        manifest["tuning_seeds"] = [101, 101]
        manifest["diagnostic_seeds"] = [101, 101, 101]
        for choice in selected.values():
            for key in ("selection_a1", "selection_a2", "confirmation"):
                choice[key] = [choice[key][0], choice[key][0]]
            choice["diagnostics"] = [choice["diagnostics"][0]] * 3
    assert audit(damaged)["status"] == "blocked"


@pytest.mark.parametrize("damage", ["missing_subject", "duplicate_effective_model", "missing_control", "self_control"])
def test_complete_distinct_subject_and_real_control_coverage_is_required(campaign, damage):
    damaged = deepcopy(campaign)
    manifest, selected, matches = damaged
    if damage == "missing_subject":
        missing = manifest["subjects"][0]["id"]
        manifest["models"] = [m for m in manifest["models"] if m["id"] != missing]
        selected.pop(missing)
    elif damage == "duplicate_effective_model":
        manifest["subjects"][1]["effective_key"] = manifest["subjects"][0]["effective_key"]
    elif damage == "missing_control":
        matches.pop()
    else:
        matches[0]["control_id"] = matches[0]["subject_id"]
    assert audit(damaged)["status"] == "blocked"


@pytest.mark.parametrize("damage", ["missing_a2", "different_seed_checkpoint", "wrong_task_hash", "corrupt_checkpoint"])
def test_end_checkpoint_readback_cannot_substitute_for_real_continuation(campaign, tmp_path, damage):
    damaged = deepcopy(campaign)
    choice = damaged[1][damaged[0]["subjects"][0]["id"]]
    if damage == "missing_a2":
        choice["selection_a2"] = []
    elif damage == "different_seed_checkpoint":
        other = g.load(choice["selection_a2"][1]["result_path"])["checkpoint"]

        def break_resume(answer):
            answer["resume"].update(path=other["path"], sha256=other["sha256"])

        overlay(choice["confirmation"][0], tmp_path, "result_path", break_resume)
    elif damage == "wrong_task_hash":
        overlay(choice["confirmation"][0], tmp_path, "result_path",
                lambda answer: answer.update(task_sha256="0" * 64))
    else:
        bad = tmp_path / "corrupted.pt"
        bad.write_bytes(b"truncated checkpoint")
        overlay(choice["confirmation"][0], tmp_path, "result_path",
                lambda answer: answer["checkpoint"].update(path=str(bad)))
    assert audit(damaged)["status"] == "blocked"


def test_parent_checkpoint_bytes_must_match_the_continuation_evidence(campaign):
    choice = campaign[1][campaign[0]["subjects"][0]["id"]]
    checkpoint = Path(g.load(choice["selection_a2"][0]["result_path"])["checkpoint"]["path"])
    original = checkpoint.read_bytes()
    try:
        checkpoint.write_bytes(b"damaged selected parent state")
        assert audit(campaign)["status"] == "blocked"
    finally:
        checkpoint.write_bytes(original)


@pytest.mark.parametrize("damage", ["missing_activity", "different_parameter_totals", "weighted_mismatch"])
def test_actual_activity_overrides_claimed_zero_matching_errors(campaign, tmp_path, damage):
    damaged = deepcopy(campaign)
    choice = damaged[1][damaged[0]["subjects"][0]["id"]]
    if damage == "missing_activity":
        overlay(choice["confirmation"][0], tmp_path, "result_path", lambda answer: answer.pop("work"))
    elif damage == "different_parameter_totals":
        overlay(choice["confirmation"][0], tmp_path, "result_path",
                lambda answer: answer["work"][-1]["work"]["activity"].update(backbone_total_parameters=1001))
    else:
        # The ordinary mean is 500, but the external-token-weighted mean is
        # 450. That breaks the joint MoE match despite zero claimed errors.
        for index, active, tokens in ((0, 600, 128), (1, 400, 384)):
            overlay(choice["confirmation"][index], tmp_path / str(index), "result_path",
                    lambda answer, active=active, tokens=tokens: answer["work"][-1]["work"]["activity"].update(
                        mean_active_backbone_parameters=active, measured_external_tokens=tokens))
    report = audit(damaged)
    assert report["status"] == "blocked"
    if damage == "weighted_mismatch":
        assert report["measured_activity"][damaged[0]["subjects"][0]["id"]]["mean_active_backbone_parameters"] == 450


@pytest.mark.parametrize("active,expected", [(520., "passed"), (520.0001, "blocked")])
def test_measured_ratio_keeps_exact_original_boundary(campaign, tmp_path, active, expected):
    damaged = deepcopy(campaign)
    for index, entry in enumerate(damaged[1]["moe-joint"]["confirmation"]):
        overlay(entry, tmp_path / str(index), "result_path",
                lambda answer: answer["work"][-1]["work"]["activity"].update(mean_active_backbone_parameters=active))
    assert audit(damaged)["status"] == expected


def formal_inputs(campaign, tmp_path):
    manifest, selected, _ = campaign
    report = audit(campaign)
    assert report["status"] == "passed", report["blocking_reasons"]
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    report_path = write_json(tmp_path / "stage-a-report.json", report)
    paths = []
    for model in manifest["models"]:
        choice = selected[model["id"]]
        request = g.load(choice["confirmation"][0]["task_path"])
        request.pop("resume_checkpoint", None)
        request.pop("resume_checkpoint_sha256", None)
        request.update(id=f'{model["id"]}-formal', phase="b", seed=manifest["formal_seed"],
                       target_tokens=12 * g.MI, source_identity=manifest["source_identity"])
        paths.append(write_json(tmp_path / "formal" / f'{model["id"]}.json', request))
    return manifest_path, report_path, paths


def test_gate_binds_the_complete_fixed_formal_list_and_artifact_hashes(campaign, tmp_path):
    manifest_path, report_path, paths = formal_inputs(campaign, tmp_path)
    gate = g.create_gate(tmp_path / "gate.json", manifest_path=manifest_path,
                         report_path=report_path, task_paths=paths)
    assert gate["manifest_sha256"] == g.sha256(manifest_path)
    assert gate["report_sha256"] == g.sha256(report_path)
    assert len(gate["allowed_task_ids"]) == 219
    for path in paths:
        assert gate["task_sha256"][g.load(path)["id"]] == g.sha256(path)


def test_selected_artifact_change_after_audit_cannot_change_the_formal_recipe(campaign, tmp_path):
    manifest_path, report_path, paths = formal_inputs(campaign, tmp_path)
    report = g.load(report_path)
    model_id = g.load(paths[0])["model_id"]
    selected_path = Path(report["selected"][model_id]["confirmation"][0]["task_path"])
    original = selected_path.read_bytes()
    try:
        changed = g.load(selected_path)
        changed["lr"] *= 100
        write_json(selected_path, changed)
        formal = g.load(paths[0])
        formal["lr"] = changed["lr"]
        write_json(paths[0], formal)
        with pytest.raises((ValueError, KeyError)):
            g.create_gate(tmp_path / "gate.json", manifest_path=manifest_path,
                          report_path=report_path, task_paths=paths)
    finally:
        selected_path.write_bytes(original)


@pytest.mark.parametrize("damage", ["one_subject_only", "duplicate_model", "wrong_recipe", "changed_lr",
                                     "reused_seed", "changed_report"])
def test_gate_cannot_authorize_an_incomplete_or_unselected_formal_manifest(campaign, tmp_path, damage):
    manifest_path, report_path, paths = formal_inputs(campaign, tmp_path)
    if damage == "one_subject_only":
        paths = paths[:1]
    elif damage == "duplicate_model":
        duplicate = g.load(paths[0])
        duplicate["id"] += "-extra"
        paths.append(write_json(tmp_path / "extra.json", duplicate))
    elif damage == "changed_report":
        # A passed label cannot override the required qualified-subject count.
        answer = g.load(report_path)
        answer["eligible_subject_count"] = 199
        write_json(report_path, answer)
    else:
        request = g.load(paths[0])
        if damage == "wrong_recipe":
            request["recipe_id"] = "unselected-lr"
        elif damage == "changed_lr":
            request["lr"] *= 100
        else:
            request["seed"] = campaign[0]["tuning_seeds"][0]
        write_json(paths[0], request)
    with pytest.raises((ValueError, KeyError)):
        g.create_gate(tmp_path / "gate.json", manifest_path=manifest_path,
                      report_path=report_path, task_paths=paths)
