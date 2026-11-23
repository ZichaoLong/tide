"""Synthetic destructive witnesses for consumed-parent retirement receipts."""
from copy import deepcopy
from pathlib import Path

import pytest

from pilotlab import chain_receipt3 as c, gate3 as g
from pilotlab.catalog3 import model_key
from test_pilot_gate3 import formal_inputs, make_campaign, model_row, write_json
from test_pilot_transfer3 import single_old_selection


@pytest.fixture
def chain(tmp_path):
    root = tmp_path / "new-round"
    model = model_row(0)
    model["model"]["config"]["allocation"] = "quad"
    model["model_key"] = model["effective_key"] = model_key(model["model"])
    identity = {"source": "new-frozen-source", "data": "fixed-data"}
    choice = single_old_selection(root, identity, model)[model["id"]]
    args = {"model": model, "choice": choice, "source_identity": identity,
            "tuning_seeds": [101, 202], "diagnostic_seeds": [101, 202, 303]}
    reference = c.write_model_chain_receipt(tmp_path / "receipt.json", **args)
    choice["chain_receipt"] = reference
    return {"root": root, "args": args, "reference": reference, "path": tmp_path}


def declare_retirement(chain, *, records=None):
    choice, reference = chain["args"]["choice"], chain["reference"]
    records = c.retire_consumed_parents(reference, choice, chain["root"]) if records is None else records
    path = write_json(chain["path"] / "retirement.json", {"schema": c.RETIREMENT_SCHEMA,
        "receipt": reference, "output_root": str(chain["root"]), "checkpoints": records})
    choice["parent_checkpoint_retirement"] = {"path": path, "sha256": g.sha256(path)}
    return records


def verify(chain):
    return c.verify_model_chain_receipt(chain["reference"], **chain["args"])


def test_only_four_consumed_parents_can_retire_after_a_complete_frozen_receipt(chain):
    reference, choice = chain["reference"], chain["args"]["choice"]
    records = c.retire_consumed_parents(reference, choice, chain["root"])
    assert len(records) == 4 and {row["phase"] for row in records} == {"a1", "a2"}
    assert all(Path(row["path"]).is_file() for row in records)  # Planning is read-only.
    declare_retirement(chain, records=records)
    for row in records:
        Path(row["path"]).unlink()
    validated = verify(chain)
    assert validated["retired_checkpoints"] == records
    assert all(Path(row["path"]).is_file() for row in validated["retained_checkpoints"])
    model = chain["args"]["model"]
    manifest = {"source_identity": chain["args"]["source_identity"], "subjects": [model], "models": [model],
                "tuning_seeds": [101, 202], "diagnostic_seeds": [101, 202, 303]}
    report = g.audit_stage_a(manifest, {model["id"]: choice}, [], remaining_resources_ok=True)
    assert model["id"] in report["eligible_models"]
    assert report["retired_parent_checkpoint_count"] == 4
    assert sum(row.get("checkpoint_state") == "retired_after_verified_continuation" for row in report["evidence"]) == 4


def test_missing_parent_without_an_explicit_retirement_manifest_still_fails(chain):
    record = c.retire_consumed_parents(chain["reference"], chain["args"]["choice"], chain["root"])[0]
    Path(record["path"]).unlink()
    with pytest.raises(ValueError, match="missing without"):
        verify(chain)


def test_declared_retirement_never_hides_corruption_of_a_still_present_parent(chain):
    records = declare_retirement(chain)
    Path(records[0]["path"]).write_bytes(b"corrupted-but-still-present")
    with pytest.raises(ValueError, match="corrupt"):
        verify(chain)


def test_a3_payloads_cannot_retire_and_a_new_receipt_cannot_be_created_after_parent_deletion(chain):
    records = declare_retirement(chain)
    for row in records:
        Path(row["path"]).unlink()
    with pytest.raises(ValueError, match="missing without"):
        c.write_model_chain_receipt(chain["path"] / "new-receipt.json", **chain["args"])
    retained = g.load(chain["reference"]["path"])["retained_checkpoints"]
    Path(retained[0]["path"]).unlink()
    with pytest.raises(ValueError, match="missing without"):
        verify(chain)


@pytest.mark.parametrize("damage", ["a3_in_list", "unknown_file", "duplicate", "wrong_hash", "different_receipt", "external_root"])
def test_retirement_manifest_cannot_authorize_other_payloads(chain, damage):
    records = c.retire_consumed_parents(chain["reference"], chain["args"]["choice"], chain["root"])
    if damage == "a3_in_list":
        records.append(g.load(chain["reference"]["path"])["retained_checkpoints"][0])
    elif damage == "unknown_file":
        records[0]["path"] = str(chain["path"] / "unrelated.pt")
    elif damage == "duplicate":
        records.append(records[0])
    elif damage == "wrong_hash":
        records[0]["sha256"] = "f"*64
    declare_retirement(chain, records=records)
    entry = chain["args"]["choice"]["parent_checkpoint_retirement"]
    if damage in {"different_receipt", "external_root"}:
        retirement = g.load(entry["path"])
        if damage == "different_receipt":
            retirement["receipt"]["sha256"] = "f"*64
        else:
            retirement["output_root"] = str(chain["root"] / "another-round")
        write_json(entry["path"], retirement)
        entry["sha256"] = g.sha256(entry["path"])
    with pytest.raises(ValueError):
        verify(chain)


@pytest.mark.parametrize("damage", ["task", "result", "recipe", "source", "receipt", "retirement"])
def test_retired_payload_receipts_do_not_weaken_remaining_evidence_bindings(chain, damage):
    records = declare_retirement(chain)
    for row in records:
        Path(row["path"]).unlink()
    choice = chain["args"]["choice"]
    if damage in {"task", "result"}:
        path = choice["selection_a1"][0][f"{damage}_path"]
        value = g.load(path)
        value["changed_after_retirement"] = True
        write_json(path, value)
    elif damage == "recipe":
        choice["recipe_id"] = "other"
    elif damage == "source":
        chain["args"]["source_identity"] = {"source": "different"}
    elif damage == "receipt":
        path = chain["reference"]["path"]
        value = g.load(path)
        value["choice_sha256"] = "f"*64
        Path(path).unlink()  # Simulate replacement of the write-once artifact.
        write_json(path, value)
    else:
        path = choice["parent_checkpoint_retirement"]["path"]
        value = g.load(path)
        value["checkpoints"].pop()
        write_json(path, value)
    with pytest.raises(ValueError):
        verify(chain)


def test_retirement_cannot_escape_its_round_or_create_a_receipt_for_failed_evidence(chain):
    with pytest.raises(ValueError, match="outside"):
        c.retire_consumed_parents(chain["reference"], chain["args"]["choice"], chain["root"] / "101")
    choice = chain["args"]["choice"]
    path = choice["selection_a2"][0]["result_path"]
    result = g.load(path)
    result["health"] = {"eligible": False, "blocking_reasons": ["failure payload must remain"]}
    write_json(path, result)
    with pytest.raises(ValueError, match="not qualified"):
        c.write_model_chain_receipt(chain["path"] / "invalid.json", **chain["args"])


def test_formal_gate_revalidates_retired_parent_receipts_and_all_surviving_evidence(chain, tmp_path):
    records = declare_retirement(chain)
    for row in records:
        Path(row["path"]).unlink()
    model = chain["args"]["model"]
    campaign = make_campaign(tmp_path / "other-models", identity=chain["args"]["source_identity"],
                             subjects=[model]+[model_row(i) for i in range(1, 216)])
    campaign[1][model["id"]] = chain["args"]["choice"]
    manifest_path, report_path, tasks = formal_inputs(campaign, tmp_path)
    gate = g.create_gate(tmp_path / "formal-gate.json", manifest_path=manifest_path,
                         report_path=report_path, task_paths=tasks)
    assert len(gate["allowed_task_ids"]) == 219
    report = g.load(report_path)
    assert report["retired_parent_checkpoint_count"] == 4
    report["chain_receipts"] = {}
    write_json(report_path, report)
    with pytest.raises(ValueError, match="does not bind"):
        g.create_gate(tmp_path / "unbound-gate.json", manifest_path=manifest_path, report_path=report_path, task_paths=tasks)
