"""No devices: adoption provenance and one-time evaluation budget accounting."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pilotlab import resume3
from pilotlab.io import sha256
from pilotlab.scheduler3 import Supervisor, Blocked


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": sha256(path)}


def fixture(tmp_path, disposition="resume", extra=32):
    task = {"id": "a1", "model_id": "m", "phase": "a1", "source_identity": {"source": "original"},
        "target_tokens": 262144, "eval_tokens": 16384, "evaluation_budget_key": "m", "evaluation_token_limit": 1048576}
    original = write(tmp_path/"old/task.json", task)
    result = write(tmp_path/"old/attempt-0/result.json", {"status": "completed" if disposition == "completed" else "budget_exhausted",
        "id": "a1", "model_id": "m", "source_identity": task["source_identity"], "task_sha256": original["sha256"],
        "training": {"tokens": task["target_tokens"]}})
    row = {"task": original, "result": result, "disposition": disposition, "worker_return_code": 0,
        "checkpoint": {"path": "old.pt", "sha256": "full-state"},
        "evaluation_reserved_tokens": 32800, "additional_evaluation_reserved_tokens": extra}
    row["resume_receipt"] = write(tmp_path/"receipt.json", {"original_task": original, "result": result,
        "checkpoint": row["checkpoint"], "evaluation_additional_reservation_tokens": extra})
    review = write(tmp_path/"review.json", {"status": "supplemental_review_completed"})
    config = {"resume_import": write(tmp_path/"import.json", {"schema": "pilot-r3-resume-import-v1",
        "reconciliation": review, "worker_source_identity": task["source_identity"], "tasks": {"a1": row}})}
    record = {"status": "pending", "imported_from": "old", "attempts": [], "evaluation_reserved_tokens": 32800,
        "evaluation_budget_key": "m", "result_path": result["path"], "result_sha256": result["sha256"]}
    return config, task, record, row


def test_resume_adds_only_audited_increment_without_refunding_prior_reservation(tmp_path):
    config, task, record, _ = fixture(tmp_path)
    owner = Supervisor.__new__(Supervisor)
    owner.config, owner.state = config, {"tasks": {"a1": record}}
    owner.reserve_task(task, record)
    assert record["evaluation_reserved_tokens"] == 32832
    # Reservation is one-shot: changing the original baseline cannot receive
    # the same credit twice, even before a process is launched.
    with pytest.raises(Blocked, match="reservation_contract_changed"):
        owner.reserve_task(task, record)


def test_imported_success_reuses_original_result_but_changed_bytes_block(tmp_path):
    config, task, record, row = fixture(tmp_path, disposition="completed", extra=0)
    assert resume3.verify_imported_completion(config, task, record)["status"] == "completed"
    Path(row["result"]["path"]).write_text('{"status":"completed"}')
    with pytest.raises(Blocked, match="reference_changed"):
        resume3.verify_imported_completion(config, task, record)


def test_resume_cannot_bypass_the_model_evaluation_limit(tmp_path):
    config, task, record, _ = fixture(tmp_path, extra=6432)
    config["prior_evaluation_reservations"] = {"m": 1048576-32800}
    owner = Supervisor.__new__(Supervisor)
    owner.config, owner.state = config, {"tasks": {"a1": record}}
    before = deepcopy(record)
    with pytest.raises(Blocked, match="evaluation_token_limit"):
        owner.reserve_task(task, record)
    assert record == before


def test_changed_reconciliation_cannot_authorize_adoption(tmp_path):
    config, task, record, _ = fixture(tmp_path, disposition="completed", extra=0)
    (tmp_path/"review.json").write_text('{}')
    with pytest.raises(Blocked, match="reference_changed"):
        resume3.verify_imported_completion(config, task, record)
