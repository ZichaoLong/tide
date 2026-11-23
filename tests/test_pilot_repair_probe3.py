"""Numerical admission and finite budget safeguards for the repair adapter."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

from pilotlab import repair_probe3 as p


def test_independent_scalar_checks_cover_all_four_budgets_and_original_guard():
    result = p.numerical_checks(torch.device("cpu"), time.time() + 600)
    assert result["status"] == "passed"
    assert len(result["cases"]) == 18
    assert result["processed_loss_tokens"] == 0
    assert all(row["passed"] for row in result["cases"])
    real = next(row for row in result["cases"] if row["id"] == "actual-failure-row")
    assert 5.8e-8 < real["values"][1] < 6e-8
    assert real["score_vjp"][0] > 0 and real["score_vjp"][1] < 0


def test_processed_budget_counts_parent_and_restored_update_exactly():
    assert p.processing_bound({"targets": torch.zeros(2, 128)}) == 1024
    with pytest.raises(ValueError, match="nonempty"):
        p.processing_bound({"targets": torch.empty(0, 128)})
    with pytest.raises(ValueError, match="ceiling"):
        p.processing_bound({"targets": torch.empty(513, 128)})


def test_changed_input_fails_before_any_runtime_is_started(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("initial")
    request = {"failure_result": str(path), "failure_result_sha256": p.sha256(path)}
    path.write_text("changed")
    with pytest.raises(ValueError, match="identity changed"):
        p.verified_input(request)


def test_local_deadline_and_device_budget_cannot_extend_global_budget(monkeypatch):
    monkeypatch.setattr(p.Supervisor, "resource_reason", lambda self: None)
    instance = object.__new__(p.RepairSupervisor)
    instance.config = {"deadline_unix": time.time() - 1, "segment_max_device_seconds": 3600}
    instance.device_seconds = lambda: 0
    assert instance.resource_reason() == "repair_walltime_limit"
    instance.config["deadline_unix"] = time.time() + 60
    instance.device_seconds = lambda: 3600
    assert instance.resource_reason() == "repair_device_hour_limit"
    monkeypatch.setattr(p.Supervisor, "resource_reason", lambda self: "global_deadline")
    assert instance.resource_reason() == "global_deadline"


def test_controller_rejects_changed_absolute_budget_before_claim(tmp_path, monkeypatch):
    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps({"max_device_seconds": 7200 * 3600 + 1,
        "preparation_max_device_seconds": 48 * 3600, "preparation_max_processed_tokens": 64 * 2**20,
        "started_unix": 0, "deadline_unix": 30 * 86400}))
    before = ledger.read_bytes()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"source_identity": {"source_files": {}},
                               "config": {"budget_ledger": str(ledger)}, "campaign_deadline_unix": 30 * 86400}))
    with pytest.raises(ValueError, match="original absolute"):
        p.controller(plan, tmp_path / "round")
    assert ledger.read_bytes() == before
    assert not (tmp_path / "round").exists()


def test_three_updates_save_complete_prefix_and_fresh_owner_repeats_final_update(tmp_path, monkeypatch):
    from pilotlab import models3
    from pilotlab.worker3 import optimizer_for

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 2)

        def forward(self, inputs, state, record_work=False):
            hidden = self.embedding(inputs)
            return SimpleNamespace(logits=hidden @ self.embedding.weight.T,
                                   state={"position": state["position"] + inputs.shape[1]}, work=None)

    monkeypatch.setattr(models3, "build_model", lambda *args: Tiny())
    model = Tiny()
    recipe = {"model": {"family": "fixture"}, "init_recipe": "default", "lr": 1e-3,
              "target_tokens": 16, "schedule_tokens": 16, "warmup_tokens": 0, "grad_clip": 1., "precision": "fp32"}
    failure = {"task": recipe, "model": model.state_dict(),
        "optimizer": optimizer_for(model, recipe).state_dict(), "private_state": {"position": 0},
        "inputs": torch.tensor([[0, 1], [2, 3]]), "targets": torch.tensor([[1, 2], [3, 0]]), "completed_tokens": 0}
    payload = tmp_path / "failure.pt"
    torch.save(failure, payload)
    manifest = tmp_path / "failure.json"
    manifest.write_text(json.dumps({"failure_capture": {"path": str(payload), "sha256": p.sha256(payload)}}))
    task = {"failure_result": str(manifest), "failure_result_sha256": p.sha256(manifest),
            "source_identity": {"source_files": {}}, "target_tokens": 16, "budget_tokens": 16}
    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir(); child.mkdir()
    first = p.updates(task, parent, torch.device("cpu"), time.time() + 600)
    assert first["completed_loss_tokens"] == 12
    assert first["initial_state_load"]["exact"] is True
    checkpoint = first["checkpoint"]
    assert Path(checkpoint["path"]).is_file()
    assert p.sha256(checkpoint["path"]) == checkpoint["sha256"]
    restored = p.updates(task, child, torch.device("cpu"), time.time() + 600, prefix=Path(checkpoint["path"]))
    assert restored["completed_loss_tokens"] == 4
    assert restored["initial_state_load"]["exact"] is True
    assert restored["initial_state_load"]["rng_exact"] is True
    assert first["final_state"]["state_sha256"] == restored["final_state"]["state_sha256"]
