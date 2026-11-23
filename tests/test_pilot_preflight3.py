"""Preparation plans remain finite and cannot reset budgets or hide failures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from pilotlab.io import sha256
from pilotlab.data3 import prepare
from pilotlab.prepare3 import ensure_budget, freeze
from pilotlab.preflight3 import (PREPARATION_TOKEN_LIMIT, build_plan, capacity_admission,
                                check_result, claim_token_budget, task_processing_bound)
from test_pilot_data3 import write_source


def test_budget_is_created_once_and_repair_does_not_reset_deadline(tmp_path):
    path = tmp_path / "budget.json"
    first = ensure_budget(path, now=100.0)
    second = ensure_budget(path, now=99999.0)
    assert first == second
    assert first["deadline_unix"] == 100 + 30 * 86400
    assert first["max_device_seconds"] == 7200 * 3600
    first["max_device_seconds"] += 1
    path.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="approved immutable ceiling"):
        ensure_budget(path)


def test_snapshot_is_separate_immutable_and_does_not_start_budget_or_jobs(tmp_path):
    repo, foundation = tmp_path / "repo", tmp_path / "foundation"
    (repo / "pilotlab").mkdir(parents=True)
    (foundation / "src/tide").mkdir(parents=True)
    files = ("__init__.py", "prepare3.py", "preflight3.py", "scheduler3.py", "worker3.py", "data3.py",
             "models3.py", "catalog3.py", "models.py", "worker.py", "io.py", "scheduler.py", "selection.py")
    for name in files:
        (repo / "pilotlab" / name).write_text(f'"""fixture {name}"""\n')
    (foundation / "src/tide/__init__.py").write_text('"""fixture foundation"""\n')
    data = tmp_path / "data"
    prepare(write_source(tmp_path / "source-data"), data)
    output, budget = tmp_path / "preparation-r01", tmp_path / "budget.json"
    draft = freeze(repo, foundation, output, python=sys.executable, data=data, budget=budget)
    assert draft["status"] == "not_submitted" and not budget.exists()
    prepared = json.loads((output / "prepared.json").read_text())
    for path, digest in prepared["source_identity"]["source_files"].items():
        assert Path(path).is_relative_to(output)
        assert sha256(path) == digest
    assert prepared["environment"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert "--slice=background.slice" in draft["launch_argv"]
    assert "--property=KillMode=control-group" in draft["launch_argv"]
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD=0" in (output / "launch.sh").read_text()
    (repo / "pilotlab/worker3.py").write_text("# changed working source\n")
    assert (output / "source/pilotlab/worker3.py").read_text() != (repo / "pilotlab/worker3.py").read_text()
    with pytest.raises(FileExistsError, match="existing preparation"):
        freeze(repo, foundation, output, python=sys.executable, data=data, budget=budget)


def test_token_reservations_are_cumulative_and_immutable(tmp_path):
    ledger = tmp_path / "budget.json"
    ensure_budget(ledger, now=1)
    claim_token_budget(ledger, "first", 100, "sha-a")
    claim_token_budget(ledger, "first", 100, "sha-a")
    with pytest.raises(ValueError, match="reservation changed"):
        claim_token_budget(ledger, "first", 101, "sha-a")
    with pytest.raises(RuntimeError, match="preparation_processed_token_limit"):
        claim_token_budget(ledger, "second", PREPARATION_TOKEN_LIMIT, "sha-b")
    value = json.loads(ledger.read_text())
    assert set(value["preparation_token_reservations"]) == {"first"}


def test_token_reservations_include_uncovered_segments_without_double_charging(tmp_path):
    ledger = tmp_path / "budget.json"
    ensure_budget(ledger, now=1)
    claim_token_budget(ledger, "macro-covered", 100, "sha-a")
    value = json.loads(ledger.read_text())
    value["segments"] = {
        "macro-covered": {"preparation_reserved_tokens": 90},
        "previous-direct-run": {"preparation_reserved_tokens": PREPARATION_TOKEN_LIMIT - 200},
    }
    ledger.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="preparation_processed_token_limit"):
        claim_token_budget(ledger, "repair", 101, "sha-b")
    # The segment already covered by its macro must not be charged twice.
    claim_token_budget(ledger, "repair", 100, "sha-b")
    assert set(json.loads(ledger.read_text())["preparation_token_reservations"]) == {"macro-covered", "repair"}


def test_plan_covers_mechanisms_and_safe_start_shapes(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    plan = build_plan({"source_identity": {"schema": "test-only"}})
    assert len(plan["core_tasks"]) == 12
    assert len(plan["shape_tasks"]) == 13
    assert len(plan["resume_templates"]) == 3
    assert plan["task_count_upper"] == 28
    assert plan["processed_token_reservation"] <= PREPARATION_TOKEN_LIMIT
    all_tasks = plan["core_tasks"] + plan["shape_tasks"] + plan["resume_templates"]
    assert len({task["id"] for task in all_tasks}) == len(all_tasks)
    assert all(task["batch_size"] == 1 and task["target_tokens"] == 768 for task in plan["core_tasks"])
    assert all(task["precision"] == "fp32" and task["lr"] > 0 and task["eval_tokens"] == 512 for task in all_tasks)
    assert {task["representative_role"] for task in plan["resume_templates"]} == {"exchange_x", "transformer", "moe"}
    assert all(task["target_tokens"] == 1280 and task["segment_target_tokens"] == 512 for task in plan["resume_templates"])
    assert plan["processed_token_reservation"] == 2 * sum(task_processing_bound(task) for task in all_tasks)


def test_capacity_expansion_needs_real_memory_and_headroom():
    task = {"batch_size": 16, "sequence_length": 256}
    assert not capacity_admission(task, {})["admitted"]
    parent = {"device_memory": {"npu_peak_allocated_bytes": 100, "npu_total_memory_bytes": 1000,
                               "npu_free_memory_bytes": 900, "npu_reserved_bytes": 100}}
    assert not capacity_admission(task, parent)["admitted"]
    parent["device_memory"]["npu_total_memory_bytes"] = 10000
    # Larger total memory alone cannot override observed device pressure.
    assert not capacity_admission(task, parent)["admitted"]
    parent["device_memory"]["npu_free_memory_bytes"] = 9000
    assert capacity_admission(task, parent)["admitted"]
    del parent["device_memory"]["npu_free_memory_bytes"]
    assert capacity_admission(task, parent)["reason"] == "missing_device_memory_measurement"


def test_admission_rejects_self_reported_success_without_scientific_evidence(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint fixture")
    task = {"id": "probe", "target_tokens": 768, "source_identity": {"id": "fixed"}}
    result = {"id": "probe", "status": "completed", "source_identity": task["source_identity"],
        "training": {"tokens": 768}, "health": {"eligible": True, "blocking_reasons": []},
        "causality": {"within_tolerance": True}, "recovery": {"checkpoint_load": {"status": "exact"}},
        "work": [{"work": {"activity": {"qualified": True}}}],
        "device_memory": {"npu_peak_allocated_bytes": 100, "npu_total_memory_bytes": 1000,
                          "npu_free_memory_bytes": 900, "npu_reserved_bytes": 100},
        "checkpoint": {"path": str(path), "sha256": sha256(path), "state_sha256": "state"}}
    assert not check_result(task, result)
    for field in ("health", "causality", "work", "device_memory", "recovery"):
        changed = copy.deepcopy(result)
        del changed[field]
        assert check_result(task, changed), field
    parent = copy.deepcopy(result)
    resumed = copy.deepcopy(result)
    resumed["training"]["from_tokens"] = 768
    resumed["resume"] = {"path": str(path), "sha256": sha256(path), "from_tokens": 768,
        "source_task_id": "probe", "state_reused": True, "checkpoint_load": {"status": "exact"},
        "state_sha256": "state"}
    assert not check_result(task, resumed, parent=parent)
    resumed["resume"]["state_sha256"] = "wrong"
    assert "fresh_worker_continuation_chain_not_exact" in check_result(task, resumed, parent=parent)
    path.write_bytes(b"changed checkpoint")
    assert "checkpoint_missing_or_changed" in check_result(task, result)
