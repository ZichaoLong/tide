"""Actual CPU continuation and failures that must not produce completion."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from pilotlab.data3 import prepare
from pilotlab.io import sha256
from pilotlab.worker import digest_tree
from pilotlab.worker3 import is_capacity_error, learning_rate, validate_gate
from test_pilot_data3 import write_source

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data(tmp_path):
    output = tmp_path / "data"
    prepare(write_source(tmp_path / "source"), output)
    return output


def source_identity(data):
    package = ROOT / "pilotlab"
    files = [package / name for name in ("worker3.py", "data3.py", "models3.py", "models.py", "worker.py", "io.py")]
    foundation = Path(next(iter(importlib.util.find_spec("tide").submodule_search_locations)))
    files.extend(foundation.rglob("*.py"))
    return {"schema": "pilot-r3-source-v1", "data_manifest_sha256": sha256(data / "manifest.json"),
            "source_files": {str(path.resolve()): sha256(path) for path in files}}


def task_spec(data, **changes):
    task = {"id": "parent", "model_id": "tiny-transformer", "phase": "a1", "seed": 84,
            "model": {"family": "transformer", "d": 16, "layers": 1, "heads": 2, "ffn_width": 32, "window": 32},
            "source_identity": source_identity(data), "init_recipe": "default", "cpu_threads": 2,
            "batch_size": 2, "sequence_length": 8, "stream_chunks": 5, "precision": "fp32",
            "target_tokens": 32, "schedule_tokens": 512, "warmup_tokens": 64, "lr": 1e-3,
            "weight_decay": 0.01, "grad_clip": 1.0, "eval_tokens": 16, "health_interval_tokens": 32,
            "eval_interval_tokens": 64, "closing_reserve_seconds": 2, "checkpoint_interval_steps": 1,
            "record_step_fingerprints": True, "capture_failure": False}
    task.update(changes)
    return task


def run_worker(tmp_path, data, task, *, interrupt=False, deadline=None, resume_override=None):
    task_file, output = tmp_path / f"{task['id']}.json", tmp_path / task["id"]
    task_file.write_text(json.dumps(task))
    command = [sys.executable, "-m", "pilotlab.worker3"]
    if interrupt:
        harness = """
import os,signal
from pilotlab import worker3
original=worker3.optimizer_for
def factory(model,task):
    optimizer=original(model,task)
    step=optimizer.step
    count=0
    def interrupted(*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:
            os.kill(os.getpid(),signal.SIGTERM)
        return step(*args,**kwargs)
    optimizer.step=interrupted
    return optimizer
worker3.optimizer_for=factory
raise SystemExit(worker3.main())
"""
        command = [sys.executable, "-c", harness]
    command += ["--task", str(task_file), "--output", str(output), "--data", str(data),
                "--device", "cpu", "--deadline-unix", str(deadline or time.time() + 300)]
    if resume_override is not None:
        command += ["--resume-checkpoint", str(resume_override)]
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    result = json.loads((output / "result.json").read_text())
    expected_code = 0 if result["status"] == "completed" else 2 if result["status"] == "budget_exhausted" else 1
    assert process.returncode == expected_code, (result, process.stdout, process.stderr)
    payload = torch.load(result["checkpoint"]["path"], map_location="cpu", weights_only=False) if result.get("checkpoint") else None
    return result, payload, output


@pytest.mark.parametrize("interrupted", [False, True])
def test_fresh_process_resume_matches_uninterrupted_updates(tmp_path, data, interrupted):
    parent_task = task_spec(data, target_tokens=1600 if interrupted else 32)
    parent, saved, parent_path = run_worker(tmp_path, data, parent_task, interrupt=interrupted)
    assert parent["status"] == ("budget_exhausted" if interrupted else "completed"), parent
    assert saved["tokens"] == 32 and saved["steps"] == 2
    resumed_task = dict(parent_task, id="resumed", phase="a2", target_tokens=128,
                        resume_checkpoint=str(tmp_path / "superseded-checkpoint.pt"),
                        resume_checkpoint_sha256="0" * 64)
    resumed, actual, resumed_path = run_worker(tmp_path, data, resumed_task, resume_override=parent["checkpoint"]["path"])
    baseline, expected, baseline_path = run_worker(tmp_path, data, dict(parent_task, id="baseline", target_tokens=128))
    assert resumed["status"] == baseline["status"] == "completed", (resumed, baseline)
    assert resumed["training"]["tokens"] == 128 and resumed["training"]["segment_tokens"] == 96
    assert resumed["resume"]["state_reused"] and resumed["resume"]["runtime_override"]
    assert resumed["initialization"] == parent["initialization"] == baseline["initialization"]
    for key in ("model", "optimizer", "private_state", "stream", "rng"):
        assert digest_tree(actual[key]) == digest_tree(expected[key]), key
    resumed_rows = [json.loads(line) for line in (resumed_path / "steps.jsonl").read_text().splitlines()]
    baseline_rows = [json.loads(line) for line in (baseline_path / "steps.jsonl").read_text().splitlines()][2:]
    for left, right in zip(resumed_rows, baseline_rows, strict=True):
        for key in ("step", "tokens", "loss", "lr", "input_sha256", "rng_before_sha256", "rng_after_sha256", "state_reset"):
            assert left[key] == right[key], (key, left, right)
    assert resumed["recovery"]["checkpoint_load"]["status"] == "exact"
    assert resumed["recovery"]["numeric_trajectory"]["status"] == "not_tested"
    assert len(list((resumed_path / "checkpoints").glob("*.pt"))) == 2


def test_deadline_never_marks_short_exposure_complete(tmp_path, data):
    task = task_spec(data, target_tokens=1024)
    result, payload, _ = run_worker(tmp_path, data, task, deadline=time.time() - 1)
    assert result["status"] == "budget_exhausted"
    assert result["training"]["tokens"] == payload["tokens"] == 0
    assert not result["health"]["eligible"]
    assert "declared_training_exposure_not_reached" in result["health"]["blocking_reasons"]


def test_two_generations_recover_from_corrupt_latest_and_reject_changed_recipe(tmp_path, data):
    task = task_spec(data, target_tokens=48)
    result, _, output = run_worker(tmp_path, data, task)
    assert result["status"] == "completed", result
    index = json.loads((output / "checkpoint.json").read_text())
    Path(index["current"]["path"]).write_bytes(b"incomplete checkpoint write")
    resumed, actual, _ = run_worker(tmp_path, data, dict(task, id="fallback", phase="a2", target_tokens=80,
                                                       resume_checkpoint=str(output / "checkpoint.json")))
    assert resumed["status"] == "completed" and resumed["resume"]["fallback_used"]
    assert actual["tokens"] == 80
    rejected, _, _ = run_worker(tmp_path, data, dict(task, id="changed-recipe", target_tokens=80, lr=2e-3,
                                                   resume_checkpoint=str(output / "checkpoint.json")))
    assert rejected["status"] == "failed" and "trajectory contract mismatch" in rejected["error"]


def test_source_tampering_and_missing_gate_fail_before_training(tmp_path, data):
    task = task_spec(data)
    changed = copy.deepcopy(task)
    changed["id"] = "bad-source"
    changed["source_identity"]["source_files"][str(ROOT / "pilotlab" / "worker3.py")] = "0" * 64
    result, _, _ = run_worker(tmp_path, data, changed)
    assert result["status"] == "failed" and "source identity mismatch" in result["error"]
    assert result["training"]["tokens"] == 0 and result["checkpoint"] is None
    result, _, _ = run_worker(tmp_path, data, dict(task, id="blocked-formal", phase="b"))
    assert result["status"] == "failed" and "requires a passed" in result["error"]
    assert result["training"]["tokens"] == 0 and result["checkpoint"] is None


def test_gate_binds_task_source_report_and_manifest(tmp_path):
    manifest, report, gate_file = tmp_path / "manifest.json", tmp_path / "report.json", tmp_path / "gate.json"
    manifest.write_text('{"tasks":["formal"]}')
    report.write_text('{"status":"passed"}')
    task = {"id": "formal", "phase": "b", "source_identity": {"source": "fixed"}, "gate_file": str(gate_file)}
    gate = {"schema": "pilot-stage-b-gate-v1", "status": "passed", "source_identity": task["source_identity"],
            "allowed_task_ids": ["formal"], "task_sha256": {"formal": "abc"},
            "manifest_path": str(manifest), "manifest_sha256": sha256(manifest),
            "report_path": str(report), "report_sha256": sha256(report)}
    gate_file.write_text(json.dumps(gate))
    assert validate_gate(task, "abc")["status"] == "passed"
    with pytest.raises(ValueError, match="task file hash mismatch"):
        validate_gate(task, "different")
    with pytest.raises(ValueError, match="not in the frozen"):
        validate_gate(dict(task, id="unlisted"), "abc")
    with pytest.raises(ValueError, match="source identity mismatch"):
        validate_gate(dict(task, source_identity={}), "abc")
    report.write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="report hash mismatch"):
        validate_gate(task, "abc")


def test_token_schedule_is_a_single_long_schedule_not_rung_restarts():
    task = {"lr": 1e-3, "warmup_tokens": 1024, "schedule_tokens": 12 * 2**20}
    assert learning_rate(task, 512) == 0.0005
    assert learning_rate(task, 1024) == 0.001
    assert learning_rate(task, 2**20) < learning_rate(task, 2**18)
    assert learning_rate(task, 12 * 2**20) == pytest.approx(0.0001)


def test_capacity_classification_does_not_hide_runtime_or_numeric_errors():
    assert is_capacity_error(torch.OutOfMemoryError("allocation exhausted"), "cpu")
    assert is_capacity_error(RuntimeError("NPU out of memory. Tried to allocate 2 GiB"), "npu:0")
    assert not is_capacity_error(RuntimeError("NPU operator allocation failed"), "npu:0")
    assert not is_capacity_error(RuntimeError("NPU out of memory diagnostic string"), "cpu")
    assert not is_capacity_error(ArithmeticError("nonfinite loss"), "npu:0")


def test_memory_observation_records_device_free_instead_of_subtracting_local_allocations(monkeypatch):
    from types import SimpleNamespace
    from pilotlab import worker3

    observed_devices = []
    def mem_get_info(device):
        observed_devices.append(device)
        return 333, 1000

    monkeypatch.setattr(worker3, "base_device_memory", lambda device: {"npu_reserved_bytes": 100})
    monkeypatch.setattr(torch, "npu", SimpleNamespace(mem_get_info=mem_get_info), raising=False)
    device = SimpleNamespace(type="npu")
    result = worker3.device_memory(device)
    assert observed_devices == [device]
    assert result["npu_total_memory_bytes"] == 1000 and result["npu_free_memory_bytes"] == 333
    assert result["npu_reserved_bytes"] == 100
