"""Real CPU continuation, update-boundary interruption and identity rejection."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from pilotlab.data import SCHEMA
from pilotlab.io import sha256
from pilotlab.worker import compare_tree, digest_tree, metadata_checkpoint_identity, validate_checkpoint


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    splits = {}
    for name, shift in (("train", 0), ("validation", 17)):
        ((np.arange(2048, dtype=np.uint16) * 7 + shift) % 31).astype("<u2").tofile(root / f"{name}.bin")
        np.save(root / f"{name}.npy", np.array([[0, 512], [512, 1024], [1024, 1536], [1536, 2048]],
                                            dtype=np.int64), allow_pickle=False)
        splits[name] = {"tokens": f"{name}.bin", "documents": f"{name}.npy"}
    (root / "manifest.json").write_text(json.dumps({"schema_version": SCHEMA, "vocab_size": 32,
        "token_dtype": "<u2", "splits": splits, "dataset_id": "resume-test-fixture"}))
    return root


def task_spec(**changes):
    task = {"id": "continuation-test", "seed": 84, "cpu_threads": 2, "backend": "cpu",
            "model": {"family": "transformer", "d": 16, "layers": 1, "heads": 2,
                      "ffn_width": 32, "window": 32},
            "batch_size": 2, "sequence_length": 8, "stream_chunks": 5,
            "lr": 1e-3, "warmup_steps": 7, "weight_decay": 0.01,
            "train_seconds": 60, "wall_seconds": 600, "max_steps": 2,
            "measurement_warmup_steps": 0, "checkpoint_interval_steps": 1,
            "eval_batches": 1, "decode_steps": 1, "precision": "fp32", "resume_check": False,
            "resume_verify_steps": 8}
    task.update(changes)
    return task


def run_worker(tmp_path, data, task, *, interrupt=False):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / task["id"]
    task_file = tmp_path / (task["id"] + ".json")
    task_file.write_text(json.dumps(task))
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
    command = [sys.executable, "-m", "pilotlab.worker"]
    if interrupt:
        # The actual OS signal arrives after backward, before optimizer.step
        # returns. The worker must finish that update before stopping, without
        # taking a checkpoint of half-updated parameters or advancing again.
        harness = """
import os, signal
from pilotlab import worker
original_factory = worker.optimizer_for
def interrupting_factory(model, task):
    optimizer = original_factory(model, task)
    original_step = optimizer.step
    calls = 0
    def step(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return original_step(*args, **kwargs)
    optimizer.step = step
    return optimizer
worker.optimizer_for = interrupting_factory
raise SystemExit(worker.main())
"""
        command = [sys.executable, "-c", harness]
    command += ["--task", str(task_file), "--data", str(data), "--output", str(output),
                "--device", "cpu", "--deadline-unix", str(time.time() + 600)]
    process = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=60)
    result = json.loads((output / "result.json").read_text())
    assert process.returncode == 0, (result, process.stdout, process.stderr)
    return result, torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("interrupted", [False, True], ids=["ordinary", "sigterm-during-update"])
def test_split_training_matches_ten_uninterrupted_real_updates(tmp_path, data, interrupted):
    parent_task = task_spec(id="parent", max_steps=100 if interrupted else 2)
    parent, saved = run_worker(tmp_path, data, parent_task, interrupt=interrupted)
    assert parent["status"] == ("budget_exhausted" if interrupted else "completed")
    assert parent["training"]["steps"] == saved["step"] == 2
    assert saved["history"]["private_state_nonempty"]
    assert saved["history"]["next_input_reuses_private_state"]
    if interrupted:
        assert parent["training"]["stop_reason"] == "signal"
    resumed_task = dict(parent_task, id="resumed", max_steps=8,
                        resume_from=parent["checkpoint"]["path"],
                        resume_checkpoint_sha256=parent["checkpoint"]["sha256"],
                        resume_allow_budget_overrides=["max_steps"])
    resumed, actual = run_worker(tmp_path, data, resumed_task)
    baseline, expected = run_worker(tmp_path, data, dict(parent_task, id="uninterrupted", max_steps=10))
    assert resumed["status"] == baseline["status"] == "completed"
    assert resumed["training"]["from_step"] == 2 and resumed["training"]["to_step"] == 10
    assert resumed["training"]["global_steps"] == 10
    assert resumed["training"]["steps"] == 8 and actual["step"] == expected["step"] == 10
    assert resumed["continuation"]["first_input_reused_private_history"]
    for key in ("model", "optimizer", "private_state"):
        comparison = compare_tree(expected[key], actual[key])
        assert comparison["exact"], (key, comparison)
    assert digest_tree(expected["stream"]) == digest_tree(actual["stream"])
    assert digest_tree(expected["rng"]) == digest_tree(actual["rng"])
    resumed_rows = [json.loads(row) for row in (tmp_path / "resumed" / "steps.jsonl").read_text().splitlines()]
    baseline_rows = [json.loads(row) for row in (tmp_path / "uninterrupted" / "steps.jsonl").read_text().splitlines()][2:]
    for resumed_row, baseline_row in zip(resumed_rows, baseline_rows, strict=True):
        for key in ("step", "loss", "grad_norm", "lr", "input_sha256", "rng_before_sha256",
                    "rng_after_sha256", "state_reset"):
            assert resumed_row[key] == baseline_row[key], (key, resumed_row, baseline_row)
    assert sha256(Path(parent["checkpoint"]["path"])) == parent["checkpoint"]["sha256"]


def test_checkpoint_rejects_changed_training_data_source_and_unlisted_budgets(tmp_path, data):
    task = task_spec()
    _, payload = run_worker(tmp_path, data, task)
    observed = metadata_checkpoint_identity(data)
    assert validate_checkpoint(payload, task, data, "cpu", observed)["contract_exact"]
    changes = [{"seed": 85}, {"batch_size": 1}, {"sequence_length": 16}, {"stream_chunks": 4},
               {"lr": 2e-3}, {"weight_decay": 0.1}, {"warmup_steps": 3}, {"grad_clip": 2.0},
               {"cpu_threads": 1}, {"precision": "bf16"}, {"eval_batches": 2}, {"decode_steps": 2},
               {"model": dict(task["model"], window=64)}]
    for change in changes:
        with pytest.raises(ValueError, match="continuation contract mismatch"):
            validate_checkpoint(payload, dict(task, **change), data, "cpu", observed)
    with pytest.raises(ValueError, match="contract mismatch: backend"):
        validate_checkpoint(payload, task, data, "npu:0", observed)
    with pytest.raises(ValueError, match="budget override not declared"):
        validate_checkpoint(payload, dict(task, max_steps=8), data, "cpu", observed)
    assert validate_checkpoint(payload, dict(task, max_steps=8, resume_allow_budget_overrides=["max_steps"]),
                               data, "cpu", observed)["budget_overrides"] == ["max_steps"]
    with pytest.raises(ValueError, match="supported runtime budget fields"):
        validate_checkpoint(payload, dict(task, resume_allow_budget_overrides=["lr"]), data, "cpu", observed)
    for key in ("experiment_sha256", "foundation_sha256", "data_manifest_sha256", "runtime_packages"):
        changed = copy.deepcopy(observed)
        changed[key] = "different"
        with pytest.raises(ValueError, match="identity mismatch"):
            validate_checkpoint(payload, task, data, "cpu", changed)
    corrupted = copy.deepcopy(payload)
    next(iter(corrupted["model"].values())).flatten()[0] += 1
    with pytest.raises(ValueError, match="state digest mismatch"):
        validate_checkpoint(corrupted, task, data, "cpu", observed)
    with pytest.raises(ValueError, match="checkpoint_identity differs"):
        validate_checkpoint(payload, dict(task, checkpoint_identity={"label": "trusted"}), data, "cpu", observed)


def test_identity_entrypoint_does_not_import_torch_or_vendor(tmp_path, data):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    program = """
import json, sys
from pilotlab.worker import metadata_checkpoint_identity
identity = metadata_checkpoint_identity(sys.argv[1])
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert identity['experiment_files']['worker.py']
assert identity['foundation_files']
print(json.dumps(identity))
"""
    process = subprocess.run([sys.executable, "-c", program, str(data)], cwd=root, env=env,
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    identity = json.loads(process.stdout)
    assert identity["data_manifest_sha256"] == sha256(data / "manifest.json")
    assert identity["runtime_packages"]["tide"] is not None
    assert identity["foundation_distributions"]
