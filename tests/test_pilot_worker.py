"""Pilot stream and checkpoint checks using small, document-bounded fixtures."""
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

from pilotlab.data import DocumentStore, SCHEMA
from pilotlab.worker import StreamBatch, compare_tree


@pytest.fixture
def prepared_data(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    splits = {}
    for split, shift in (("train", 0), ("validation", 7)):
        rows = [((np.arange(128, dtype=np.uint16) + shift + index * 3) % 29 + 1) for index in range(4)]
        np.concatenate(rows).astype("<u2").tofile(root / f"{split}.tokens.bin")
        ranges = np.array([[index * 128, (index + 1) * 128] for index in range(4)], dtype=np.int64)
        np.save(root / f"{split}.documents.npy", ranges, allow_pickle=False)
        splits[split] = {"tokens": f"{split}.tokens.bin", "documents": f"{split}.documents.npy"}
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA, "vocab_size": 32, "token_dtype": "<u2", "splits": splits,
        "dataset_id": "worker-test-fixture", "split_policy": "disjoint synthetic fixture documents",
    }))
    return root


def stream_task():
    return dict(seed=47, batch_size=2, sequence_length=5, stream_chunks=3)


def test_stream_continuity_and_block_reset(prepared_data):
    stream = StreamBatch(DocumentStore(prepared_data), stream_task())
    resets, previous_y = [], None
    for index in range(7):
        x, y, reset = stream.next()
        resets.append(reset)
        assert x.shape == y.shape == (2, 5)
        torch.testing.assert_close(x[:, 1:], y[:, :-1], rtol=0, atol=0)
        if not reset:
            torch.testing.assert_close(x[:, 0], previous_y[:, -1], rtol=0, atol=0)
        previous_y = y
        assert stream.cursor == (index % 3 + 1) * 5
    assert resets == [True, False, False, True, False, False, True]


def test_stream_resume_replays_partial_block_and_future_sampling(prepared_data):
    store, task = DocumentStore(prepared_data), stream_task()
    original = StreamBatch(store, task)
    original.next()
    original.next()
    restored = StreamBatch(store, task, copy.deepcopy(original.state_dict()))
    for _ in range(8):
        a_x, a_y, a_reset = original.next()
        b_x, b_y, b_reset = restored.next()
        assert a_reset == b_reset
        torch.testing.assert_close(a_x, b_x, rtol=0, atol=0)
        torch.testing.assert_close(a_y, b_y, rtol=0, atol=0)


def test_finite_tolerance_does_not_require_exact_float_bits():
    result = compare_tree({"weight": torch.tensor([1.0])}, {"weight": torch.tensor([1.00001])},
                          atol=2e-5, rtol=0)
    assert result["finite"] and result["structure_equal"] and result["integer_equal"]
    assert result["within_tolerance"] and not result["exact"]
    assert result["max_abs"] > 0


@pytest.mark.parametrize("expected,actual", [
    ({"clock": torch.tensor([4])}, {"clock": torch.tensor([5])}),
    ({"position": 4}, {"position": 5}),
    ({"clock": torch.tensor([4], dtype=torch.int32)}, {"clock": torch.tensor([4], dtype=torch.int64)}),
    ({"clock": torch.tensor([4])}, {"clock": torch.tensor([[4]])}),
])
def test_discrete_changes_never_receive_floating_tolerance(expected, actual):
    result = compare_tree(expected, actual, atol=100, rtol=100)
    assert not result["integer_equal"]
    assert not result["exact"] and not result["within_tolerance"]


def test_nonfinite_and_disconnected_gradients_remain_failures():
    result = compare_tree(torch.tensor([float("inf")]), torch.tensor([float("inf")]))
    assert not result["finite"] and not result["within_tolerance"]
    result = compare_tree({"grad": None}, {"grad": torch.zeros(1)})
    assert not result["structure_equal"] and not result["within_tolerance"]


@pytest.mark.parametrize("model_spec", [
    {"family": "transformer", "d": 16, "layers": 1, "heads": 2, "ffn_width": 32, "window": 32},
    {"family": "tide", "config": {"topology": {"kind": "fc", "n": 1, "layers": 2},
                                    "d": 64, "memory": "S", "budget": 1}},
], ids=["transformer", "ssd_private_cache"])
def test_cpu_worker_replays_a_real_update_in_a_fresh_process(prepared_data, tmp_path, model_spec):
    task = {
        "id": "worker-test-transformer", "seed": 912, "cpu_threads": 2,
        "model": model_spec,
        "batch_size": 2, "sequence_length": 8, "stream_chunks": 3,
        "lr": 1e-3, "warmup_steps": 1, "weight_decay": 0.01,
        "train_seconds": 10, "max_steps": 2, "measurement_warmup_steps": 0,
        "eval_batches": 1, "decode_steps": 1, "precision": "fp32", "resume_check": True,
        "resume_verify_steps": 8,
    }
    task_path, output = tmp_path / "task.json", tmp_path / "output"
    task_path.write_text(json.dumps(task))
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
    command = [sys.executable, "-m", "pilotlab.worker", "--task", str(task_path),
               "--data", str(prepared_data), "--output", str(output), "--device", "cpu",
               "--deadline-unix", str(time.time() + 600)]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=45)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "completed", result
    assert result["training"]["steps"] == 2
    assert result["training"]["tokens"] == 32
    assert result["training"]["tokens_per_second"] > 0
    assert result["causality"]["within_tolerance"]
    recovery = result["recovery"]
    assert recovery["status"] == "within_tolerance", recovery
    assert recovery["saved_data_exact"] and recovery["next_input_exact"]
    assert recovery["rng_after_exact"] and recovery["continued_private_state"]
    assert set(recovery["comparisons"]) >= {"gradients", "model", "optimizer", "private_state", "logits"}
    assert all(value["within_tolerance"] for value in recovery["comparisons"].values())
    assert recovery["from_step"] == 2 and recovery["to_step"] == 10
    assert recovery["verified_steps"] == 8
    assert len(recovery["per_step"]) == 8
    assert all(row["next_input_exact"] and row["rng_before_exact"] and row["rng_after_exact"]
               and row["stream_after_exact"] and row["step_exact"] and row["state_reset_exact"]
               and row["within_tolerance"] for row in recovery["per_step"])
    assert any(not row["continued_private_state"] for row in recovery["per_step"])
    worker = json.loads((output / "worker.json").read_text())
    assert recovery["fresh_pid_result"]["pid"] != worker["pid"]
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["private_state"].position == 16
    replay_stream = StreamBatch(DocumentStore(prepared_data), task, checkpoint["stream"])
    _, _, reset = replay_stream.next()
    assert not reset, "the replay must actually consume the saved private history"
