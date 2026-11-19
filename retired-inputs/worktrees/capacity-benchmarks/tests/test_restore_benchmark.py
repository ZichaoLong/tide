"""Fresh-process restoration timing uses real saved slots and advances state."""
import hashlib
import json
import subprocess
import sys

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig, save_checkpoint
from tide.precision import precision_context
from benchmarks.restore import checkpoint_identity, checkpoint_stat
from benchmarks.workloads import paired_input


def test_checkpoint_hash_is_streamed_and_rejects_changed_input(tmp_path):
    path = tmp_path/"checkpoint"
    payload = b"abc"*(3*2**20)
    path.write_bytes(payload)
    initial = checkpoint_stat(path)
    identity = checkpoint_identity(path, initial)
    assert identity["sha256"] == hashlib.sha256(payload).hexdigest()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        checkpoint_identity(path, initial)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_restored_first_update_loads_slots_and_records_separate_complete_time(tmp_path, precision):
    torch.manual_seed(67)
    model = TideModel(ModelConfig(topology=TopologyConfig(n=2, layers=2), d=64, sharing="region_shared"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, foreach=True)
    with precision_context("cpu", precision):
        result = model(paired_input(1, 2, 64, 68))
        loss = result.readout().float().square().mean()
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path/"saved.pt"
    save_checkpoint(checkpoint, model, state=result.state, optimizer=optimizer)
    expected = checkpoint_identity(checkpoint, checkpoint_stat(checkpoint))["sha256"]
    output = tmp_path/"restore.json"
    child = subprocess.run([sys.executable, "-m", "benchmarks.restore", "--checkpoint", str(checkpoint),
                            "--precision", precision, "--batch", "1", "--length", "2", "--threads", "4",
                            "--expected-checkpoint-sha256", expected, "--output", str(output)],
                           text=True, capture_output=True, timeout=40)
    assert child.returncode == 0, child.stderr+child.stdout
    row = json.loads(output.read_text())
    assert row["state"] == "completed_finite"
    assert row["checkpoint_file"]["sha256"] == expected
    assert row["loaded_state_position"] == 2 and row["restored_state_position"] == 4
    assert row["load_complete_seconds"] > 0 and row["next_update_complete_seconds"] > 0
    assert all(value == 1 for value in row["optimizer_before"]["steps"].values())
    assert all(value == 2 for value in row["optimizer_after"]["steps"].values())
    assert row["optimizer_after"]["groups"][0]["lr"] == 2e-4
    assert row["optimizer_after"]["groups"][0]["foreach"] is True
    assert row["parameter_update_sample"]["changed_scalars"] > 0
    assert row["private_state_detached"] and row["optimizer_step_increment_errors"] == []
    assert row["numerical_equivalence_qualification"] is False
    assert set(row["phases"]) == {"load_model_state", "load_optimizer", "validate_loaded", "zero_grad",
                                 "forward_loss_flush", "backward", "optimizer", "detach", "validate_updated"}
    assert row["next_update_complete_seconds"] >= sum(row["phases"][name]["seconds"] for name in
                                                    ("zero_grad", "forward_loss_flush", "backward", "optimizer", "detach"))


def test_optimizerless_checkpoint_is_not_mislabelled_as_restored_training(tmp_path):
    model = TideModel(ModelConfig(topology=TopologyConfig(n=1, layers=1), d=64))
    checkpoint = tmp_path/"weights.pt"
    save_checkpoint(checkpoint, model)
    output = tmp_path/"restore.json"
    child = subprocess.run([sys.executable, "-m", "benchmarks.restore", "--checkpoint", str(checkpoint),
                            "--precision", "fp32", "--batch", "1", "--length", "1", "--threads", "4",
                            "--output", str(output)], text=True, capture_output=True, timeout=30)
    assert child.returncode == 1
    row = json.loads(output.read_text())
    assert row["state"] == "failed"
    assert "no optimizer state" in row["error"]
    assert "next_update_complete_seconds" not in row
    assert "checkpoint_file" in row
