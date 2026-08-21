from __future__ import annotations

import json
import pathlib

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from tide.data import UInt32TokenWriter
from tide.qwen3 import TideQwen3ForCausalLM
from tide.runtime import RuntimeRequest, resolve_runtime
from tide.train import _scheduled_learning_rate_scale, run_fixed_diagnostics, run_training


class _Runtime:
    torch = torch
    device = torch.device("cpu")

    @staticmethod
    def synchronize() -> None:
        return None


class _Dataset:
    sequence_length = 8

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(61)
        self.rows = [torch.randint(0, 64, (8,), generator=generator) for _ in range(4)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.rows[index]


def _model() -> TideQwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = TideQwen3ForCausalLM(
        Qwen3ForCausalLM(config),
        layer_indices=[1],
        profile="bo",
        state_size=8,
    )
    with torch.no_grad():
        for receiver in model.wrapped_layers[1].receiver_group.ffns:
            receiver.down_proj.weight.normal_(std=0.03)
    return model


def test_warmup_stable_schedule_is_token_based() -> None:
    assert _scheduled_learning_rate_scale(
        schedule="warmup-stable",
        step=1,
        tokens=5,
        total_steps=100,
        warmup_ratio=0.05,
        warmup_tokens=20,
        minimum_ratio=0.1,
    ) == 0.25
    assert _scheduled_learning_rate_scale(
        schedule="warmup-stable",
        step=99,
        tokens=25,
        total_steps=100,
        warmup_ratio=0.05,
        warmup_tokens=20,
        minimum_ratio=0.1,
    ) == 1.0


def test_fixed_diagnostics_records_paths_and_state_interventions(
    tmp_path: pathlib.Path,
) -> None:
    model = _model()
    initial = run_fixed_diagnostics(
        model=model,
        runtime=_Runtime(),
        dataset=_Dataset(),
        batch_size=2,
        max_tokens=32,
        output_dir=tmp_path,
        label="initial",
    )
    current = run_fixed_diagnostics(
        model=model,
        runtime=_Runtime(),
        dataset=_Dataset(),
        batch_size=2,
        max_tokens=32,
        output_dir=tmp_path,
        label="token-0000000032",
    )
    assert initial["path"]["churn_from_initial"] == 0.0
    assert current["path"]["churn_from_initial"] == 0.0
    assert set(current["state_interventions"]) == {"no_read", "clear", "shuffle"}
    assert (tmp_path / "diagnostics" / "probe-initial.pt").is_file()


def test_random_training_milestone_end_to_end(tmp_path: pathlib.Path) -> None:
    model_dir = tmp_path / "model"
    _model().config.save_pretrained(model_dir)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    files = {}
    for name in ("train", "validation"):
        writer = UInt32TokenWriter(data_dir / f"{name}.bin")
        for token in range(32):
            writer.append(token % 64)
        files[name] = writer.close()
    manifest = {
        "schema_version": 1,
        "packing": {"sequence_length": 8},
        "files": files,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "run"
    arguments = {
        "model_path": str(model_dir),
        "initialization": "random",
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "init_from": None,
        "resume": None,
        "profile": "bo",
        "layer_indices": [1],
        "receiver_count": 4,
        "expert_count": 8,
        "state_size": 8,
        "implementation": "packed",
        "scan_implementation": "vectorized",
        "attention_implementation": "eager",
        "sequence_length": 8,
        "micro_batch_size": 1,
        "gradient_accumulation": 1,
        "evaluation_batch_size": 2,
        "max_tokens": 16,
        "max_steps": None,
        "validation_tokens": 16,
        "backbone_lr": 1e-3,
        "extension_lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "balance_coefficient": 0.01,
        "router_z_coefficient": 0.001,
        "warmup_ratio": 0.05,
        "minimum_lr_ratio": 0.1,
        "lr_schedule": "warmup-stable",
        "warmup_tokens": 8,
        "milestone_tokens": [8, 16],
        "diagnostic_tokens": 16,
        "checkpoint_every": 0,
        "checkpoint_keep_last": 1,
        "log_every": 1,
        "tracking": "off",
        "trackio_project": "test",
        "run_name": "tiny-random",
        "run_initial_validation": True,
        "device": "cpu",
        "device_index": None,
        "dtype": "float32",
        "seed": 67,
    }
    result = run_training(
        resolve_runtime(RuntimeRequest("cpu", 0, "float32", 67)),
        arguments,
    )
    assert result["tokens"] == 16
    checkpoint_dir = output_dir / "checkpoints"
    assert not (checkpoint_dir / "token-0000000008-step-000001.pt").exists()
    assert (checkpoint_dir / "token-0000000016-step-000002.pt").is_file()
    latest = json.loads((checkpoint_dir / "latest.json").read_text())
    assert latest["path"] == "token-0000000016-step-000002.pt"
    assert latest["retention"] == {"keep_last": 1}
    retention = [
        json.loads(line)
        for line in (checkpoint_dir / "retention.jsonl").read_text().splitlines()
    ]
    assert [event["path"] for event in retention] == [
        "token-0000000008-step-000001.pt"
    ]
    assert (output_dir / "diagnostics" / "probe-token-0000000008.pt").is_file()
    assert (output_dir / "diagnostics" / "probe-token-0000000016.pt").is_file()
    assert json.loads((output_dir / "tracking.json").read_text())["status"] == "disabled"
    assert json.loads((output_dir / "manifest.json").read_text())["status"] == "completed"
