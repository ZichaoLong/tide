from __future__ import annotations

import copy
import io
import os
import pathlib

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from tide.qwen3 import TideQwen3ForCausalLM, load_qwen3_model


def _tiny_model() -> Qwen3ForCausalLM:
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
    return Qwen3ForCausalLM(config)


def test_qwen_neutral_injection_backward_and_state_dict_round_trip() -> None:
    torch.manual_seed(29)
    baseline = _tiny_model().eval()
    expanded_base = copy.deepcopy(baseline)
    expanded = TideQwen3ForCausalLM(
        expanded_base,
        layer_indices=[1],
        profile="bo",
        state_size=8,
        implementation="packed",
    ).eval()
    input_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        baseline_logits = baseline(input_ids=input_ids, attention_mask=attention_mask).logits
        expanded_logits = expanded(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.equal(expanded_logits, baseline_logits)

    expanded.train()
    output = expanded(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    output.loss.backward()
    group = expanded.wrapped_layers[1].receiver_group
    assert sum(receiver.down_proj.weight.grad.norm() for receiver in group.ffns) > 0

    payload = io.BytesIO()
    torch.save(expanded.state_dict(), payload)
    payload.seek(0)
    restored = TideQwen3ForCausalLM(
        copy.deepcopy(baseline),
        layer_indices=[1],
        profile="bo",
        state_size=8,
        implementation="packed",
    )
    restored.load_state_dict(torch.load(payload, weights_only=True), strict=True)


def test_qwen_upcycled_moe_preserves_initial_logits_and_has_router_gradient() -> None:
    torch.manual_seed(41)
    baseline = _tiny_model().eval()
    expanded = TideQwen3ForCausalLM(
        copy.deepcopy(baseline),
        layer_indices=[1],
        profile="upcycled-moe",
        expert_count=4,
    ).eval()
    input_ids = torch.randint(0, 64, (4, 8))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = baseline(input_ids=input_ids, attention_mask=attention_mask).logits
        actual = expanded(input_ids=input_ids, attention_mask=attention_mask).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    expanded.train()
    output = expanded(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()
    router_gradient = expanded.moe_layers[1].router.weight.grad
    assert router_gradient is not None
    assert router_gradient.norm() > 0


def test_random_initialization_uses_config_without_checkpoint_weights(
    tmp_path: pathlib.Path,
) -> None:
    config = _tiny_model().config
    config.save_pretrained(tmp_path)
    arguments = {
        "dtype": torch.float32,
        "layer_indices": [1],
        "profile": "selected-dispatch",
        "receiver_count": 4,
        "expert_count": 8,
        "state_size": 8,
        "implementation": "packed",
        "scan_implementation": "vectorized",
        "balance_coefficient": 0.01,
        "router_z_coefficient": 0.001,
        "attention_implementation": "eager",
        "initialization": "random",
    }
    torch.manual_seed(53)
    selected = load_qwen3_model(str(tmp_path), **arguments)
    torch.manual_seed(53)
    bo = load_qwen3_model(str(tmp_path), **{**arguments, "profile": "bo"})
    torch.manual_seed(53)
    stateless = load_qwen3_model(
        str(tmp_path), **{**arguments, "profile": "stateless"}
    )
    for left, right in zip(selected.parameters(), bo.parameters(), strict=True):
        assert torch.equal(left, right)
    for left, right in zip(stateless.parameters(), bo.parameters(), strict=True):
        assert torch.equal(left, right)


def test_qwen_stateless_control_ignores_state_interventions() -> None:
    torch.manual_seed(57)
    expanded = TideQwen3ForCausalLM(
        _tiny_model(),
        layer_indices=[1],
        profile="stateless",
        state_size=8,
        implementation="packed",
    ).eval()
    group = expanded.wrapped_layers[1].receiver_group
    extension_ids = expanded.extension_parameter_ids()
    assert id(group.router.weight) in extension_ids
    assert id(group.ffns[0].down_proj.weight) in extension_ids
    assert id(group.observers[0].weight) not in extension_ids
    assert id(group.decay_logits) not in extension_ids
    assert id(group.state_projections[0].weight) not in extension_ids
    with torch.no_grad():
        for receiver in group.ffns:
            receiver.down_proj.weight.normal_(std=0.03)
    input_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        normal = expanded(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tide_return_artifacts=True,
        )
        changed = expanded(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tide_read_state=False,
            tide_clear_positions={1: (4,)},
            tide_shuffle_positions={1: (4,)},
        )
    assert torch.equal(normal.logits, changed.logits)
    assert torch.count_nonzero(normal.states[1]).item() == 0
    assert normal.metrics[1]["receive_counts"].sum().item() == 0


def test_qwen_state_interventions_reach_receiver_scan() -> None:
    torch.manual_seed(59)
    expanded = TideQwen3ForCausalLM(
        _tiny_model(),
        layer_indices=[1],
        profile="bo",
        state_size=8,
        implementation="packed",
    ).eval()
    with torch.no_grad():
        for receiver in expanded.wrapped_layers[1].receiver_group.ffns:
            receiver.down_proj.weight.normal_(std=0.03)
    input_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        normal = expanded(input_ids=input_ids, attention_mask=attention_mask).logits
        no_read = expanded(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tide_read_state=False,
        ).logits
        cleared = expanded(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tide_clear_positions={1: (4,)},
        ).logits
        shuffled = expanded(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tide_shuffle_positions={1: (4,)},
        ).logits
    assert not torch.equal(normal, no_read)
    assert not torch.equal(normal, cleared)
    assert not torch.equal(normal, shuffled)
