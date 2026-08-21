from __future__ import annotations

import copy
import io
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from tide.qwen3 import TideQwen3ForCausalLM


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
