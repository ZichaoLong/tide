"""Independent causal/readout and continuation checks for the pilot LM boundary."""
import copy
import math

import pytest
import torch
from torch.nn import functional as F

from pilotlab.models import CausalAttention, TransformerState, build_model, detach_state


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


TF = dict(family="transformer", d=32, layers=2, heads=2, ffn_width=64, window=3)
SETTLE = dict(family="tide", config=dict(topology=dict(kind="fc", n=2, layers=2), d=64,
                                       memory="K", sharing="region_shared", budget=1, attention_backend="math"))
TIMED = dict(family="tide", config=dict(topology=dict(kind="multiscale", profile="compact-5-dd4", n=2,
                                                     timing="timed"), d=64, memory="K",
                                      sharing="region_shared", budget=1, attention_backend="math"))
SSD = dict(family="tide", config=dict(topology=dict(kind="fc", n=2, layers=2), d=64,
                                    memory="S", sharing="region_shared", budget=1, attention_backend="math"))


@pytest.mark.parametrize("spec", [TF, SETTLE, TIMED, SSD], ids=["transformer", "settle-k", "timed-k", "settle-s"])
def test_causal_prefix_and_partitioned_inference(spec):
    torch.manual_seed(712)
    model = build_model(spec, 31).eval()
    tokens = torch.tensor([[1, 4, 2, 7, 3, 5]], dtype=torch.int64)
    changed = tokens.clone()
    changed[:, 3:] = torch.tensor([9, 11, 12])
    with torch.no_grad():
        full = model(tokens, record_work=True)
        perturbed = model(changed)
        first = model(tokens[:, :2])
        second = model(tokens[:, 2:4], first.state)
        third = model(tokens[:, 4:], second.state)
        state, singles = None, []
        for position in range(tokens.shape[1]):
            result = model(tokens[:, position:position + 1], state)
            state = result.state
            singles.append(result.logits)
    torch.testing.assert_close(full.logits[:, :3], perturbed.logits[:, :3], rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(full.logits, torch.cat([first.logits, second.logits, third.logits], 1), rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(full.logits, torch.cat(singles, 1), rtol=3e-4, atol=3e-5)
    assert state.position == tokens.shape[1]
    assert full.logits.shape == (1, 6, 31)
    assert full.work["input_positions"] == 6
    assert 0 <= full.work["empty_output_positions"] <= 6
    if spec is TIMED:
        assert full.work["logical_ticks"] == 24
        assert full.work["readout_policy"] == "timed_last_tick_before_next_external_input"


@pytest.mark.parametrize("spec", [TF, SETTLE, TIMED, SSD], ids=["transformer", "settle-k", "timed-k", "settle-s"])
def test_actual_backbone_and_embedding_update(spec):
    torch.manual_seed(71)
    model = build_model(spec, 31)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    output = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.int64))
    loss = F.cross_entropy(output.logits.flatten(0, 1), torch.tensor([2, 3, 4, 5]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embedding.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    prefix = "core." if spec["family"] == "tide" else "blocks."
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for n, p in model.named_parameters() if n.startswith(prefix))
    optimizer.step()
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(prefix))
    assert not torch.equal(before["embedding.weight"], model.embedding.weight)
    detached = detach_state(output.state)
    assert detached.position == output.state.position


def test_transformer_continuation_keeps_prefix_gradients():
    torch.manual_seed(4)
    spec = dict(TF, window=8)
    whole = build_model(spec, 23)
    chunked = copy.deepcopy(whole)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    loss_a = F.cross_entropy(whole(tokens).logits[:, -1], torch.tensor([7]))
    loss_a.backward()
    prefix = chunked(tokens[:, :3])
    loss_b = F.cross_entropy(chunked(tokens[:, 3:], prefix.state).logits[:, -1], torch.tensor([7]))
    loss_b.backward()
    assert torch.count_nonzero(chunked.embedding.weight.grad[1]) > 0
    for (name_a, parameter_a), (name_b, parameter_b) in zip(whole.named_parameters(), chunked.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=5e-4, atol=1e-5)


def test_attention_against_scalar_rotary_window_formula():
    torch.manual_seed(9)
    attention = CausalAttention(d=8, heads=2, window=3)
    x = torch.randn(1, 5, 8)
    actual, _ = attention(x, None, 0)
    projection = F.linear(x.double(), attention.qkv.weight.double()).reshape(5, 3, 2, 4)
    rotated = projection.clone()
    for position in range(5):
        for kind in (0, 1):
            for head in range(2):
                for pair in range(2):
                    phase = position * 10000.0 ** (-2 * pair / 4)
                    a, b = projection[position, kind, head, 2 * pair:2 * pair + 2].tolist()
                    rotated[position, kind, head, 2 * pair] = a * math.cos(phase) - b * math.sin(phase)
                    rotated[position, kind, head, 2 * pair + 1] = a * math.sin(phase) + b * math.cos(phase)
    context = torch.zeros((5, 2, 4), dtype=torch.float64)
    for query in range(5):
        for head in range(2):
            keys = list(range(max(0, query - 2), query + 1))
            scores = [(rotated[query, 0, head] * rotated[key, 1, head]).sum().item() / 2 for key in keys]
            weights = [math.exp(score - max(scores)) for score in scores]
            denominator = sum(weights)
            for key, weight in zip(keys, weights):
                context[query, head] += projection[key, 2, head] * (weight / denominator)
    expected = F.linear(context.reshape(1, 5, 8), attention.out.weight.double())
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)


def test_invalid_tokens_and_cache_are_rejected():
    model = build_model(TF, 17)
    with pytest.raises(TypeError):
        model(torch.ones(1, 2))
    with pytest.raises(ValueError):
        model(torch.ones(1, 0, dtype=torch.int64))
    with pytest.raises(ValueError):
        model(torch.ones(1, 2, dtype=torch.int64), TransformerState(0, ()))
