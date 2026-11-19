"""Bounded mode capture, actual dispatch observation and independent row oracle."""
import importlib

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig, placement_adamw
from scripts.check_training_paths import observe, training_data
from scripts.diagnose_attention_modes import Calls, CaptureLimit, diagnose, dispatch, scalar_rows


def test_scalar_oracle_obeys_compact_priors_masks_and_zero_rows():
    q = torch.zeros(2, 2, 3, 4)
    k = torch.zeros(2, 2, 5, 4)
    v = torch.arange(40.).reshape(2, 2, 5, 2)
    weights = torch.tensor([[1., 0., 2., 0., 1.], [0., 1., 0., 3., 0.]])
    mask = torch.ones(3, 5, dtype=torch.bool); mask[1] = False
    value = scalar_rows(dict(q=q, k=k, v=v, weights=weights, mask=mask), [0, 1, 3, 6, 9])
    expected = torch.stack(((v[0, 0, 0]+2*v[0, 0, 2]+v[0, 0, 4])/4,
        torch.zeros(2), (v[0, 1, 0]+2*v[0, 1, 2]+v[0, 1, 4])/4,
        (v[1, 0, 1]+3*v[1, 0, 3])/4, (v[1, 1, 1]+3*v[1, 1, 3])/4)).double()
    torch.testing.assert_close(value, expected, atol=1e-14, rtol=1e-14)


def test_dispatch_observes_zero_prior_native_and_tiled_modes_without_changing_policy():
    module = importlib.import_module("tide.attention")
    original = module._native_attention
    q = torch.zeros(1, 1, 2, 4)
    k = torch.zeros(1, 1, 3, 4)
    v = torch.arange(12.).reshape_as(k)
    for grad, expected in ((False, "sdpa"), (True, "tiled")):
        w = torch.tensor([[1., 0., 2.]], requires_grad=grad)
        with torch.set_grad_enabled(grad), dispatch("cpu") as path:
            out = module.weighted_attention(q, k, v, w)
        assert path["effective"] == expected
        assert out.requires_grad == grad
        oracle = scalar_rows(dict(q=q, k=k, v=v, weights=w, mask=None), [0, 1])
        torch.testing.assert_close(out.reshape(2, 4).double(), oracle, atol=1e-6, rtol=1e-6)
    assert module._native_attention is original


def test_small_training_anchor_locates_injected_mode_error_and_oracle_exposes_it(monkeypatch):
    torch.manual_seed(410)
    config = ModelConfig(topology=TopologyConfig(n=2, layers=2), d=64, memory="K", attention_backend="math")
    model = TideModel(config)
    optimizer = placement_adamw(model, lr=1e-4, foreach=False)
    inputs = [torch.randn(1, 2, 64), torch.randn(1, 2, 64)]
    _, state, _ = observe(model, optimizer, None, inputs[0], "fp32", ["cpu"], 1, update=True)
    bundle = dict(config=config.to_dict(), anchor=training_data(model, optimizer, state), inputs=inputs)
    module = importlib.import_module("tide.attention")
    original = module.weighted_attention
    def faulty(*args, **kwargs):
        out = original(*args, **kwargs)
        return out + .01 if torch.is_grad_enabled() else out
    monkeypatch.setattr(module, "weighted_attention", faulty)
    result, payload = diagnose(bundle, device="cpu", chunk=1, oracle_rows=2)
    assert result["state"] == "first_attention_difference" and result["first_index"] == 0
    assert result["actual_model_forward_attempts"] == 3
    assert result["inference_repeat_exact"] and result["input_values_equal"]
    assert result["grad_forward"]["output_requires_grad"]
    assert result["same_input"]["replays"]["grad_forward"]["matches_capture"]["exact"]
    assert result["same_input"]["oracle"]["inference"]["outside_threshold"] == 0
    assert result["same_input"]["oracle"]["grad_forward"]["outside_threshold"] > 0
    assert payload["same_input"]["oracle"].shape[0] == 2
    bounded, _ = diagnose(bundle, device="cpu", chunk=1, elements=1)
    assert bounded["state"] == "capture_limit"
    assert bounded["actual_model_forward_attempts"] == 1


def test_call_capture_bound_stops_before_the_operator():
    calls = Calls("cpu", elements=1, max_calls=1)
    q = torch.zeros(1, 1, 2, 4)
    with calls.install(), pytest.raises(CaptureLimit, match="input/output elements"):
        calls.call("test", (q, q, q, None, None), "math")
