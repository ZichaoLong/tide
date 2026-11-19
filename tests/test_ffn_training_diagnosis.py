"""Actual cotangents, repeated-owner accumulation and bounded call retention."""
import json
import pytest
import torch

from scripts.diagnose_ffn_training_vjp import LastDownCall, compare_tensor, literal_linear_vjp, local_replay
from tide.parameters import NodeLinear, NodeOwners


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_literal_repeated_owner_vjp_matches_autograd_and_both_execution_paths(dtype):
    torch.manual_seed(830)
    x = torch.randn(4, 2, 3, 5, dtype=dtype)
    weight = torch.randn(4, 7, 5, dtype=dtype)
    ids = torch.tensor([2, 0, 2, 1])
    incoming = torch.randn(4, 2, 3, 7, dtype=dtype)
    expected = literal_linear_vjp(x, weight, ids, incoming)
    xx, ww = x.double().requires_grad_(), weight.double().requires_grad_()
    output = torch.stack([row @ ww[owner].T for row, owner in zip(xx, ids.tolist())])
    dx, dw = torch.autograd.grad(output, (xx, ww), incoming.double())
    torch.testing.assert_close(expected["output"], output, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(expected["input_vjp"], dx, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(expected["weight_vjp"], dw, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(expected["weight_vjp"][3]) == 0
    repeated = sum(incoming[i].double().reshape(-1, 7).T @ x[i].double().reshape(-1, 5) for i in (0, 2))
    torch.testing.assert_close(expected["weight_vjp"][2], repeated, atol=1e-12, rtol=1e-12)
    capture = dict(x=x, weight=weight, ids=ids, incoming=incoming,
                   possible_ids=(0, 1, 2, 3), exact_ids=None)
    for access in ("gather", "grouped"):
        replay = local_replay(capture, access)
        for key in expected:
            torch.testing.assert_close(replay[key].double(), expected[key], atol=3e-6 if dtype == torch.float32 else 1e-12,
                                       rtol=3e-6 if dtype == torch.float32 else 1e-12)


def test_final_call_captures_real_incoming_and_input_vjp_without_earlier_call_gradient():
    torch.manual_seed(831)
    layer = NodeLinear(4, 5, 7).double()
    first = torch.randn(1, 2, 5, dtype=torch.float64, requires_grad=True)
    last = torch.randn(3, 2, 5, dtype=torch.float64, requires_grad=True)
    owners = NodeOwners(torch.tensor([2, 0, 2]), (0, 1, 2))
    incoming = torch.cos(torch.arange(42, dtype=torch.float64)).reshape(3, 2, 7)
    with LastDownCall(layer) as recorder:
        earlier = layer(first, NodeOwners(torch.tensor([2]), (2,)))
        output = layer(last, owners)
        (earlier.square().sum() + (output * incoming).sum()).backward()
        captured = recorder.finish()
    assert captured["call_index"] == captured["observed_forward_calls"] == 2
    assert captured["original_owner_ids"] == [2, 0, 2]
    torch.testing.assert_close(captured["incoming"], incoming, atol=0, rtol=0)
    expected = literal_linear_vjp(captured["x"], captured["weight"], captured["ids"], captured["incoming"])
    torch.testing.assert_close(captured["input_vjp"], expected["input_vjp"], atol=1e-12, rtol=1e-12)
    assert not torch.equal(layer.weight.grad[2], expected["weight_vjp"][2])
    assert not layer._forward_hooks


def test_capture_limit_fails_without_silently_selecting_another_call():
    layer = NodeLinear(2, 3, 4)
    x = torch.randn(2, 1, 3, requires_grad=True)
    with LastDownCall(layer, max_bytes=128) as recorder:
        with pytest.raises(MemoryError, match="rejected before cotangent clone"):
            layer(x, NodeOwners(torch.tensor([0, 1]), (0, 1))).sum().backward()
        assert "incoming" not in recorder.last and "input_vjp" not in recorder.last
    assert not layer._forward_hooks


def test_shared_or_biased_projection_is_rejected_for_this_narrow_formula():
    for layer in (NodeLinear(2, 3, 4, groups=[0, 0]), NodeLinear(2, 3, 4, bias=True)):
        with pytest.raises(ValueError, match="independent bias-free"):
            LastDownCall(layer)


def test_nonfinite_comparison_remains_explicit_and_serializes_without_json_nan():
    actual = torch.tensor([0., float("nan"), float("inf")])
    expected = torch.tensor([0., 1., float("inf")])
    result = compare_tensor(actual, expected, dict(atol=1e-6, rtol=2e-3))
    assert result["state"] == "nonfinite"
    assert result["nonfinite_pairs"] == result["failed_elements"] == 2
    assert result["worst_actual"] is None
    json.dumps(result, allow_nan=False)
