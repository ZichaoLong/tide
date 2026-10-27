"""Quantized operands against literal real Jacobians, including zero SSD reads."""
import math
import pytest
import torch
from tidegraph.weighted.normalization import norm, epsilon_l2, scaled_rms
from tidegraph.weighted.attention_mean import attention_mean


@pytest.mark.parametrize("kind", ["rms", "l2", "scaled"])
def test_half_normalization_literal_jacobian_and_broadcast_scale(kind):
    x = torch.tensor([[0., 0.], [1e-4, -2e-4], [3., 4.]], dtype=torch.float16, requires_grad=True)
    scale = torch.tensor([.3, -.7], dtype=torch.float16, requires_grad=True)
    bar = torch.tensor([[.4, -.2], [.6, .8], [-.3, .5]], dtype=torch.float16)
    epsilon = 1e-5
    value = scaled_rms(x, scale, epsilon) if kind == "scaled" else (norm if kind == "rms" else epsilon_l2)(x, epsilon)
    expected, dx, ds = torch.empty_like(x), torch.empty_like(x), torch.zeros(2, dtype=torch.float64)
    divisor = 1 if kind == "l2" else 2
    for i, row in enumerate(x.detach().double().tolist()):
        den = math.sqrt(epsilon+sum(v*v for v in row)/divisor)
        gains = scale.detach().double().tolist() if kind == "scaled" else [1., 1.]
        cotangent = bar[i].double().tolist()
        for j in range(2):
            expected[i, j] = gains[j]*row[j]/den
            dx[i, j] = sum(cotangent[k]*gains[k]*(float(j == k)/den-row[j]*row[k]/(divisor*den**3)) for k in range(2))
            ds[j] += cotangent[j]*row[j]/den
    torch.testing.assert_close(value, expected, atol=1e-3, rtol=2e-3)
    gradients = torch.autograd.grad(value, (x, scale), bar, allow_unused=True)
    assert torch.isfinite(gradients[0]).all()
    torch.testing.assert_close(gradients[0], dx, atol=1e-3, rtol=2e-3)
    if kind == "scaled":
        torch.testing.assert_close(gradients[1], ds.half(), atol=1e-3, rtol=2e-3)
    else:
        assert gradients[1] is None


def test_half_scaled_rms_original_version_and_first_order():
    x = torch.zeros(2, dtype=torch.float16, requires_grad=True)
    scale = torch.ones(2, dtype=torch.float16, requires_grad=True)
    y = scaled_rms(x, scale, 1e-5)
    dx, ds = torch.autograd.grad(y.sum()*0, (x, scale))
    assert torch.equal(dx, torch.zeros_like(x)) and torch.equal(ds, torch.zeros_like(scale))
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(scaled_rms(x, scale, 1e-5).sum(), x, create_graph=True)
    y = scaled_rms(x, scale, 1e-5)
    with torch.no_grad():
        scale.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()


@pytest.mark.parametrize("score_dtype", [torch.float16, torch.float32])
def test_half_attention_literal_two_entry_jacobian(score_dtype):
    scores = torch.tensor([10000., 10000.], dtype=score_dtype, requires_grad=True)
    values = torch.tensor([[.25, -.5], [-.75, .3]], dtype=torch.float16, requires_grad=True)
    bar = torch.tensor([.2, -.4], dtype=torch.float16)
    actual = attention_mean(scores, values)
    torch.testing.assert_close(actual, values.float().mean(0).half(), atol=0, rtol=0)
    ds, dv = torch.autograd.grad(actual, (scores, values), bar)
    operands = values.detach()
    contrast = sum(float(bar[j])*(float(operands[0, j])-float(operands[1, j])) for j in range(2))/4
    torch.testing.assert_close(ds, scores.new_tensor([contrast, -contrast]), atol=1e-4, rtol=2e-3)
    torch.testing.assert_close(dv, (bar.float()/2).expand(2, -1).half(), atol=0, rtol=0)
