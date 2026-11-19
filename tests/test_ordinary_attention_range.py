"""Weak priors with ordinary feature/score magnitudes, including all roots.

CPU FP64 direct quotients are local oracles, never a production fallback.
TIDE_STABLE_DEVICE selects the FP32/BF16 device for frozen qualification.
"""
import os

import pytest
import torch

from tide.attention import _weighted_softmax, weighted_attention


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
    return torch.device(name)


def formula(q, k, v, w, mask):
    logits = q @ k.transpose(-2, -1) / q.shape[-1] ** .5
    terms = logits.exp() * w * mask
    denominator = terms.sum(-1, keepdim=True)
    return terms @ v / torch.where(denominator > 0, denominator, 1)


@pytest.mark.parametrize("backend", ["math", "tiled", "auto"])
@pytest.mark.parametrize("mass", [1e-30, 1e-40])
def test_weak_ordinary_values_and_parameter_input_vjp(device, backend, mass):
    # Projection-like shared roots; all Q/K/V entries stay O(1). The small
    # upstream avoids a genuinely unrepresentable 1/mass input derivative.
    rng = torch.Generator().manual_seed(173)
    inputs = [torch.randn(shape, generator=rng) * .2 for shape in
              ((2, 1, 2, 32), (1, 1, 3, 32), (1, 1, 3, 32))]
    inputs.append(torch.tensor([mass, 2*mass, 0.]).reshape(1, 1, 1, 3))
    mask = torch.tensor([[[[True, True, True], [True, False, False]]]])
    grad = torch.randn(2, 1, 2, 32, generator=rng) * 1e-20
    roots = [value.to(device).requires_grad_() for value in inputs]
    refs = [value.double().requires_grad_() for value in inputs]
    actual = weighted_attention(*roots, mask.to(device), backend)
    expected = formula(*refs, mask)
    got = torch.autograd.grad(actual, roots, grad.to(device), retain_graph=True)
    again = torch.autograd.grad(actual, roots, grad.to(device))
    want = torch.autograd.grad(expected, refs, grad.double())
    torch.testing.assert_close(actual.cpu(), expected.float(), rtol=1e-5, atol=1e-7)
    for a, b, repeat in zip(got, want, again):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.cpu(), b.float(), rtol=1e-5, atol=1e-26)
        torch.testing.assert_close(a, repeat, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["math", "tiled", "auto"])
def test_tiny_singleton_is_independent_of_its_prior(device, backend):
    q = torch.full((1, 1, 1, 32), .2, device=device, requires_grad=True)
    k = torch.full_like(q, .1, requires_grad=True)
    v = torch.full_like(q, 3., requires_grad=True)
    w = torch.tensor([1e-40], device=device, requires_grad=True)
    output = weighted_attention(q, k, v, w, backend=backend)
    torch.testing.assert_close(output, v, rtol=0, atol=0)
    grads = torch.autograd.grad(output.sum(), (q, k, v, w))
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(torch.count_nonzero(g) == 0 for g in (grads[0], grads[1], grads[3]))
    torch.testing.assert_close(grads[2], torch.ones_like(v), rtol=0, atol=0)


def test_softmax_cotangent_product_never_materializes_infinite_scale(device):
    scores = torch.zeros(2, device=device, requires_grad=True)
    weights = torch.tensor([1e-40, 1e-40], device=device, requires_grad=True)
    p = _weighted_softmax(scores, weights, torch.tensor(True, device=device))
    gs, gw = torch.autograd.grad(p, (scores, weights), torch.tensor([1e-20, 0.], device=device))
    torch.testing.assert_close(gs.cpu(), torch.tensor([2.5e-21, -2.5e-21]), rtol=1e-5, atol=0)
    expected = torch.tensor([.25, -.25], dtype=torch.float64) * 1e-20 / float(weights[0].detach().cpu())
    torch.testing.assert_close(gw.cpu(), expected.float(), rtol=1e-5, atol=0)


@pytest.mark.parametrize("backend", ["math", "tiled", "auto"])
def test_unit_features_and_cotangent_have_finite_prior_gradient(device, backend):
    q = torch.zeros(1, 1, 1, 32, device=device)
    k = torch.zeros(1, 1, 2, 32, device=device)
    v = torch.ones_like(k)
    v[..., 1, 0] += 2**-20
    w = torch.tensor([1e-40, 1e-40], device=device, requires_grad=True)
    actual = weighted_attention(q, k, v, w, backend=backend)
    gw = torch.autograd.grad(actual[..., 0].sum(), w)[0]
    torch.testing.assert_close(actual[..., 0], torch.full_like(actual[..., 0], 1+2**-21), rtol=0, atol=0)
    expected = torch.tensor([-2**-22, 2**-22], dtype=torch.float64) / w.detach().cpu().double()
    torch.testing.assert_close(gw.cpu(), expected.float(), rtol=1e-5, atol=0)


def test_ordinary_native_admission_and_grad_mode_match(device):
    q = torch.zeros(1, 1, 1, 32, device=device)
    v = torch.ones_like(q)
    w = torch.tensor([1e-40], device=device)
    native = "npu_fusion" if device.type == "npu" else "sdpa"
    with pytest.raises(ValueError, match="weak priors"):
        weighted_attention(q, q, v, w, backend=native)
    with torch.inference_mode():
        inferred = weighted_attention(q, q, v, w)
    trained = weighted_attention(q.requires_grad_(), q, v, w.requires_grad_())
    torch.testing.assert_close(trained, inferred, rtol=0, atol=0)


def test_true_ordinary_prior_gradient_overflow_is_explicit(device):
    q = torch.zeros(1, 1, 1, 32, device=device)
    k = torch.zeros(1, 1, 2, 32, device=device)
    v = torch.tensor([0., 1.], device=device).reshape(1, 1, 2, 1).expand_as(k)
    w = torch.tensor([1e-40, 1e-40], device=device, requires_grad=True)
    with pytest.raises(ArithmeticError, match="arithmetic range"):
        weighted_attention(q, k, v, w).sum().backward()


@pytest.mark.parametrize("backend", ["math", "tiled", "auto"])
def test_bf16_payload_keeps_fp32_weak_prior_arithmetic(device, backend):
    q = torch.zeros(1, 1, 1, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(1, 1, 2, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.tensor([1., 2.], device=device, dtype=torch.bfloat16).reshape(1, 1, 2, 1).expand_as(k).clone().requires_grad_()
    w = torch.tensor([1e-30, 3e-30], device=device, requires_grad=True)
    output = weighted_attention(q, k, v, w, backend=backend)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.full_like(output, 1.75), rtol=0, atol=0)
    dq, dk, dv, dw = torch.autograd.grad(output[..., 0].sum(), (q, k, v, w))
    assert torch.count_nonzero(dq) == torch.count_nonzero(dk) == 0
    torch.testing.assert_close(dv[0, 0, :, 0], v.new_tensor([.25, .75]), rtol=0, atol=0)
    total = w.detach().cpu().double().sum()
    expected = torch.tensor([-.75, .25], dtype=torch.float64) / total
    torch.testing.assert_close(dw.cpu(), expected.float(), rtol=1e-5, atol=0)
