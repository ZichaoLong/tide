"""Independent zero-derivative identities and real zero-prior right derivatives."""
import os

import pytest
import torch

from tide.attention import weighted_attention


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
    return torch.device(name)


@pytest.mark.parametrize("backend", ["math", "tiled", "auto"])
@pytest.mark.parametrize("mass_scaled,amount", [(False, 1.), (False, 1e-30), (True, .5), (True, 2.)])
def test_single_key_cannot_manufacture_qk_gradients_or_adam_updates(device, backend, mass_scaled, amount):
    rng = torch.Generator().manual_seed(843)
    q, k, v = [torch.randn(2, 1, 3, 32, generator=rng) for _ in range(3)]
    k, v = k[..., :1, :], v[..., :1, :]
    q, k, v = [torch.nn.Parameter(t.to(device)) for t in (q, k, v)]
    w = torch.nn.Parameter(torch.tensor([amount], device=device))
    root = torch.randn(q.shape, generator=rng).to(device) * .01
    optimizer = torch.optim.AdamW((q, k), lr=.001, weight_decay=0., foreach=False)
    initial = [t.detach().clone() for t in (q, k)]
    actual = weighted_attention(q, k, v, w, backend=backend, mass_scaled=mass_scaled)
    scale = min(amount, 1.) if mass_scaled else 1.
    torch.testing.assert_close(actual, v.expand_as(q)*scale, atol=0, rtol=0)
    actual.backward(root)
    assert torch.count_nonzero(q.grad) == 0
    assert torch.count_nonzero(k.grad) == 0
    optimizer.step()
    for saved, current in zip(initial, (q, k)):
        torch.testing.assert_close(saved, current, atol=0, rtol=0)
    if not mass_scaled or amount > 1:
        assert torch.count_nonzero(w.grad) == 0


@pytest.mark.parametrize("backend", ["tiled", "auto"])
@pytest.mark.parametrize("mass_scaled,amount", [(False, .7), (False, 1e-30), (True, .7)])
def test_single_positive_key_after_a_zero_key_and_across_tiles(device, backend, mass_scaled, amount):
    rng = torch.Generator().manual_seed(844)
    q = torch.randn(1, 1, 65, 32, generator=rng) * .2
    k = torch.randn(1, 1, 257, 32, generator=rng) * .2
    v = torch.randn(1, 1, 257, 32, generator=rng)
    w = torch.zeros(1, 1, 65, 257)
    w[..., ::2, 256] = amount
    w[..., 1::2, 3] = amount
    mask = torch.ones_like(w, dtype=torch.bool)
    mask[..., 0] = False
    root = torch.randn(q.shape, generator=rng) * (1e-20 if amount < 1e-20 else .01)
    if amount < 1e-20:
        # Isolate the singleton/range identity from an ill-conditioned dot
        # product: the live values are zero and real zero-prior edges have
        # nonzero, one-coordinate right derivatives. The literal quotient
        # then also preserves the analytically zero live-prior derivative.
        v[..., (0, 3, 256), :] = 0
        root.zero_()
        root[..., 0] = 1e-20
    roots = [t.to(device).requires_grad_() for t in (q, k, v, w)]
    actual = weighted_attention(*roots, mask.to(device), backend, mass_scaled=mass_scaled)
    gradients = torch.autograd.grad(actual, roots, root.to(device))
    assert torch.count_nonzero(gradients[0]) == 0
    assert torch.count_nonzero(gradients[1]) == 0
    assert torch.count_nonzero(gradients[3][..., 0]) == 0
    if not mass_scaled:
        assert torch.count_nonzero(gradients[3][w.to(device) > 0]) == 0
    # Independent normalized quotient and ordinary CPU FP64 autograd. All
    # real zero weights remain variables, including those before the live key.
    refs = [t.double().requires_grad_() for t in (q, k, v, w)]
    rq, rk, rv, rw = refs
    terms = (rq @ rk.transpose(-2, -1) / 32**.5).exp() * rw * mask
    expected = terms @ rv / terms.sum(-1, keepdim=True)
    if mass_scaled:
        expected = expected * (rw * mask).sum(-1, keepdim=True).clamp(max=1)
    refgrads = torch.autograd.grad(expected, refs, root.double())
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=1e-4)
    for actual_grad, expected_grad in zip(gradients, refgrads):
        assert torch.isfinite(actual_grad).all()
        # Q/K have an exact independent zero identity above, while a literal
        # quotient's FP64 cancellation residual may be nonzero.
        torch.testing.assert_close(actual_grad.cpu(), expected_grad.float(), atol=1e-5, rtol=1e-4)


def test_constant_values_have_no_ordinary_prior_gradient(device):
    rng = torch.Generator().manual_seed(845)
    q, k = [torch.randn(1, 1, n, 32, generator=rng).to(device).requires_grad_() for n in (3, 5)]
    value = torch.randn(1, 1, 1, 32, generator=rng).to(device)
    v = value.expand(1, 1, 5, 32).clone().requires_grad_()
    w = torch.ones(5, device=device, requires_grad=True)
    # Constant V may acquire ordinary forward rounding for multiple keys;
    # compare the independent literal formula, without clipping small VJPs.
    root = torch.randn(q.shape, generator=rng).to(device)*.01
    out = weighted_attention(q, k, v, w, backend="tiled")
    grads = torch.autograd.grad(out, (q, k, w), root)
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(float(g.abs().max()) < 1e-7 for g in grads)
