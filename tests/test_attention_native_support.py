"""Independent support identities and shared-root updates for the NPU adapter.

CPU runs use SDPA as a stand-in for the internal adapter; only TIDE_STABLE_DEVICE
NPU runs exercise public auto/explicit native dispatch and qualify that device.
"""
import os

import pytest
import torch

import tide.attention as attention


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
    return torch.device(name)


def call(q, k, v, w, mask, *, backend="auto", mass_scaled=False):
    if q.device.type == "npu":
        return attention.weighted_attention(q, k, v, w, mask, backend, mass_scaled=mass_scaled)
    shape = q.shape[:-1] + (k.shape[-2],)
    prior = q.new_ones((1,) * len(shape)) if w is None else w
    visible = torch.ones((1,) * len(shape), dtype=torch.bool, device=q.device) if mask is None else mask
    result = attention._support_safe_native_attention(q, k, v, prior, visible, "sdpa",
                                                      unit_prior=w is None, no_mask=mask is None)
    if mass_scaled:
        result = result.float() * torch.where(visible, prior.float(), 0).expand(shape).sum(-1, keepdim=True).clamp(max=1)
    return result


def literal(q, k, v, w, mask, mass_scaled=False):
    terms = (q @ k.transpose(-2, -1) / q.shape[-1] ** .5).exp() * w * mask
    denominator = terms.sum(-1, keepdim=True)
    out = (terms @ v) / torch.where(denominator > 0, denominator, 1)
    if mass_scaled:
        out = out * (w * mask).sum(-1, keepdim=True).clamp(max=1)
    return out


@pytest.mark.parametrize("backend", ["auto", "npu_fusion"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("keys", [1, 5])
def test_singleton_and_empty_rows_bypass_fusion_with_connected_zero_gradients(device, monkeypatch, backend, dtype, keys):
    rng = torch.Generator().manual_seed(9041)
    roots = [torch.randn(2, 2, n, 32, generator=rng).to(device, dtype).requires_grad_()
             for n in (4, keys, keys)]
    w = torch.ones(1, 1, 1, keys, device=device, requires_grad=True)
    mask = torch.zeros(1, 1, 4, keys, dtype=torch.bool, device=device)
    mask[..., :3, keys - 1] = True
    def forbidden(*args, **kwargs):
        raise AssertionError("singleton/empty-only call must not execute fusion")
    monkeypatch.setattr(attention, "_native_attention", forbidden)
    out = call(*roots, w, mask, backend=backend)
    expected = torch.cat((roots[2][..., -1:, :].expand(2, 2, 3, 32), torch.zeros_like(out[..., :1, :])), -2)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    out.sum().backward()
    for root in (*roots[:2], w):
        assert root.grad is not None
        assert torch.count_nonzero(root.grad) == 0
    expected_v = torch.zeros_like(roots[2])
    expected_v[..., -1, :] = 3
    torch.testing.assert_close(roots[2].grad, expected_v, atol=0, rtol=0)


@pytest.mark.parametrize("amount", [.5, 1., 2.])
def test_singleton_mass_keeps_amplitude_derivative_including_clamp_boundary(device, amount):
    rng = torch.Generator().manual_seed(9042)
    q, k, v = [torch.randn(1, 1, n, 32, generator=rng).to(device).requires_grad_() for n in (3, 5, 5)]
    w = torch.full((1, 1, 1, 5), amount, device=device, requires_grad=True)
    mask = torch.zeros(1, 1, 3, 5, dtype=torch.bool, device=device)
    mask[..., 2] = True
    root = torch.randn(q.shape, generator=rng).to(device) * .01
    out = call(q, k, v, w, mask, mass_scaled=True)
    grads = torch.autograd.grad(out, (q, k, v, w), root)
    assert torch.count_nonzero(grads[0]) == torch.count_nonzero(grads[1]) == 0
    want = torch.zeros_like(w)
    if amount <= 1:
        want[..., 2] = (root * v[..., 2:3, :]).sum()
    torch.testing.assert_close(grads[3], want, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mass_scaled", [False, True])
def test_mixed_support_shared_qkv_adam_matches_independent_fp64(device, mass_scaled):
    rng = torch.Generator().manual_seed(9043)
    initial = torch.randn(2, 1, 5, 32, generator=rng) * .2
    shared = torch.nn.Parameter(initial.to(device))
    w = torch.nn.Parameter(torch.tensor([[[[.2, .3, .4, .5, .6]]]], device=device))
    ref = torch.nn.Parameter(initial.double())
    rw = torch.nn.Parameter(w.detach().cpu().double())
    mask = torch.zeros(1, 1, 4, 5, dtype=torch.bool)
    mask[..., 0, 2] = True
    mask[..., 2, :2] = True
    mask[..., 3, :] = True
    root = torch.randn(2, 1, 4, 32, generator=rng) * .01
    opts = [torch.optim.AdamW(params, lr=.001, eps=1e-8, weight_decay=.01, foreach=False)
            for params in ((shared, w), (ref, rw))]
    for _ in range(3):
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        actual = call(shared[..., :4, :], shared, shared * 1.7, w, mask.to(device), mass_scaled=mass_scaled)
        expected = literal(ref[..., :4, :], ref, ref * 1.7, rw, mask, mass_scaled)
        torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=1e-4)
        actual.backward(root.to(device))
        expected.backward(root.double())
        for a, b in ((shared, ref), (w, rw)):
            torch.testing.assert_close(a.grad.cpu(), b.grad.float(), atol=1e-5, rtol=1e-4)
        for opt in opts:
            opt.step()
        for a, b in ((shared, ref), (w, rw)):
            torch.testing.assert_close(a.cpu(), b.float(), atol=1e-5, rtol=1e-4)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(opts[0].state[a][key].cpu(), opts[1].state[b][key].float(), atol=1e-5, rtol=1e-4)


def test_mixed_call_masks_singleton_cotangents_before_shared_native_reduction(device, monkeypatch):
    rng = torch.Generator().manual_seed(9044)
    q, k, v = [torch.randn(1, 1, n, 32, generator=rng).to(device).requires_grad_() for n in (3, 5, 5)]
    w = torch.ones(1, 1, 1, 5, device=device, requires_grad=True)
    mask = torch.ones(1, 1, 3, 5, dtype=torch.bool, device=device)
    mask[..., 0, :] = False
    mask[..., 0, 3] = True
    mask[..., 1, :] = False
    seen = []
    native = attention._native_attention
    def observed(*args, **kwargs):
        out = native(*args, **kwargs)
        out.register_hook(lambda grad: seen.append(grad.detach().cpu()))
        return out
    monkeypatch.setattr(attention, "_native_attention", observed)
    call(q, k, v, w, mask).sum().backward()
    assert len(seen) == 1
    assert torch.count_nonzero(seen[0][..., :2, :]) == 0
    assert torch.count_nonzero(seen[0][..., 2:, :]) > 0


@pytest.mark.parametrize("value_difference", [0., 1e-5])
def test_multiple_nearly_identical_values_do_not_manufacture_adam_updates(device, value_difference):
    rng = torch.Generator().manual_seed(9045)
    q, k = [torch.randn(1, 1, n, 32, generator=rng) * .2 for n in (3, 5)]
    value = torch.randn(1, 1, 1, 32, generator=rng)
    v = value.expand(1, 1, 5, 32).clone()
    v[..., 1, :] += value_difference
    w = torch.ones(1, 1, 1, 5)
    roots = [torch.nn.Parameter(t.to(device)) for t in (q, k, v, w)]
    refs = [torch.nn.Parameter(t.double()) for t in (q, k, v, w)]
    root = torch.randn(q.shape, generator=rng) * .01
    mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)
    opts = [torch.optim.Adam(params, lr=.001, eps=1e-8, foreach=False) for params in (roots, refs)]
    actual = call(*roots, mask.to(device))
    expected = literal(*refs, mask)
    actual.backward(root.to(device))
    expected.backward(root.double())
    for a, b in zip(roots, refs):
        torch.testing.assert_close(a.grad.cpu(), b.grad.float(), atol=1e-5, rtol=1e-4)
    for opt in opts:
        opt.step()
    for a, b in zip(roots, refs):
        torch.testing.assert_close(a.cpu(), b.float(), atol=1e-5, rtol=1e-4)


def test_unit_prior_single_key_never_forms_irrelevant_overflowing_logits(device, monkeypatch):
    q = torch.full((1, 1, 3, 32), 1e30, device=device, requires_grad=True)
    k = torch.full((1, 1, 1, 32), 1e30, device=device, requires_grad=True)
    v = torch.arange(32, dtype=torch.float32, device=device).reshape(1, 1, 1, 32).requires_grad_()
    def forbidden(*args, **kwargs):
        raise AssertionError("unit-prior singleton must not execute fusion")
    monkeypatch.setattr(attention, "_native_attention", forbidden)
    out = call(q, k, v, None, None)
    torch.testing.assert_close(out, v.expand_as(q), atol=0, rtol=0)
    out.sum().backward()
    assert torch.count_nonzero(q.grad) == torch.count_nonzero(k.grad) == 0
    torch.testing.assert_close(v.grad, torch.full_like(v, 3), atol=0, rtol=0)


@pytest.mark.parametrize("trainable_prior", [False, True])
def test_concentrated_two_key_distribution_retains_real_small_gradients(device, trainable_prior):
    q = torch.zeros(1, 1, 1, 32)
    q[..., 0] = 1
    k = torch.zeros(1, 1, 2, 32)
    k[..., 0, 0] = 20 * 32 ** .5
    v = torch.ones(1, 1, 2, 32)
    v[..., 1, :] = 2
    w = torch.ones(1, 1, 1, 2)
    roots = [torch.nn.Parameter(t.to(device)) for t in (q, k, v, w)]
    refs = [torch.nn.Parameter(t.double()) for t in (q, k, v, w)]
    actual = call(*roots[:3], roots[3] if trainable_prior else None, None)
    mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    expected = literal(*refs, mask)
    root = torch.zeros_like(q)
    root[..., 0] = 1
    actual.backward(root.to(device))
    expected.backward(root.double())
    assert torch.count_nonzero(roots[0].grad) > 0
    for params in (roots, refs):
        torch.optim.Adam(params[:3], lr=.001, eps=1e-8, foreach=False).step()
    for a, b in zip(roots[:3], refs[:3]):
        torch.testing.assert_close(a.cpu(), b.float(), atol=1e-5, rtol=1e-4)


def test_value_translation_does_not_introduce_overflow_for_finite_extreme_values(device):
    q = torch.zeros(1, 1, 1, 32, device=device)
    k = torch.zeros(1, 1, 2, 32, device=device)
    v = torch.full_like(k, 3e38)
    v[..., 1, :] = -3e38
    out = call(q, k, v, None, None)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=1e-4)
