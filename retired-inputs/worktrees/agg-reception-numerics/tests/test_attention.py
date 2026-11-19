"""Literal per-row exponential quotients independent of kernel dispatch."""

import sys
from types import SimpleNamespace

import pytest
import torch

from tide.attention import _native_attention, weighted_attention


def literal_attention(q, k, v, weights, mask):
    batches = []
    for batch in range(q.shape[0]):
        heads = []
        for head in range(q.shape[1]):
            rows = []
            for row in range(q.shape[2]):
                score = (q[batch, head, row] * k[batch, head]).sum(-1) / q.shape[-1] ** .5
                terms = torch.exp(score) * weights[batch] * mask[batch, head, row]
                denominator = terms.sum()
                quotient = (terms[:, None] * v[batch, head]).sum(0) / torch.where(denominator > 0, denominator, 1)
                rows.append(torch.where(denominator > 0, quotient, 0))
            heads.append(torch.stack(rows))
        batches.append(torch.stack(heads))
    return torch.stack(batches)


def fixture(queries=3):
    rng = torch.Generator().manual_seed(610)
    q = torch.randn(2, 2, queries, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 5, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 5, 3, generator=rng, dtype=torch.float64, requires_grad=True)
    w = torch.tensor([[.3, 0., .7, .1, .4], [0., 0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 2, queries, 5, dtype=torch.bool)
    mask[0, 0, 0] = False
    mask[0, 1, :, 3] = False
    return q, k, v, w, mask


@pytest.mark.parametrize("backend", ["math", "sdpa", "tiled", "auto"])
def test_values_and_every_input_vjp_match_literal_quotient(backend):
    q, k, v, w, mask = fixture()
    if backend == "sdpa":
        w = (w.detach() + .05).requires_grad_()
    actual = weighted_attention(q, k, v, w, mask, backend)
    reference = literal_attention(q, k, v, w, mask)
    torch.testing.assert_close(actual, reference, atol=2e-14, rtol=2e-13)
    loss_weights = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual) / 13
    gradients = torch.autograd.grad((actual * loss_weights).sum(), (q, k, v, w), retain_graph=True)
    expected = torch.autograd.grad((reference * loss_weights).sum(), (q, k, v, w))
    for got, want in zip(gradients, expected):
        torch.testing.assert_close(got, want, atol=3e-14, rtol=3e-13)
    assert gradients[-1][0, 1] != 0
    if backend != "sdpa":  # A zero prior keeps its right derivative; an empty row uses zero.
        assert torch.equal(gradients[-1][1], torch.zeros_like(w[1]))


@pytest.mark.parametrize("backend", ["math", "sdpa", "tiled"])
def test_smooth_gradcheck(backend):
    rng = torch.Generator().manual_seed(44)
    values = [torch.randn(shape, generator=rng, dtype=torch.float64, requires_grad=True)
              for shape in ((1, 1, 2, 2), (1, 1, 3, 2), (1, 1, 3, 2))]
    values.append(torch.tensor([[.2, .4, .8]], dtype=torch.float64, requires_grad=True))
    assert torch.autograd.gradcheck(lambda *xs: weighted_attention(*xs, backend=backend), tuple(values),
                                   atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("backend", ["sdpa", "tiled"])
def test_query_tiles_and_query_dependent_weight_gradients(backend):
    q, k, v, _, mask = fixture(queries=67)
    rng = torch.Generator().manual_seed(904)
    w = torch.rand(2, 1, 67, 5, generator=rng, dtype=torch.float64, requires_grad=True)
    result = weighted_attention(q, k, v, w, mask, backend)
    expected = weighted_attention(q, k, v, w, mask, "math")
    torch.testing.assert_close(result, expected, atol=2e-14, rtol=2e-13)
    got = torch.autograd.grad(result.square().sum(), (q, k, v, w), retain_graph=True)
    want = torch.autograd.grad(expected.square().sum(), (q, k, v, w))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=3e-13, rtol=3e-12)


@pytest.mark.parametrize("backend", ["math", "sdpa", "tiled"])
@pytest.mark.parametrize("queries,keys", [(0, 3), (2, 0), (0, 0)])
def test_empty_dimensions_return_connected_zeros(backend, queries, keys):
    tensors = [torch.empty(shape, dtype=torch.float64, requires_grad=True)
               for shape in ((1, 2, queries, 4), (1, 2, keys, 4), (1, 2, keys, 3), (1, keys))]
    # Empty tensors may share an expression with nonempty tensors: use finite data.
    for value in tensors:
        with torch.no_grad():
            value.zero_()
    result = weighted_attention(*tensors, backend=backend)
    assert result.shape == (1, 2, queries, 3)
    assert torch.equal(result, torch.zeros_like(result))
    gradients = torch.autograd.grad(result.sum(), tensors)
    assert all(torch.equal(g, torch.zeros_like(g)) for g in gradients)


@pytest.mark.parametrize("backend", ["math", "sdpa", "tiled"])
def test_noncontiguous_broadcast_and_repeated_roots(backend):
    rng = torch.Generator().manual_seed(774)
    shared = torch.randn(2, 3, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    qk = shared.unsqueeze(1)
    v = shared[..., ::2].unsqueeze(1)
    w = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True], [True, True, False], [True, True, True]])
    actual = weighted_attention(qk, qk, v, w, mask, backend)
    score = qk @ qk.transpose(-2, -1) / 2
    ref = torch.softmax(score.masked_fill(~mask, -torch.inf), -1) @ v
    torch.testing.assert_close(actual, ref)
    got = torch.autograd.grad(actual.square().sum(), (shared, w), retain_graph=True)
    want = torch.autograd.grad(ref.square().sum(), shared)[0]
    torch.testing.assert_close(got[0], want, atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(got[1], torch.zeros_like(w), atol=1e-13, rtol=0)


@pytest.mark.parametrize("backend", ["math", "sdpa", "tiled"])
def test_tiny_positive_weight_and_common_logit_offset(backend):
    q = torch.tensor([[[[100.]]]], dtype=torch.float64)
    k = torch.tensor([[[[1.], [1.]]]], dtype=torch.float64)
    v = torch.tensor([[[[2.], [6.]]]], dtype=torch.float64)
    weights = torch.tensor([[1e-250, 3e-250]], dtype=torch.float64, requires_grad=True)
    out = weighted_attention(q, k, v, weights, backend=backend)
    torch.testing.assert_close(out, torch.tensor([[[[5.]]]], dtype=torch.float64), atol=2e-13, rtol=1e-13)
    grad = torch.autograd.grad(out.sum(), weights)[0]
    expected = torch.tensor([[-7.5e249, 2.5e249]], dtype=torch.float64)
    torch.testing.assert_close(grad, expected, atol=0, rtol=1e-12)


def test_explicit_unavailable_backend_and_invalid_inputs_fail():
    q, k, v, w, mask = fixture()
    with pytest.raises(ValueError, match="NPU"):
        weighted_attention(q, k, v, w, mask, "npu_fusion")
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_attention(q, k, v, -torch.ones_like(w))
    with pytest.raises(ValueError, match="boolean"):
        weighted_attention(q, k, v, w, mask.to(q.dtype))
    with pytest.raises(ValueError, match="right derivative"):
        weighted_attention(q, k, v, w, mask, "sdpa")


def test_saved_input_version_guard():
    q, k, v, w, mask = fixture()
    result = weighted_attention(q, k, v, w, mask, "tiled")
    with torch.no_grad():
        k.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()


def test_masked_zero_priors_keep_native_sdpa_available():
    q, k, v, w, mask = fixture()
    mask[..., 1] = False
    mask[1] = False
    out = weighted_attention(q, k, v, w, mask, "sdpa")
    reference = literal_attention(q, k, v, w, mask)
    torch.testing.assert_close(out, reference)
    got = torch.autograd.grad(out.square().sum(), (q, k, v, w), retain_graph=True)
    want = torch.autograd.grad(reference.square().sum(), (q, k, v, w))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_online_key_tiles_with_masked_first_block(dtype):
    rng = torch.Generator().manual_seed(145)
    q, k, v = [torch.randn(shape, generator=rng, dtype=dtype, requires_grad=True)
               for shape in ((1, 2, 67, 4), (1, 2, 517, 4), (1, 2, 517, 3))]
    w = torch.rand(1, 517, generator=rng, dtype=dtype, requires_grad=True)
    mask = torch.ones(1, 1, 67, 517, dtype=torch.bool)
    mask[..., :256] = False
    mask[..., 3, :] = False
    output = weighted_attention(q, k, v, w, mask, "tiled")
    reference = weighted_attention(q, k, v, w, mask, "math")
    atol, rtol = (2e-13, 2e-12) if dtype == torch.float64 else (2e-6, 3e-5)
    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
    actual = torch.autograd.grad(output.square().sum(), (q, k, v, w), retain_graph=True)
    expected = torch.autograd.grad(reference.square().sum(), (q, k, v, w))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize("trainable,unit", [(True, False), (False, False), (False, True)])
def test_npu_adapter_algebra_and_compact_arguments_without_device(monkeypatch, trainable, unit):
    """A CPU stand-in checks adapter algebra/gradients, never NPU capability."""
    seen = {}

    def fake_fusion(q, k, v, heads, layout, **kwargs):
        seen.update(kwargs, q_shape=q.shape, k_shape=k.shape)
        assert layout == "BNSD" and heads == 2
        bias = kwargs["pse"]
        if bias is None and kwargs["pse_type"] != 1:
            raise RuntimeError("PseInput is nullptr while pseType is not default")
        if bias is not None:
            assert kwargs["pse_type"] == 0
        blocked = kwargs["atten_mask"]
        if blocked is not None:
            bias = torch.zeros_like(blocked, dtype=q.dtype) if bias is None else bias
            bias = torch.where(blocked, -torch.inf, bias)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=kwargs["scale"])
        return (out,)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention_v2=fake_fusion))
    q, k, _, _, _ = fixture()
    v = k.detach().clone().requires_grad_()
    w = torch.tensor([[[[.2, .4, .6, .8, .9]]], [[[.8, .6, .3, .2, .4]]]],
                     dtype=torch.float64, requires_grad=trainable)
    if unit:
        w = torch.ones_like(w)
    mask = torch.ones(2, 1, 3, 5, dtype=torch.bool)
    mask[:, :, 0, 1] = False
    out = _native_attention(q, k, v, w, mask, "npu_fusion")
    reference = weighted_attention(q, k, v, w, mask, "math")
    torch.testing.assert_close(out, reference, atol=2e-14, rtol=2e-13)
    roots = (q, k, v, w) if trainable else (q, k, v)
    actual = torch.autograd.grad(out.square().sum(), roots, retain_graph=True)
    expected = torch.autograd.grad(reference.square().sum(), roots)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-13, rtol=2e-12)
    assert seen["atten_mask"].shape == (2, 1, 3, 5)
    assert seen["scale"] == .5  # Original D=4 even after padding.
    if trainable:
        assert seen["pse"] is None and seen["q_shape"][-1] == 16
    elif unit:
        assert seen["pse"] is None and seen["q_shape"][-1] == 4
    else:
        assert seen["pse"].shape == (2, 2, 1, 5)
