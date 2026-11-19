"""Small independent baseline formulas and native-attention path checks."""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.baselines import SSDBaseline, TransformerBaseline, parameter_count


def rms(x, layer):
    if isinstance(layer, nn.Identity):
        return x
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + layer.eps) * layer.weight


def affine(x, layer):
    return x @ layer.weight.T


def feedforward(x, layer):
    gate, up = affine(x, layer.gate_up).chunk(2, -1)
    return affine(F.silu(gate)*up, layer.down)


def transformer_formula(model, x):
    """Dense whole-sequence attention, independent of production tile layout."""
    batch, length, d = x.shape
    positions = torch.arange(length)
    head_dim = model.head_dim
    frequencies = x.new_tensor([10000**(-2*i/head_dim) for i in range(head_dim//2)])
    phase = positions.to(x.dtype)[:, None] * frequencies
    mask = ((positions[:, None] - positions[None, :]) >= 0)
    mask = mask & ((positions[:, None] - positions[None, :]) < model.window)
    for block in model.blocks:
        parts = affine(rms(x, block.attention_norm), block.qkv).chunk(3, -1)
        q, k, v = [part.reshape(batch, length, d//head_dim, head_dim).permute(0, 2, 1, 3) for part in parts]
        rotated = []
        for tensor in (q, k):
            pairs = tensor.reshape(batch, d//head_dim, length, head_dim//2, 2)
            real = pairs[..., 0]*phase.cos() - pairs[..., 1]*phase.sin()
            imag = pairs[..., 0]*phase.sin() + pairs[..., 1]*phase.cos()
            rotated.append(torch.stack((real, imag), -1).flatten(-2))
        logits = (rotated[0] @ rotated[1].transpose(-1, -2)) / math.sqrt(head_dim)
        attention = logits.masked_fill(~mask, -torch.inf).softmax(-1) @ v
        x = x + affine(attention.transpose(1, 2).reshape(batch, length, d), block.output)
        x = x + feedforward(rms(x, block.ffn_norm), block.ffn)
    return rms(x, model.final_norm)


def ssd_formula(model, x):
    """Literal convolution and one matrix recurrence per time position."""
    batch, length, _ = x.shape
    for block in model.blocks:
        mixer = block.mixer
        projection = affine(rms(x, block.norm), mixer.input)
        conv_width = mixer.inner + 2*mixer.groups*mixer.state_dim
        gate, projected, dt_raw = projection.split((mixer.inner, conv_width, mixer.heads), -1)
        matrix = x.new_zeros(batch, mixer.heads, mixer.head_dim, mixer.state_dim)
        outputs = []
        for time in range(length):
            convolved = mixer.conv.bias.expand(batch, conv_width)
            for lag in range(min(time+1, mixer.conv.kernel_size[0])):
                convolved = convolved + projected[:, time-lag] * mixer.conv.weight[:, 0, -lag-1]
            values, B, C = F.silu(convolved).split((mixer.inner, mixer.groups*mixer.state_dim,
                                                   mixer.groups*mixer.state_dim), -1)
            values = values.reshape(batch, mixer.heads, mixer.head_dim)
            group_index = torch.arange(mixer.heads) // (mixer.heads//mixer.groups)
            B = B.reshape(batch, mixer.groups, mixer.state_dim)[:, group_index]
            C = C.reshape(batch, mixer.groups, mixer.state_dim)[:, group_index]
            dt = F.softplus(dt_raw[:, time] + mixer.dt_bias)
            decay = torch.exp(-mixer.a_log.exp()*dt)
            matrix = decay[..., None, None]*matrix + dt[..., None, None]*values[..., None]*B[..., None, :]
            read = (matrix * C[..., None, :]).sum(-1)
            y = (read + mixer.skip[None, :, None]*values).reshape(batch, mixer.inner)
            outputs.append(affine(rms(y*F.silu(gate[:, time]), mixer.output_norm), mixer.output))
        x = x + torch.stack(outputs, 1)
        if block.ffn is not None:
            x = x + feedforward(rms(x, block.ffn_norm), block.ffn)
    return rms(x, model.final_norm)


@pytest.mark.parametrize("kind", ["transformer", "ssd", "ssd_ffn"])
def test_baseline_values_and_all_parameter_input_gradients(kind):
    torch.manual_seed(490)
    if kind == "transformer":
        model, formula = TransformerBaseline(64, 2, window=4, chunk=3).double(), transformer_formula
    else:
        model = SSDBaseline(32, 2, state_dim=7, groups=2 if kind == "ssd_ffn" else 1,
                            ffn=kind == "ssd_ffn", chunk=3).double()
        formula = ssd_formula
    x = torch.randn(2, 9, model.d, dtype=torch.float64, requires_grad=True)
    actual, expected = model(x), formula(model, x)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    root = torch.sin(torch.arange(actual.numel(), dtype=x.dtype)).reshape_as(actual)
    inputs = [x, *model.parameters()]
    got = torch.autograd.grad(actual, inputs, root, retain_graph=True)
    want = torch.autograd.grad(expected, inputs, root)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=3e-11, rtol=3e-11)


def test_transformer_uses_native_unit_prior_and_reuses_bounded_window_masks(monkeypatch):
    original = F.scaled_dot_product_attention
    calls = []

    def sdpa(q, k, v, **kwargs):
        mask = kwargs["attn_mask"]
        calls.append((q.shape, k.shape, mask.data_ptr()))
        assert mask.dtype == torch.bool
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", sdpa)
    model = TransformerBaseline(32, 2, window=4, chunk=3)
    x = torch.randn(3, 10, 32)
    first = model(x)
    torch.testing.assert_close(model(x), first)
    assert len(calls) == 16
    assert all(q[:2] == (3, 1) and q[-2] <= 3 and k[-2] <= 6 for q, k, _ in calls)
    pointers = [row[2] for row in calls]
    assert pointers[:4] == pointers[4:8] == pointers[8:12] == pointers[12:]
    cache = model._layout_cache
    assert sum(tile[-1].numel() for tile in cache[1][2]) < x.shape[1]**2


def test_baseline_causality_and_declared_window():
    torch.manual_seed(72)
    x = torch.randn(2, 12, 32)
    altered = x.clone()
    altered[:, 8:] += torch.randn_like(altered[:, 8:])
    for model in (TransformerBaseline(32, 2, window=4, chunk=5), SSDBaseline(32, 2)):
        torch.testing.assert_close(model(x)[:, :8], model(altered)[:, :8], atol=1e-6, rtol=1e-6)
    single = TransformerBaseline(32, 1, window=4, chunk=5)
    altered = x.clone()
    altered[:, :4] += 10*torch.randn_like(altered[:, :4])
    torch.testing.assert_close(single(x)[:, 7:], single(altered)[:, 7:], atol=1e-6, rtol=1e-6)


def test_reported_counts_include_norms_and_only_enabled_ffn_parameters():
    d, layers = 32, 3
    transformer = TransformerBaseline(d, layers)
    plain = SSDBaseline(d, layers)
    with_ffn = SSDBaseline(d, layers, ffn=True)
    assert parameter_count(transformer) == layers*(16*d*d + 2*d) + d
    assert parameter_count(with_ffn) - parameter_count(plain) == layers*(12*d*d + d)
    for model in (transformer, plain, with_ffn):
        assert model.parameter_count == sum(p.numel() for p in model.parameters())
        assert model.configuration()["parameter_count"] == model.parameter_count
        assert model.configuration()["d"] == d and model.configuration()["layers"] == layers
        assert model(torch.empty(2, 0, d)).shape == (2, 0, d)
    assert plain.configuration()["ffn_width"] == 0
    assert with_ffn.configuration()["ffn_width"] == 4*d
    assert SSDBaseline(d, 1, chunk=128).configuration()["chunk"] == 32


@pytest.mark.parametrize("constructor,kwargs", [
    (TransformerBaseline, {"d": 33, "layers": 1}),
    (TransformerBaseline, {"d": 32, "layers": 1, "window": 0}),
    (SSDBaseline, {"d": 32, "layers": 1, "groups": 3}),
    (SSDBaseline, {"d": 32, "layers": 0}),
])
def test_invalid_baseline_shape_fails(constructor, kwargs):
    with pytest.raises(ValueError):
        constructor(**kwargs)


@pytest.mark.parametrize("batch,head_dim,window", [(1, 32, 4), (3, 16, 5), (2, 64, 1)])
@pytest.mark.parametrize("contiguous_qkv", [False, True])
def test_transformer_incremental_cache_crosses_window_with_full_gradients(batch, head_dim, window, contiguous_qkv):
    torch.manual_seed(752)
    model = TransformerBaseline(64, 2, window=window, chunk=3,
                                head_dim=head_dim, ffn_width=152, contiguous_qkv=contiguous_qkv).double()
    x = torch.randn(batch, 13, 64, dtype=torch.float64, requires_grad=True)
    # Independent dense formula checks the values and derivatives, not a second
    # application of the incremental implementation's schedule.
    expected = transformer_formula(model, x)
    whole, whole_state = model(x, return_state=True)
    state, outputs = None, []
    projected_lengths = []
    hook = model.blocks[0].qkv.register_forward_pre_hook(
        lambda module, args: projected_lengths.append(args[0].shape[1]))
    for start, stop in ((0, 3), (3, 8), (8, 9), (9, 10), (10, 13)):
        value, state = model(x[:, start:stop], state, return_state=True)
        outputs.append(value)
    hook.remove()
    assert projected_lengths == [3, 5, 1, 1, 3]  # No full-prefix recomputation.
    continued = torch.cat(outputs, 1)
    for value in (whole, continued):
        torch.testing.assert_close(value, expected, atol=3e-12, rtol=3e-12)
    assert state.position == whole_state.position == 13
    assert all(t.shape[-2] == window-1 for t in (*state.keys, *state.values))
    for a, b in zip((*state.keys, *state.values), (*whole_state.keys, *whole_state.values)):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
        assert a.untyped_storage().nbytes() == a.numel()*a.element_size()
    root = torch.cos(torch.arange(expected.numel(), dtype=x.dtype)).reshape_as(expected)
    inputs = (x, *model.parameters())
    actual_grad = torch.autograd.grad(continued, inputs, root, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, inputs, root)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=6e-11, rtol=6e-11)


@pytest.mark.parametrize("candidate", [dict(chunk=256), dict(contiguous_qkv=True)])
def test_bounded_transformer_candidates_cross_tile_boundary_against_dense_formula(candidate):
    torch.manual_seed(191)
    model = TransformerBaseline(32, 1, window=17, **candidate).double()
    x = torch.randn(1, 259, 32, dtype=torch.float64, requires_grad=True)
    actual, expected = model(x), transformer_formula(model, x)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    root = torch.sin(torch.arange(actual.numel(), dtype=x.dtype)).reshape_as(actual)
    inputs = [x, *model.parameters()]
    got = torch.autograd.grad(actual, inputs, root, retain_graph=True)
    want = torch.autograd.grad(expected, inputs, root)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=6e-11, rtol=6e-11)


def test_transformer_cache_ownership_and_shape_validation():
    model = TransformerBaseline(32, 2, window=4)
    x = torch.randn(2, 3, 32)
    _, state = model(x, return_state=True)
    originals = [t.clone() for t in (*state.keys, *state.values)]
    _, following = model(x[:, :1], state, return_state=True)
    assert following.position == 4 and state.position == 3
    for a, b in zip((*state.keys, *state.values), originals):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError, match="cache shape"):
        model(x[:1, :1], state)
    other = TransformerBaseline(32, 1, window=4)
    with pytest.raises(ValueError, match="layer count"):
        other(x, state)


def test_transformer_adjustable_width_count_and_bfloat16_cache():
    model = TransformerBaseline(64, 2, head_dim=16, ffn_width=136, window=4)
    assert parameter_count(model) == 2*(4*64**2+3*64*136+2*64)+64
    x = torch.randn(2, 7, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        whole = model(x)
        first, state = model(x[:, :3], return_state=True)
        second, state = model(x[:, 3:], state, return_state=True)
    assert state.keys[0].dtype == state.values[0].dtype == torch.bfloat16
    torch.testing.assert_close(torch.cat((first, second), 1), whole, atol=.04, rtol=.04)
    torch.cat((first, second), 1).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("ffn", [False, True])
def test_ssd_bfloat16_matrix_arithmetic_with_float32_recurrence(ffn):
    from tide.precision import ArithmeticAudit, precision_context
    torch.manual_seed(381)
    model = SSDBaseline(32, 2, ffn=ffn, state_dim=7)
    x = torch.randn(2, 11, 32, requires_grad=True)
    expected = model(x)
    audit = ArithmeticAudit()
    with audit, precision_context("cpu", "bf16"):
        actual = model(x)
    torch.testing.assert_close(actual, expected, atol=.035, rtol=.035)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any("cpu:bfloat16" in row["outputs"] for row in audit.report())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_transformer_empty_cache_can_continue_under_requested_arithmetic(dtype):
    model = TransformerBaseline(32, 1)
    with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
        empty, state = model(torch.empty(2, 0, 32), return_state=True)
        actual, state = model(torch.randn(2, 1, 32), state, return_state=True)
    assert empty.shape == (2, 0, 32) and actual.shape == (2, 1, 32)
    assert state.position == 1 and state.keys[0].dtype == dtype
