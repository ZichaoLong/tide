"""Independent extreme-range Reception witnesses; optional device is explicit."""

import copy
import math
import os

import pytest
import torch

from tide.reception import Reception
from tide.scaled_arithmetic import restore_binary_scale
from tide.weighted_mean import stable_weighted_mean


@pytest.fixture
def device():
    name = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401; only an explicitly requested device.
    return torch.device(name)


def _oracle(mass, values, root):
    """Literal CPU FP64 quotient; deliberately no production helper."""
    m = mass.detach().cpu().double().requires_grad_()
    v = values.detach().cpu().double().requires_grad_()
    total = m.sum(-1)
    denominator = torch.where(total > 0, total, 1)
    result = (m[..., None] * v).sum(-2) / denominator[..., None]
    gradients = torch.autograd.grad(result, (m, v), root.detach().cpu().double())
    return result.detach(), gradients


def _compare(mass, values, root, *, atol=1e-6, rtol=1e-5):
    actual = stable_weighted_mean(mass, values)
    expected, reference = _oracle(mass, values, root)
    gradients = torch.autograd.grad(actual, (mass, values), root)
    # Compare to the independent value after its unavoidable public-dtype
    # storage rounding, including a genuinely unrepresentable half-subnormal.
    torch.testing.assert_close(actual.detach().cpu(), expected.to(actual.dtype), atol=atol, rtol=rtol)
    for got, want in zip(gradients, reference):
        torch.testing.assert_close(got.detach().cpu(), want.to(got.dtype), atol=atol, rtol=rtol)
    return actual, gradients


def _scale_boundary_inputs(dtype):
    precision, minimum, maximum = ((24, -149, 128) if dtype == torch.float32
                                   else (53, -1074, 1024))
    # Both signs of max finite, smallest subnormal, ties and adjacent values
    # around zero/min-subnormal and subnormal/normal boundaries. Several
    # fractions deliberately need normalization before exponent splitting.
    cases = [
        (1 - 2.**-precision, maximum),
        (.5, minimum + 1),
        (.5, minimum),
        (.5 + 2.**-precision, minimum),
        (.5 - 2.**-(precision + 1), minimum),
        (.75, minimum + 1),
        (1 - 2.**-(precision - 1), minimum + precision - 1),
        (1 - 2.**-precision, minimum + precision - 1),
        (.5, minimum + precision),
        (.25, maximum + 1),
        (2.**minimum, -minimum),
    ]
    signed = [(sign * fraction, exponent) for fraction, exponent in cases for sign in (1, -1)]
    fractions = torch.tensor([fraction for fraction, _ in signed], dtype=dtype)
    exponents = torch.tensor([exponent for _, exponent in signed], dtype=torch.int32)
    # Python math.ldexp uses independent host C arithmetic. For FP32 it has
    # enough precision to preserve the actual quantized operands until the
    # single final FP32 cast; for FP64 it supplies the final IEEE operation.
    expected = torch.tensor([math.ldexp(float(fraction), int(exponent))
                             for fraction, exponent in zip(fractions, exponents)], dtype=dtype)
    return fractions, exponents, expected


def test_fp32_binary_scale_extrema_and_rounding_boundaries(device):
    fraction, exponent, expected = _scale_boundary_inputs(torch.float32)
    actual = restore_binary_scale(fraction.to(device), exponent.to(device)).cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def test_fp64_binary_scale_extrema_and_rounding_boundaries_stay_on_cpu():
    fraction, exponent, expected = _scale_boundary_inputs(torch.float64)
    actual = restore_binary_scale(fraction, exponent)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def test_binary_scale_first_stage_is_normal_and_only_final_stage_rounds(monkeypatch):
    # Reproduce the faulty vendor strategy on CPU: materialize 2**exponent,
    # then multiply. The implementation must remain correct under it, and
    # must not round into subnormals before its final multiplication.
    calls = []

    def materialized_power(fraction, exponent):
        result = fraction * torch.pow(fraction.new_tensor(2.), exponent)
        calls.append((fraction.detach().clone(), exponent.detach().clone(), result.detach().clone()))
        return result

    monkeypatch.setattr(torch, "ldexp", materialized_power)
    for dtype in (torch.float32, torch.float64):
        fraction, exponent, expected = _scale_boundary_inputs(dtype)
        actual = restore_binary_scale(fraction, exponent)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.equal(torch.signbit(actual), torch.signbit(expected))
        first = calls[-2][2]
        assert bool(((first == 0) | (first.abs() >= torch.finfo(dtype).tiny)).all())
        assert bool(torch.isfinite(first).all())


def test_binary_scale_zero_and_true_overflow_have_distinct_results(device):
    fraction = torch.tensor([0., -0., .5, -.5, .5, -.5], device=device)
    exponent = torch.tensor([10000, -10000, 129, 129, -10000, -10000], device=device)
    actual = restore_binary_scale(fraction, exponent).cpu()
    expected = torch.tensor([0., -0., torch.inf, -torch.inf, 0., -0.])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def test_smallest_subnormal_single_edge_keeps_content_and_exact_zero_mass_vjp(device):
    mass = torch.tensor([2.**-149], device=device, requires_grad=True)
    values = torch.tensor([[.25, -3., 1e-20]], device=device, requires_grad=True)
    root = torch.tensor([.5, -.25, 1.], device=device)
    actual = stable_weighted_mean(mass, values)
    dm, dv = torch.autograd.grad(actual, (mass, values), root)
    torch.testing.assert_close(actual.cpu(), values.detach().cpu()[0], atol=0, rtol=0)
    assert torch.equal(dm, torch.zeros_like(dm))
    torch.testing.assert_close(dv.cpu(), root.cpu()[None], atol=0, rtol=0)


def test_reception_preserves_a_real_subnormal_receipt(device):
    module = Reception([{"ext"}], 8, "source_residual").to(device)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    content = torch.full((1, 1, 1, 1, 8), .25, device=device, requires_grad=True)
    mass = torch.full((1, 1, 1, 1), 2.**-149, device=device, requires_grad=True)
    total, result = module(content, mass, torch.zeros(1, 1, dtype=torch.long, device=device),
                           torch.zeros(1, dtype=torch.long, device=device))
    dz, dm = torch.autograd.grad(result.sum(), (content, mass))
    torch.testing.assert_close(total.cpu(), mass.detach().cpu().sum(-1), atol=0, rtol=0)
    torch.testing.assert_close(result.cpu(), torch.full((1, 1, 1, 8), .25), atol=0, rtol=0)
    assert torch.equal(dm, torch.zeros_like(dm))
    assert torch.equal(dz, torch.ones_like(dz))


def test_structural_padding_roots_are_removed_before_range_checks(device):
    mass = torch.tensor([2.**-149, 1.], device=device, requires_grad=True)
    values = torch.tensor([[.25], [-.75]], device=device, requires_grad=True)
    mask = torch.tensor([True, False], device=device)
    result = stable_weighted_mean(mass, values, mask)
    dm, dv = torch.autograd.grad(result.sum(), (mass, values))
    torch.testing.assert_close(result.cpu(), torch.tensor([.25]), atol=0, rtol=0)
    assert torch.equal(dm, torch.zeros_like(dm))
    torch.testing.assert_close(dv.cpu(), torch.tensor([[1.], [0.]]), atol=0, rtol=0)
    # A real zero-mass edge has a different domain. Its right derivative is
    # -1/min_subnormal here and must be diagnosed as genuinely out of range.
    real_mass = torch.tensor([2.**-149, 0.], device=device, requires_grad=True)
    with pytest.raises(ArithmeticError, match="weighted mean VJP.*dtype range"):
        stable_weighted_mean(real_mass, values).sum().backward()


@pytest.mark.parametrize("receipt,value,upstream", [
    (4.967183787530951e-38, 17.48714638, 0.),
    (8.636574620505972e-38, -31.55163383, 0.),
    (2.612801141005781e-38, 10.61332703, -2.802596928649634e-45),
])
def test_historical_quotient_overflow_has_finite_composite_vjp(device, receipt, value, upstream):
    mass = torch.tensor([receipt], device=device, requires_grad=True)
    values = torch.tensor([[value]], device=device, requires_grad=True)
    root = torch.tensor([upstream], device=device)
    # Historical DivBackward formed value/receipt before multiplying root.
    assert abs(value / receipt) > torch.finfo(torch.float32).max
    output = stable_weighted_mean(mass, values)
    dm, dv = torch.autograd.grad(output, (mass, values), root)
    torch.testing.assert_close(output.cpu(), values.detach().cpu()[0], atol=0, rtol=0)
    assert torch.equal(dm, torch.zeros_like(dm))
    torch.testing.assert_close(dv.cpu(), root.cpu()[None], atol=0, rtol=0)


@pytest.mark.parametrize("mass,values,root", [
    ([1e-30, 1e-30], [[1e-30], [-1e-30]], [1e-30]),
    ([1e30, 1e30], [[1e30], [-1e30]], [1e30]),
    ([1., 1.], [[1e30, 1e30], [-1e30, -1e30]], [1e30, -1e30]),
    ([0., 1.], [[3e38], [-3e38]], [1e-30]),
    ([2.**-149, 1., 1.], [[1.], [0.], [0.]], [1e30]),
])
def test_representable_extreme_composite_vjps_do_not_lose_intermediate_products(device, mass, values, root):
    m = torch.tensor(mass, device=device, requires_grad=True)
    v = torch.tensor(values, device=device, requires_grad=True)
    g = torch.tensor(root, device=device)
    # Zero absolute tolerance is intentional: these include tiny nonzero
    # gradients for which an ordinary 1e-6 absolute tolerance would hide loss.
    _compare(m, v, g, atol=0, rtol=1e-5)


def test_tiny_probability_times_large_content_is_preserved(device):
    mass = torch.tensor([2.**-149, 1., 1.], device=device, requires_grad=True)
    values = torch.tensor([[1e30], [0.], [0.]], device=device, requires_grad=True)
    root = torch.tensor([1.], device=device)
    _compare(mass, values, root, atol=0, rtol=1e-5)


def test_broadcast_sources_accumulate_the_independent_cpu_formula(device):
    generator = torch.Generator().manual_seed(421)
    mass = (.2 + torch.rand(1, 3, generator=generator)).to(device).requires_grad_()
    values = torch.randn(4, 3, 5, generator=generator).to(device).requires_grad_()
    root = torch.randn(4, 5, generator=generator).to(device)
    _compare(mass, values, root)


def test_broadcast_reduction_precedes_public_gradient_range_check(device):
    # Individual expanded mass roots exceed FP32, but each source is reused
    # by two opposite loss roots, giving the representable public VJP zero.
    mass = torch.tensor([[1e-30, 1e-30]], device=device, requires_grad=True)
    values = torch.tensor([[[1.], [-1.]], [[1.], [-1.]]], device=device, requires_grad=True)
    root = torch.tensor([[1e30], [-1e30]], device=device)
    output = stable_weighted_mean(mass, values)
    dm, dv = torch.autograd.grad(output, (mass, values), root)
    assert torch.equal(dm, torch.zeros_like(dm))
    torch.testing.assert_close(dv.cpu(), (root[:, None] * .5).expand_as(dv).cpu(), atol=0, rtol=0)


def test_all_zero_keeps_the_existing_denominator_one_mass_branch(device):
    mass = torch.zeros(2, 3, device=device, requires_grad=True)
    values = torch.tensor([[[1., 2.], [-3., 4.], [5., -6.]]], device=device, requires_grad=True)
    root = torch.tensor([[.25, -.5], [.5, .75]], device=device)
    actual, (dm, dv) = _compare(mass, values, root, atol=0, rtol=0)
    assert torch.equal(actual, torch.zeros_like(actual))
    assert torch.count_nonzero(dm) > 0
    assert torch.equal(dv, torch.zeros_like(dv))


def test_empty_edges_have_connected_zero_source_adjoints(device):
    mass = torch.empty(1, 0, device=device, requires_grad=True)
    values = torch.empty(2, 0, 4, device=device, requires_grad=True)
    result = stable_weighted_mean(mass, values)
    dm, dv = torch.autograd.grad(result.sum(), (mass, values))
    assert result.shape == (2, 4) and not result.any()
    assert dm.shape == mass.shape and dv.shape == values.shape


@pytest.mark.parametrize("source", ["mass", "values"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_original_sources_keep_version_guards_after_promotion(device, source, dtype):
    mass = torch.tensor([.25, .75], dtype=dtype, device=device, requires_grad=True)
    values = torch.tensor([[1., 2.], [-3., 4.]], dtype=dtype, device=device, requires_grad=True)
    result = stable_weighted_mean(mass, values)
    with torch.no_grad():
        (mass if source == "mass" else values).add_(.125)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        result.sum().backward()


def test_bf16_internal_mean_and_vjp_compute_in_fp32(device):
    mass = torch.tensor([2.**-100, 2.**-100], dtype=torch.bfloat16, device=device, requires_grad=True)
    values = torch.tensor([[.25, -2.], [.75, 4.]], dtype=torch.bfloat16, device=device, requires_grad=True)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        result = stable_weighted_mean(mass, values)
    assert result.dtype == torch.float32
    _compare(mass, values, torch.tensor([.25, -.5], device=device), atol=0, rtol=.008)


def test_true_unrepresentable_public_mass_gradient_raises(device):
    mass = torch.tensor([2.**-149, 2.**-149], device=device, requires_grad=True)
    values = torch.tensor([[1.], [-1.]], device=device, requires_grad=True)
    with pytest.raises(ArithmeticError, match="weighted mean VJP.*dtype range"):
        stable_weighted_mean(mass, values).sum().backward()


def test_an_unrequested_mass_adjoint_does_not_block_a_finite_value_vjp(device):
    mass = torch.tensor([2.**-149, 2.**-149], device=device)
    values = torch.tensor([[1.], [-1.]], device=device, requires_grad=True)
    gradient, = torch.autograd.grad(stable_weighted_mean(mass, values).sum(), (values,))
    torch.testing.assert_close(gradient.cpu(), torch.full((2, 1), .5), atol=0, rtol=0)


def test_zero_loss_root_does_not_create_nonfinite_local_jacobians(device):
    mass = torch.tensor([2.**-149, 2.**-149], device=device, requires_grad=True)
    values = torch.tensor([[1e30], [-1e30]], device=device, requires_grad=True)
    result = stable_weighted_mean(mass, values)
    gradients = torch.autograd.grad(result, (mass, values), torch.zeros_like(result))
    assert all(torch.equal(gradient, torch.zeros_like(gradient)) for gradient in gradients)


def test_true_unrepresentable_public_total_raises(device):
    mass = torch.full((2,), 3e38, device=device)
    values = torch.ones(2, 1, device=device)
    with pytest.raises(ArithmeticError, match="weighted mean forward.*dtype range"):
        stable_weighted_mean(mass, values)


def test_reception_masks_padding_but_keeps_real_zero_root_and_shared_updates(device):
    torch.manual_seed(726)
    module = Reception([{"ext"}] * 2, 8, "source_residual", groups=[0, 0], rank=2).to(device)
    reference = copy.deepcopy(module).cpu().double()
    optimizer = torch.optim.SGD(module.parameters(), lr=.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    generator = torch.Generator().manual_seed(288)
    owners = torch.tensor([1, 0, 1], device=device)
    classes = torch.zeros(3, 3, dtype=torch.long, device=device)
    mask = torch.tensor([[True, True, False]] * 3, device=device)
    for _ in range(2):
        z = torch.randn(3, 1, 1, 3, 8, generator=generator).to(device).requires_grad_()
        m = torch.tensor([[[[.5, 0., 1.]]]] * 3, device=device, requires_grad=True)
        zr, mr = z.detach().cpu().double().requires_grad_(), m.detach().cpu().double().requires_grad_()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        total, actual = module(z, m, classes, owners, mask)
        # Independent source-residual expression with literal owner matrices.
        p = dict(reference.named_parameters())
        rows = []
        for row, owner in enumerate(owners.cpu().tolist()):
            x = zr[row, 0, 0]
            normalized = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-5)
            a = p["a.weight"][0] + p["a.left"][owner] @ p["a.right"][owner]
            b = p["b.weight"][0] + p["b.left"][owner] @ p["b.right"][owner]
            hidden = normalized @ a.T + p["embedding.weight"][0]
            transformed = x + (hidden * torch.sigmoid(hidden)) @ b.T
            weights = mr[row, 0, 0] * mask[row].cpu()
            rows.append((weights[:, None] * transformed).sum(0) / weights.sum())
        expected = torch.stack(rows)[:, None, None]
        root = torch.randn(actual.shape, generator=generator).to(device)
        loss = (actual * root).sum() + .2 * total.sum()
        ref_loss = (expected * root.cpu().double()).sum() + .2 * (mr * mask.cpu()[:, None, None]).sum()
        loss.backward()
        ref_loss.backward()
        torch.testing.assert_close(actual.cpu().double(), expected, atol=1e-6, rtol=1e-5)
        for got, want in ((z.grad, zr.grad), (m.grad, mr.grad),
                          *((p.grad, q.grad) for p, q in zip(module.parameters(), reference.parameters()))):
            torch.testing.assert_close(got.cpu().double(), want, atol=1e-6, rtol=1e-5)
        assert not m.grad[..., 2].any()  # Padding is absent even with positive supplied mass.
        assert m.grad[..., 1].abs().sum() > 0  # The real zero-mass edge is a root.
        before = module.b.weight.detach().clone()
        optimizer.step()
        reference_optimizer.step()
        assert not torch.equal(before, module.b.weight)
        for got, want in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(got.detach().cpu().double(), want, atol=1e-6, rtol=1e-5)
