"""Independent finite-range witnesses for allocation's composite first VJP.

FP64 references stay on CPU. TIDE_STABLE_DEVICE optionally selects an admitted
device for the production FP32 calls; these are mechanism, not learning, tests.
"""

import math
import os

import pytest
import torch

from tide.allocation import allocate, pool


DEVICE = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
if DEVICE.startswith("npu"):
    import torch_npu  # noqa: F401


def fp32(values, *, grad=False):
    return torch.tensor(values, device=DEVICE, dtype=torch.float32, requires_grad=grad)


def pool_oracle(scores, mu, root):
    """Direct real formula in a wider diagnostic range, independent of pool."""
    s = scores.detach().cpu().double().requires_grad_()
    m = mu.detach().cpu().double().requires_grad_()
    mass = m.sum(-1, keepdim=True)
    numerator = m * torch.exp(s - s.amax(-1, keepdim=True))
    probabilities = numerator / numerator.sum(-1, keepdim=True)
    output = torch.where(mass >= 1, torch.ones_like(mass), mass) * probabilities
    gradients = torch.autograd.grad(output, (s, m), root.detach().cpu().double())
    return output, gradients


@pytest.mark.parametrize("mass,root", [(1e-40, 1.), (1e-30, 1e-30), (2.**-149, 1.)])
def test_equal_score_pool_is_identity_in_mu_below_unit_total(mass, root):
    scores = fp32([0., 0.], grad=True)
    mu = fp32([mass, mass], grad=True)
    output = pool(scores, mu)
    torch.testing.assert_close(output, mu, rtol=0, atol=0)
    ds, dm = torch.autograd.grad(output, (scores, mu), fp32([root, 0.]))
    torch.testing.assert_close(dm, fp32([root, 0.]), rtol=1e-6, atol=0)
    # A genuinely unrepresentable score derivative may round to zero.
    expected = (mu.detach().cpu().double() * torch.tensor([root / 2, -root / 2])).float()
    torch.testing.assert_close(ds.cpu(), expected, rtol=1e-6, atol=0)


def test_pool_weak_individual_prior_with_unit_regional_mass():
    scores = fp32([90., 0.], grad=True)
    mu = fp32([math.exp(-90), 1.], grad=True)
    root = fp32([1e-30, 0.])
    output = pool(scores, mu)
    expected, gradients = pool_oracle(scores, mu, root)
    actual = torch.autograd.grad(output, (scores, mu), root)
    torch.testing.assert_close(output.cpu().double(), expected, rtol=1e-6, atol=0)
    for got, want in zip(actual, gradients):
        torch.testing.assert_close(got.cpu().double(), want, rtol=1e-5, atol=0)
    assert actual[1][0] > 1e8


def test_pool_mean_below_storage_range_still_contributes_to_mu_vjp():
    scores = fp32([90., 0.], grad=True)
    mu = fp32([math.exp(-90), 1.], grad=True)
    root = fp32([0., 2.**-149])
    output = pool(scores, mu)
    _, reference = pool_oracle(scores, mu, root)
    actual = torch.autograd.grad(output, (scores, mu), root)
    # Its weighted cotangent mean is about half the smallest FP32 subnormal;
    # a weak-prior reciprocal makes the first mu derivative representable.
    torch.testing.assert_close(actual[1][0].cpu().double(), reference[1][0], rtol=1e-5, atol=0)
    assert actual[1][0] < 0


def test_pool_vjp_uses_unrounded_composite_before_subnormal_sender_storage():
    scores = fp32([math.log(1.5), 0.], grad=True)
    mu = fp32([2.**-149, 2.**-149], grad=True)
    root = fp32([1., 0.])
    output = pool(scores, mu)
    _, gradients = pool_oracle(scores, mu, root)
    _, got = torch.autograd.grad(output, (scores, mu), root)
    torch.testing.assert_close(got.cpu().double(), gradients[1], rtol=1e-6, atol=0)


@pytest.mark.parametrize("mu,expected", [([.4, .6], [.6, -.4]), ([.2, .3], [1., 0.])])
def test_pool_unit_mass_tie_has_no_total_mass_derivative(mu, expected):
    mass = fp32(mu, grad=True)
    output = pool(fp32([0., 0.]), mass)
    got, = torch.autograd.grad(output, (mass,), fp32([1., 0.]))
    torch.testing.assert_close(got, fp32(expected), rtol=1e-6, atol=1e-7)


def test_pool_zero_prior_right_derivative_and_empty_row():
    for mu, expected in [([0., 1e-40], [1., 0.]), ([0., 0.], [0., 0.])]:
        scores, mass = fp32([0., 0.], grad=True), fp32(mu, grad=True)
        output = pool(scores, mass)
        ds, dm = torch.autograd.grad(output, (scores, mass), fp32([1., 0.]))
        torch.testing.assert_close(dm, fp32(expected), rtol=1e-6, atol=0)
        torch.testing.assert_close(ds, torch.zeros_like(ds), rtol=0, atol=0)


def test_pool_broadcast_roots_repeated_vjp_and_version_guard():
    scores = fp32([[0., 0.], [0., 0.], [0., 0.]], grad=True)
    mass = fp32(1e-30, grad=True)
    output = pool(scores, mass)
    roots = fp32([[1e-30, 0.], [0., 2e-30], [3e-30, 0.]])
    first = torch.autograd.grad(output, (scores, mass), roots, retain_graph=True)
    second = torch.autograd.grad(output, (scores, mass), roots, retain_graph=True)
    torch.testing.assert_close(first[1], fp32(6e-30), rtol=1e-6, atol=0)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        mass.add_(1e-30)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def test_pool_keeps_true_sender_underflow_and_public_vjp_overflow_explicit():
    with pytest.raises(ArithmeticError, match="positive sender"):
        pool(fp32([-120., 0.]), fp32([1., 1.]))
    # Rounding probability to the minimum subnormal first would incorrectly
    # retain a sender: .6 * round(.7 * 2^-149) rounds up, while .42 * 2^-149
    # must round to zero and keep the explicit strict failure.
    with pytest.raises(ArithmeticError, match="positive sender"):
        pool(fp32([math.log(.7 * 2.**-149), 0.]), fp32([.3, .3]))
    scores, mu = fp32([100., 0.], grad=True), fp32([0., 1.], grad=True)
    output = pool(scores, mu)
    with pytest.raises(ArithmeticError, match="VJP exceeds"):
        torch.autograd.grad(output, (scores, mu), fp32([1., 0.]))


def test_shannon_reconstruction_scales_before_probability_underflow():
    scores = fp32([2.] * 63 + [-98.], grad=True)
    ones = torch.ones_like(scores)
    output = allocate(scores, ones, ones, 0., ones, 32., "shannon")
    # The probability alone is below FP32 range; its 32x output is not.
    weak = 32 * math.exp(-100) / (63 + math.exp(-100))
    assert output[-1] > 0
    torch.testing.assert_close(output[-1], fp32(weak), rtol=0, atol=0)
    reference = torch.tensor([32 / 63] * 63 + [weak], dtype=torch.float64)
    torch.testing.assert_close(output.cpu().double(), reference, rtol=2e-6, atol=2.**-149)


@pytest.mark.parametrize("unit", [1e-30, 2.**-149])
@pytest.mark.parametrize("count", [1, 2])
def test_shannon_coupling_keeps_weak_slope_weighted_average(unit, count):
    scores = fp32([-105., -105.], grad=True)
    w = fp32([count * unit, count * unit], grad=True)
    mu = fp32([2 * count * unit, 2 * count * unit], grad=True)
    output = allocate(scores, w, mu, 0., mu, 1., "shannon")
    root = fp32([.25, 0.])
    ds, dw, dm = torch.autograd.grad(output, (scores, w, mu), root)
    torch.testing.assert_close(output, w, rtol=1e-6, atol=0)
    torch.testing.assert_close(dw, fp32([.125, .125]), rtol=1e-6, atol=0)
    expected_score = (w.detach().cpu().double() * torch.tensor([.125, -.125])).float()
    torch.testing.assert_close(ds.cpu(), expected_score, rtol=1e-6, atol=0)
    torch.testing.assert_close(dm, torch.zeros_like(dm), rtol=0, atol=0)


def test_shannon_score_vjp_combines_slope_before_large_centered_difference():
    scores = fp32([2 + math.log(.01), 2 + math.log(.99)], grad=True)
    output = allocate(scores, fp32([.5, .5]), fp32([1., 1.]), 0., fp32([1., 1.]), 1., "shannon")
    root = fp32([3e38, -3e38])
    got, = torch.autograd.grad(output, (scores,), root)
    probability = torch.softmax(scores.detach().cpu().double(), -1)
    g = root.cpu().double()
    want = probability * (g - (probability * g).sum())
    torch.testing.assert_close(got.cpu().double(), want, rtol=2e-5, atol=0)


def test_quad_average_does_not_overflow_before_dividing_by_interior_count():
    scores, budget = fp32([3.] * 4, grad=True), fp32(1., grad=True)
    output = allocate(scores, fp32([.25] * 4), fp32([1.] * 4), 0., fp32([1.] * 4), budget)
    ds, dk = torch.autograd.grad(output, (scores, budget), fp32([3e38] * 4))
    torch.testing.assert_close(ds, torch.zeros_like(ds), rtol=0, atol=0)
    torch.testing.assert_close(dk, fp32(3e38), rtol=1e-6, atol=0)


@pytest.mark.parametrize("kind", ["quad", "shannon"])
def test_shared_score_is_reduced_before_expanded_adjoint_range_check(kind):
    # Each expanded first score adjoint would be 4.5e38 for quad, but the
    # actual source changes all scores together. Its fixed-total VJP is zero.
    scores = fp32([3.], grad=True)
    output = allocate(scores, fp32([.25] * 4), fp32([1.] * 4), 0., fp32([1.] * 4), 1., kind)
    got, = torch.autograd.grad(output, (scores,), fp32([3e38, -3e38, -3e38, -3e38]), retain_graph=True)
    torch.testing.assert_close(got, fp32([0.]), rtol=0, atol=0)
    with torch.no_grad():
        scores.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def test_shared_score_common_shift_rule_keeps_slack_rows():
    scores = fp32([[3.], [.2]], grad=True)
    output = allocate(scores, fp32([[.25] * 4, [.01] * 4]), fp32([[1.] * 4] * 2),
                      0., fp32([[1.] * 4] * 2), 1.)
    root = fp32([[3e38, -3e38, -3e38, -3e38], [1., 2., 3., 4.]])
    got, = torch.autograd.grad(output, (scores,), root)
    torch.testing.assert_close(got, fp32([[0.], [10.]]), rtol=0, atol=0)


def test_fixed_through_shared_mu_sums_binary_parts_at_original_root():
    mu = fp32(.25, grad=True)
    output = allocate(fp32([3.] * 4), fp32([.1] * 4), mu, mu, mu, 1.)
    got, = torch.autograd.grad(output, (mu,), fp32([3e38, 3e38, -3e38, -3e38]))
    torch.testing.assert_close(got, fp32(0.), rtol=0, atol=0)
