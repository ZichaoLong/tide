"""Independent active-set and differential checks for the new allocation API."""

import itertools
import math

import pytest
import torch

import tide.allocation as allocation
from tide.allocation import allocate, pool


DTYPE = torch.float64


def tensor(values, grad=False):
    return torch.tensor(values, dtype=DTYPE, requires_grad=grad)


def active_set_oracle(scores, w, lower, upper, k, kind, total_bounds=None):
    """Enumerate faces and their stationary points; no multiplier root solve."""
    n = len(scores)
    high = min(k, sum(upper))
    low = max(sum(lower), min(min(1, sum(w)), high))
    if total_bounds is not None:
        low, high = total_bounds
    candidates = []
    for faces in itertools.product((0, 1, 2), repeat=n):
        free = [i for i, face in enumerate(faces) if face == 1]
        for total in (None, low, high):
            values = [lower[i] if face == 0 else upper[i] if face == 2 else 0
                      for i, face in enumerate(faces)]
            if free:
                if total is None:
                    interior = [scores[i] if kind == "quad" else math.exp(scores[i] - 1) for i in free]
                elif kind == "quad":
                    shift = (sum(scores[i] for i in free) + sum(values) - total) / len(free)
                    interior = [scores[i] - shift for i in free]
                else:
                    exps = [math.exp(scores[i] - max(scores[j] for j in free)) for i in free]
                    interior = [(total - sum(values)) * e / sum(exps) for e in exps]
                for i, value in zip(free, interior):
                    values[i] = value
            if any(x < l - 1e-12 or x > u + 1e-12 for x, l, u in zip(values, lower, upper)):
                continue
            if not low - 1e-12 <= sum(values) <= high + 1e-12:
                continue
            if total is not None and abs(sum(values) - total) > 1e-12:
                continue
            regularizer = (-sum(x * x for x in values) / 2 if kind == "quad"
                           else -sum(x * math.log(x) for x in values if x > 0))
            candidates.append((sum(s * x for s, x in zip(scores, values)) + regularizer, values))
    assert candidates
    return max(candidates, key=lambda pair: pair[0])[1]


@pytest.mark.parametrize("kind", ["quad", "shannon"])
def test_batched_allocation_against_independent_face_enumeration(kind):
    generator = torch.Generator().manual_seed(304)
    scores = torch.randn(2, 6, 3, generator=generator, dtype=DTYPE)
    mu = .4 + .5 * torch.rand(2, 6, 3, generator=generator, dtype=DTYPE)
    w, lower, upper = mu * .55, mu * .12, mu * .85
    budget = tensor([[1., 1.4, 2., 1., 1.5, 2.]])
    actual = allocate(scores, w, mu, lower, upper, budget, kind)
    expected = []
    for batch in range(2):
        expected.append([active_set_oracle(scores[batch, r].tolist(), w[batch, r].tolist(),
                                         lower[batch, r].tolist(), upper[batch, r].tolist(),
                                         budget[0, r].item(), kind) for r in range(6)])
    torch.testing.assert_close(actual, tensor(expected), atol=2e-13, rtol=2e-13)


def test_upstream_role_and_weak_received_examples():
    ones = torch.ones(3, dtype=DTYPE)
    actual = allocate(tensor([-2., 1., .8]), ones, ones, tensor([.25, 0., 0.]), ones, 2.)
    torch.testing.assert_close(actual, tensor([.25, .975, .775]))
    actual = allocate(tensor([-2., 1.]), tensor([.1, .2]), tensor([.2, .4]),
                      tensor([0., 0.]), tensor([.2, .4]), 1.)
    torch.testing.assert_close(actual, tensor([0., .4]))


@pytest.mark.parametrize("kind", ["quad", "shannon"])
@pytest.mark.parametrize("branch", ["lower", "upper", "slack"])
def test_all_independent_input_vjps_on_smooth_faces(kind, branch):
    if branch == "lower":
        scores, w, bounds, budget = [-1., -.7, -.8], [.2, .25, .15], [.7, .6, .5], 1.4
        if kind == "shannon":
            scores = [-2., -1.8, -1.9]
    elif branch == "upper":
        scores, w, bounds, budget = [1.2, 1.02, .8], [.1, .15, .15], [.4, .6, .5], 1.3
        if kind == "shannon":
            scores = [2.2, 1.9, 1.6]
    else:
        scores, w, bounds, budget = [.3, .35, .25], [.1, .1, .1], [.7, .6, .5], 1.6
        if kind == "shannon":
            scores = [1 + math.log(value) for value in scores]
    inputs = tuple(tensor(value, True) for value in
                   (scores, w, [.9, .8, .7], [.05, .04, .03], bounds, budget))
    assert torch.autograd.gradcheck(lambda *xs: allocate(*xs, kind=kind), inputs,
                                   eps=1e-6, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("kind", ["quad", "shannon"])
def test_fixed_through_and_floor_exhausted_vjps(kind):
    scores, w, mu = tensor([8.], True), tensor([.2], True), tensor([.4], True)
    lower, upper = tensor([.4], True), tensor([.4], True)
    output = allocate(scores, w, mu, lower, upper, 1., kind)
    gradients = torch.autograd.grad(output.sum(), (scores, w, mu, lower, upper))
    torch.testing.assert_close(output, mu)
    for grad, expected in zip(gradients, ([0.], [0.], [1.], [0.], [0.])):
        torch.testing.assert_close(grad, tensor(expected))
    low = tensor([.5, .5], True)
    scores = tensor([3., 2.], True)
    out = allocate(scores, tensor([.2, .2]), tensor([1., 1.]), low, tensor([1., 1.]), 1., kind)
    gs, gl = torch.autograd.grad((out * tensor([2., 3.])).sum(), (scores, low))
    torch.testing.assert_close(out, low)
    torch.testing.assert_close(gs, tensor([0., 0.]))
    torch.testing.assert_close(gl, tensor([2., 3.]))


def test_declared_ties_slack_clip_and_first_min():
    scores, w = tensor([.5, .5], True), tensor([.5, .5], True)
    out = allocate(scores, w, tensor([1., 1.]), tensor([0., 0.]), tensor([1., 1.]), 1.)
    gs, gw = torch.autograd.grad((out * tensor([2., 3.])).sum(), (scores, w))
    torch.testing.assert_close(gs, tensor([2., 3.]))  # S(0)==L==U: slack convention.
    torch.testing.assert_close(gw, tensor([0., 0.]))
    scores, lower, upper = (tensor(x, True) for x in ([.2, .7], [.2, .1], [.8, .7]))
    out = allocate(scores, tensor([.1, .1]), tensor([1., 1.]), lower, upper, 2.)
    gs, gl, gu = torch.autograd.grad(out.sum(), (scores, lower, upper))
    torch.testing.assert_close(gs, tensor([0., 0.]))
    torch.testing.assert_close(gl, tensor([1., 0.]))
    torch.testing.assert_close(gu, tensor([0., 1.]))
    w = tensor([.5, .5], True)
    out = allocate(tensor([-1., -1.]), w, tensor([1., 1.]), tensor([0., 0.]), tensor([1., 1.]), 2.)
    torch.testing.assert_close(torch.autograd.grad(out.sum(), w)[0], tensor([0., 0.]))


@pytest.mark.parametrize("kind", ["quad", "shannon"])
def test_empty_zero_and_zero_insertion(kind):
    empty = [torch.empty(3, 0, dtype=DTYPE, requires_grad=True) for _ in range(5)]
    budget = tensor(1., True)
    out = allocate(*empty, budget, kind)
    assert out.shape == (3, 0)
    assert all(g is not None for g in torch.autograd.grad(out.sum(), (*empty, budget)))
    actual = allocate(tensor([.2, .8, 900.]), tensor([.4, .3, 0.]), tensor([.8, .6, 0.]),
                      tensor([0., 0., 0.]), tensor([.8, .6, 0.]), 1., kind)
    compact = allocate(tensor([.2, .8]), tensor([.4, .3]), tensor([.8, .6]),
                       tensor([0., 0.]), tensor([.8, .6]), 1., kind)
    torch.testing.assert_close(actual[:2], compact)
    assert actual[2] == 0
    zeros = torch.zeros(2, 3, dtype=DTYPE)
    torch.testing.assert_close(allocate(zeros, zeros, zeros, zeros, zeros, 1., kind), zeros)


@pytest.mark.parametrize("kind", ["quad", "shannon"])
def test_singleton_received_interval_and_large_common_scores(kind):
    torch.testing.assert_close(allocate(tensor([-8.]), tensor([.2]), tensor([.4]),
                                       tensor([0.]), tensor([.4]), 1., kind), tensor([.2]))
    actual = allocate(tensor([-1e12, -1e12]), tensor([1., 1.]), tensor([1., 1.]),
                      tensor([0., 0.]), tensor([1., 1.]), 1., kind)
    torch.testing.assert_close(actual, tensor([.5, .5]), atol=2e-14, rtol=2e-14)


def test_pool_formula_right_derivative_and_not_a_mu_cap():
    scores, mu = tensor([math.log(9.), 0., -.3], True), tensor([.1, .1, 0.], True)
    actual = pool(scores, mu)
    numerator = mu * scores.exp()
    expected = mu.sum() * numerator / numerator.sum()
    torch.testing.assert_close(actual, tensor([.18, .02, 0.]))
    torch.testing.assert_close(actual, expected)
    upstream = tensor([2., -.5, 1.])
    grads = torch.autograd.grad((actual * upstream).sum(), (scores, mu), retain_graph=True)
    reference = torch.autograd.grad((expected * upstream).sum(), (scores, mu))
    for got, want in zip(grads, reference):
        torch.testing.assert_close(got, want)
    assert grads[1][2] != 0
    zeros = tensor([0., 0.], True)
    output = pool(tensor([2., -1.], True), zeros)
    torch.testing.assert_close(torch.autograd.grad(output.sum(), zeros)[0], tensor([0., 0.]))


def test_broadcast_aliases_and_invalid_descriptors():
    mu = tensor([.5, .6, .7], True)
    out = allocate(tensor([[.2, .6, .9], [-1., -.5, -.3]]), mu / 2, mu, 0., mu, tensor([[1.], [2.]]))
    assert out.shape == (2, 3)
    assert torch.isfinite(torch.autograd.grad(out.square().sum(), mu)[0]).all()
    with pytest.raises(ValueError, match="floors"):
        allocate(tensor([1., 1.]), tensor([1., 1.]), tensor([1., 1.]), tensor([.6, .6]), tensor([1., 1.]), 1.)
    with pytest.raises(ValueError, match="invalid"):
        allocate(tensor([1.]), tensor([0.]), tensor([1.]), tensor([0.]), tensor([1.]), 1.)


@pytest.mark.parametrize("kind", ["quad", "shannon", "pool"])
def test_fp32_matches_fp64_values_and_vjp(kind):
    scores = torch.tensor([-.8, -.6, -.7], dtype=torch.float32, requires_grad=True)
    mu = torch.tensor([.7, .8, .9], dtype=torch.float32, requires_grad=True)
    def evaluate(s, m):
        return allocate(s, m / 2, m, m * .1, m * .9, 1.5, kind)
    output = evaluate(scores, mu)
    s64, m64 = (x.detach().double().requires_grad_() for x in (scores, mu))
    reference = evaluate(s64, m64)
    torch.testing.assert_close(output.double(), reference, atol=1e-7, rtol=2e-6)
    got = torch.autograd.grad(output.square().sum(), (scores, mu))
    want = torch.autograd.grad(reference.square().sum(), (s64, m64))
    for a, b in zip(got, want):
        torch.testing.assert_close(a.double(), b, atol=2e-7, rtol=2e-6)


def test_allocation_retains_source_version_guards():
    scores = tensor([.2, .8], True)
    out = allocate(scores, tensor([.5, .5]), tensor([1., 1.]), 0., tensor([1., 1.]), 1.)
    with torch.no_grad():
        scores.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


@pytest.mark.parametrize("kind", ["quad", "shannon"])
def test_weak_received_total_is_checked_relatively(kind):
    scores = tensor([-2., -2.] if kind == "quad" else [-30., -30.])
    w, mu = tensor([1e-9, 2e-9]), tensor([4e-9, 4e-9])
    out = allocate(scores, w, mu, 0., mu, 1., kind)
    torch.testing.assert_close(out, tensor([1.5e-9, 1.5e-9]), atol=0, rtol=1e-14)


@pytest.mark.parametrize("kind", ["shannon", "pool"])
def test_positive_sender_underflow_is_explicit(kind):
    with pytest.raises(ArithmeticError, match="positive sender"):
        allocate(tensor([-1000., 0.]), tensor([1., 1.]), tensor([1., 1.]),
                 0., tensor([1., 1.]), 1., kind)


def test_pool_rejects_final_mass_underflow_even_when_probability_survives():
    with pytest.raises(ArithmeticError, match="positive sender"):
        pool(tensor([-700., 0.]), tensor([1e-200, 1e-200]))


@pytest.mark.parametrize("n", [8, 32, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sorted_quad_against_independent_repeated_face_enumeration(n, dtype):
    # Strict convexity makes identical coordinates equal. Enumerating four
    # distinct classes therefore independently solves all n coordinates.
    repeats = n // 4
    cases = [([1.6, 1.2, .8, -.3], [.9, .8, .7, .6], [.1, 0., .05, 0.]),
             ([-1.1, -.9, -.7, -.6], [.9, .8, .7, .6], [.1, 0., .05, 0.]),
             ([-.01, -.02, -.03, -.04], [.002, .003, .004, .005], [.002, .0003, .0004, .0005])]
    for values, caps, floors in cases:
        s, mu, lower = [torch.tensor(x, dtype=dtype) for x in (values, caps, floors)]
        w, upper, k = mu / 2, mu, max(1, n // 8)
        arguments = [x.repeat_interleave(repeats) for x in (s, w, mu, lower, upper)]
        result = allocate(*arguments, k)
        high = min(k, repeats * sum(upper.tolist()))
        low = max(repeats * sum(lower.tolist()), min(1, repeats * sum(w.tolist()), high))
        expected = active_set_oracle(s.tolist(), w.tolist(), lower.tolist(), upper.tolist(), k, "quad",
                                     total_bounds=(low / repeats, high / repeats))
        expected = torch.tensor(expected, dtype=dtype).repeat_interleave(repeats)
        atol, rtol = (1e-6, 5e-6) if dtype == torch.float32 else (2e-13, 2e-13)
        torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n", [8, 32, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sorted_quad_matches_retained_bisection_values_and_vjps(monkeypatch, n, dtype):
    generator = torch.Generator().manual_seed(172)
    scores = torch.randn(6, n, generator=generator, dtype=dtype)
    mu = .5 + .5 * torch.rand(6, n, generator=generator, dtype=dtype)
    # Two tiny role floors per region leave room for the ordinary competitors.
    lower = torch.zeros_like(mu)
    lower[:, :2] = .03
    roots = tuple(x.requires_grad_() for x in (scores, mu / 2, mu, lower, mu * .9))
    budget = float(max(1, n // 32))
    result = allocate(*roots, budget)
    seed = torch.randn(result.shape, generator=generator, dtype=dtype)
    gradients = torch.autograd.grad(result, roots, seed)
    # Exercise the retained production bisection, not a second copy of the
    # sorted-prefix algorithm. The enumeration above is the independent anchor.
    def force_original(shifted, low, high, target):
        return torch.zeros_like(shifted), torch.zeros_like(target, dtype=torch.bool)
    monkeypatch.setattr(allocation, "_quad_sorted", force_original)
    reference_roots = tuple(x.detach().clone().requires_grad_() for x in roots)
    reference = allocate(*reference_roots, budget)
    reference_gradients = torch.autograd.grad(reference, reference_roots, seed)
    atol, rtol = (1e-6, 5e-6) if dtype == torch.float32 else (2e-13, 2e-13)
    torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)
    for actual, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("n", [8, 32, 128])
@pytest.mark.parametrize("dtype,offset", [(torch.float32, 1e6), (torch.float64, 1e12)])
def test_equal_enter_leave_breakpoints_keep_boundary_vjp(monkeypatch, n, dtype, offset):
    # Half of the nodes leave the free set exactly when the other half enter.
    scores = torch.tensor([offset + .25] * (n // 2) + [offset + 1.25] * (n // 2),
                          dtype=dtype, requires_grad=True)
    caps = torch.ones(n, dtype=dtype)
    def no_fallback(*args):
        pytest.fail("exact tied breakpoints should be solved by the finite path")
    monkeypatch.setattr(allocation, "_bisect", no_fallback)
    result = allocate(scores, caps, caps, 0., caps, n // 2)
    expected = torch.cat((torch.zeros(n // 2, dtype=dtype), torch.ones(n // 2, dtype=dtype)))
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    gradient = torch.autograd.grad(result.sum(), scores)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(scores), atol=0, rtol=0)


def test_weak_mass_prefix_cancellation_uses_original_fallback(monkeypatch):
    scores = torch.tensor([[-2.] * 8, [-2., -3., -100., -1000., 0., 0., 0., 0.]], requires_grad=True)
    w = torch.tensor([[1.] * 8, [2e-7, 1e-8, 1e-8, 1e-8, 0., 0., 0., 0.]], requires_grad=True)
    mu = (2 * w).clamp_max(1)
    called = []
    original = allocation._bisect
    def count_fallback(shifted, lower, upper, target, solve, kind):
        called.append(solve.clone())
        return original(shifted, lower, upper, target, solve, kind)
    monkeypatch.setattr(allocation, "_bisect", count_fallback)
    result = allocate(scores, w, mu, 0., mu, 1.)
    assert len(called) == 1 and torch.equal(called[0], torch.tensor([[False], [True]]))
    torch.testing.assert_close(result[0], torch.full((8,), 1 / 8), atol=0, rtol=0)
    expected = torch.zeros(8)
    # Sum the binary32 descriptors exactly before rounding the received target.
    expected[0] = w[1].detach().double().sum().float()
    torch.testing.assert_close(result[1], expected, atol=0, rtol=0)
    score_grad, received_grad = torch.autograd.grad(result[1, 0], (scores, w))
    torch.testing.assert_close(score_grad, torch.zeros_like(scores), atol=0, rtol=0)
    torch.testing.assert_close(received_grad[1, :4], torch.ones(4), atol=0, rtol=0)


def test_collapsed_breakpoints_are_rejected_even_if_total_residual_hides_them():
    shifted = torch.tensor([0., -1.])
    lower, upper, target = torch.zeros(2), torch.tensor([1., 1e-8]), torch.tensor([.5])
    candidate, valid = allocation._quad_sorted(shifted, lower, upper, target)
    assert candidate.sum() == target[0] and not valid.item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("positive", [False, True])
def test_flat_multiplier_interval_preserves_exact_zero_and_tiny_positive_support(monkeypatch, dtype, positive):
    # Actual FC regression: prefix cancellation placed the multiplier one ULP
    # before the flat interval and invented a 2.22e-16 sender in FP64.
    scores = torch.tensor([0.5230857817411676, 1.7626989407125895],
                          dtype=dtype, requires_grad=True)
    caps = torch.ones(2, dtype=dtype)
    budget = torch.tensor(1 + (4 * torch.finfo(dtype).eps if positive else 0),
                          dtype=dtype, requires_grad=True)
    def no_fallback(*args):
        pytest.fail("the selected corner face should be solved by the finite path")
    monkeypatch.setattr(allocation, "_bisect", no_fallback)
    result = allocate(scores, caps, caps, 0., caps, budget)
    expected = torch.stack((budget.detach() - 1, budget.new_ones(())))
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert bool(result[0] > 0) == positive
    score_gradient, budget_gradient = torch.autograd.grad(result[0], (scores, budget))
    torch.testing.assert_close(score_gradient, torch.zeros_like(scores), atol=0, rtol=0)
    torch.testing.assert_close(budget_gradient, budget.new_tensor(float(positive)), atol=0, rtol=0)


@pytest.mark.parametrize("dtype,gap", [(torch.float32, -1.9289610385894775),
                                      (torch.float64, -1.641518376635315)])
@pytest.mark.parametrize("positive", [False, True])
def test_repeated_score_corner_keeps_support_after_face_mean_reconstruction(dtype, gap, positive):
    # Three identical low scores expose rounding in an uncentered face mean.
    shifted = torch.tensor([gap] * 3 + [0.] * 3, dtype=dtype)
    lower, upper = torch.zeros_like(shifted), torch.ones_like(shifted)
    target = shifted.new_tensor([3 * (1 + (4 * torch.finfo(dtype).eps if positive else 0))])
    result, valid = allocation._quad_sorted(shifted, lower, upper, target)
    assert valid.item()
    expected = torch.cat(((target - 3).expand(3) / 3, torch.ones(3, dtype=dtype)))
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert bool((result[:3] > 0).all()) == positive


@pytest.mark.parametrize("n", [8, 32, 128])
def test_sorted_quad_directional_derivative_in_strict_interior(n):
    generator = torch.Generator().manual_seed(654)
    scores = (torch.rand(n, generator=generator, dtype=DTYPE) * .02 + 1).requires_grad_()
    w = torch.full((n,), .1 / n, dtype=DTYPE, requires_grad=True)
    mu = torch.full((n,), .8, dtype=DTYPE, requires_grad=True)
    lower = torch.full((n,), .01, dtype=DTYPE, requires_grad=True)
    upper = torch.full((n,), .7, dtype=DTYPE, requires_grad=True)
    k = tensor(n / 4, True)
    roots = (scores, w, mu, lower, upper, k)
    directions = tuple(torch.randn(x.shape, generator=generator, dtype=DTYPE) * .01 for x in roots)
    seed = torch.randn(n, generator=generator, dtype=DTYPE)
    result = allocate(*roots)
    gradients = torch.autograd.grad(result, roots, seed)
    expected = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions))
    epsilon = 1e-6
    with torch.no_grad():
        plus = allocate(*(x + epsilon * direction for x, direction in zip(roots, directions)))
        minus = allocate(*(x - epsilon * direction for x, direction in zip(roots, directions)))
    numerical = ((plus - minus) * seed).sum() / (2 * epsilon)
    torch.testing.assert_close(expected, numerical, atol=2e-8, rtol=2e-6)
