"""Independent entropy/KKT and boundary-VJP checks for exact Shannon caps.

The reference solves the scalar constrained entropy objective in Decimal;
it does not call production response, face selection, or reconstruction code.
"""

from decimal import Decimal, localcontext
import math

import pytest
import torch

from tide.allocation import allocate


DTYPES = [torch.float32, torch.float64]
CASES = {
    "capped_received_total": ([5., 0.], [.05, .5], [.1, 1.], [0., 0.], [.1, 1.], 1.),
    "high_total": ([5., 1., .2, -2.], [.1, .1, .1, .05], [.9, .8, .9, .7],
                   [.02, .05, .1, .2], [.7, .6, .8, .55], 1.25),
    "received_lower_total": ([-3., -2., 2., -3.], [.2, .1, .2, .1], [.7, .8, .5, .9],
                             [.02, .02, .03, .04], [.6, .7, .25, .8], 1.2),
    "slack_capped": ([3., -1., -2.], [.01, .01, .01], [.4, .6, .7],
                     [0., .01, .02], [.1, .5, .6], 1.),
    "tiny_cap": ([5., 0., -1.], [5e-31, .25, .2], [1e-30, .8, .7],
                 [0., 0., 0.], [1e-30, .8, .7], 1.),
    "large_score_gap": ([1000., -1000.], [.05, .5], [.1, 1.], [0., 0.], [.1, 1.], 1.),
}


def inputs(case, dtype, *, requires_grad=False):
    return tuple(torch.tensor(value, dtype=dtype, requires_grad=requires_grad)
                 for value in CASES[case])


def entropy_reference(scores, w, mu, lower, upper, budget):
    """Solve the constrained entropy objective at exact tensor input values."""
    with localcontext() as ctx:
        ctx.prec = 80
        scores, w, lower, upper = (
            [Decimal.from_float(float(x)) for x in value]
            for value in (scores, w, lower, upper))
        budget = Decimal.from_float(float(budget))
        high = min(budget, sum(upper))
        low = max(sum(lower), min(Decimal(1), sum(w), high))

        def response(multiplier):
            return [max(lo, min(hi, (score - 1 - multiplier).exp()))
                    for score, lo, hi in zip(scores, lower, upper)]

        values = response(Decimal(0))
        total = sum(values)
        target = max(low, min(high, total))
        multiplier = Decimal(0)
        if total != target:
            left, right = Decimal(-4096), Decimal(4096)
            assert sum(response(left)) >= target >= sum(response(right))
            for _ in range(240):
                middle = (left + right) / 2
                if sum(response(middle)) > target:
                    left = middle
                else:
                    right = middle
            multiplier = (left + right) / 2
            values = response(multiplier)
        return dict(values=[float(x) for x in values], multiplier=float(multiplier),
                    target=float(target))


@pytest.mark.parametrize("dtype", DTYPES)
def test_capped_received_total_keeps_exact_upper(dtype):
    args = inputs("capped_received_total", dtype)
    answer = allocate(*args, kind="shannon")
    assert answer[0] == args[4][0]
    assert torch.isfinite(answer).all()
    torch.testing.assert_close(answer.sum(), args[1].sum(), rtol=32 * torch.finfo(dtype).eps, atol=0)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_total_bounds_and_independent_kkt(case, dtype):
    args = inputs(case, dtype)
    scores, w, mu, lower, upper, budget = args
    answer = allocate(*args, kind="shannon")
    ref = entropy_reference(*(x.tolist() for x in args))
    expected = torch.tensor(ref["values"], dtype=dtype)
    eps = torch.finfo(dtype).eps
    # Match the existing total contract; no looser residual or support floor.
    tolerance = 32 * eps * max(abs(ref["target"]), torch.finfo(dtype).tiny)
    assert abs(float(answer.sum()) - ref["target"]) <= tolerance
    assert bool(((answer >= lower) & (answer <= upper)).all())
    assert bool(torch.isfinite(answer).all())
    torch.testing.assert_close(answer, expected, atol=tolerance, rtol=32 * eps)
    multiplier = ref["multiplier"]
    for score, actual, lo, hi in zip(scores.tolist(), answer.tolist(), lower.tolist(), upper.tolist()):
        if lo == hi:
            assert actual == lo
            continue
        assert actual > 0
        marginal = score - 1 - math.log(actual)
        kkt_tolerance = 128 * eps * max(1., abs(multiplier), abs(marginal), abs(score))
        if actual == lo:
            assert marginal <= multiplier + kkt_tolerance
        elif actual == hi:
            assert marginal >= multiplier - kkt_tolerance
        else:
            assert abs(marginal - multiplier) <= kkt_tolerance


@pytest.mark.parametrize("case", ["high_total", "received_lower_total", "slack_capped"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_vjp_matches_independent_objective_finite_differences(case, dtype):
    args = inputs(case, dtype, requires_grad=True)
    cotangent = torch.tensor([.7, -1.2, 2., .3][:len(args[0])], dtype=dtype)
    actual = torch.autograd.grad((allocate(*args, kind="shannon") * cotangent).sum(), args)
    baseline = [x.detach().tolist() for x in args]
    step = 1e-5

    def objective(values):
        result = entropy_reference(*values)["values"]
        return sum(a * b for a, b in zip(result, cotangent.tolist()))

    for parameter, gradient in enumerate(actual):
        expected = []
        length = len(baseline[parameter]) if parameter < 5 else 1
        for index in range(length):
            plus = [list(x) if isinstance(x, list) else x for x in baseline]
            minus = [list(x) if isinstance(x, list) else x for x in baseline]
            if parameter < 5:
                plus[parameter][index] += step
                minus[parameter][index] -= step
            else:
                plus[parameter] += step
                minus[parameter] -= step
            expected.append((objective(plus) - objective(minus)) / (2 * step))
        expected = torch.tensor(expected, dtype=dtype).reshape(gradient.shape)
        torch.testing.assert_close(gradient, expected,
                                   atol=3e-6 if dtype == torch.float32 else 2e-8,
                                   rtol=3e-5 if dtype == torch.float32 else 1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_slack_cap_preserves_declared_boundary_vjp(dtype):
    args = inputs("slack_capped", dtype, requires_grad=True)
    answer = allocate(*args, kind="shannon")
    score_grad, upper_grad = torch.autograd.grad(answer.sum(), (args[0], args[4]))
    assert answer[0] == args[4][0]
    assert score_grad[0] == 0 and upper_grad[0] == 1


@pytest.mark.parametrize("fixed_through_mu", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fixed_faces_keep_the_declared_vjp(fixed_through_mu, dtype):
    fixed = .4 if fixed_through_mu else .2
    values = ([3.], [.1], [.4], [fixed], [fixed], 1.)
    args = tuple(torch.tensor(value, dtype=dtype, requires_grad=True) for value in values)
    answer = allocate(*args, kind="shannon")
    gradients = torch.autograd.grad(answer.sum(), args)
    expected = [0., 0., 1. if fixed_through_mu else 0., 0. if fixed_through_mu else 1., 0., 0.]
    assert answer == args[3]
    for actual, value in zip(gradients, expected):
        assert bool((actual == value).all())


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_caps_and_empty_candidates(dtype):
    zero = torch.zeros(3, dtype=dtype)
    assert torch.equal(allocate(zero, zero, zero, zero, zero, 1., "shannon"), zero)
    empty = torch.empty(2, 0, dtype=dtype, requires_grad=True)
    answer = allocate(empty, empty, empty, empty, empty, 1., "shannon")
    answer.sum().backward()
    assert answer.shape == (2, 0) and empty.grad.shape == (2, 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_unrepresentable_positive_sender_still_fails(dtype):
    scores = torch.tensor([-1000., 0.], dtype=dtype)
    upper = torch.tensor([.4, .5], dtype=dtype)
    with pytest.raises(ArithmeticError, match="underflow would erase a positive sender"):
        allocate(scores, torch.tensor([.1, .1], dtype=dtype), upper,
                 torch.zeros_like(upper), upper, 1., "shannon")


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_noncontiguous_mixed_faces(dtype):
    cases = [CASES[name] for name in ("high_total", "received_lower_total", "slack_capped")]
    arrays = []
    for parameter in range(5):
        rows = [list(case[parameter]) + [0.] * (4 - len(case[parameter])) for case in cases]
        arrays.append(torch.tensor(rows, dtype=dtype).T.contiguous().T.requires_grad_())
    budgets = torch.tensor([case[5] for case in cases], dtype=dtype, requires_grad=True)
    assert not arrays[0].is_contiguous()
    answer = allocate(*arrays, budgets, kind="shannon")
    for index in range(3):
        ref = entropy_reference(*(value[index].detach().tolist() for value in arrays),
                                float(budgets[index].detach()))
        tolerance = 32 * torch.finfo(dtype).eps * ref["target"]
        assert abs(float(answer[index].sum().detach()) - ref["target"]) <= tolerance
        torch.testing.assert_close(answer[index], torch.tensor(ref["values"], dtype=dtype),
                                   atol=tolerance, rtol=32 * torch.finfo(dtype).eps)
    gradients = torch.autograd.grad((answer * torch.tensor([1., -2., .5, 3.], dtype=dtype)).sum(),
                                    (*arrays, budgets))
    assert all(bool(torch.isfinite(value).all()) for value in gradients)


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_log_cap_tie_keeps_upper_boundary_convention(dtype):
    values = ([1.], [.1], [1.], [0.], [1.], 1.)
    args = tuple(torch.tensor(value, dtype=dtype, requires_grad=True) for value in values)
    answer = allocate(*args, kind="shannon")
    gradients = torch.autograd.grad(answer.sum(), args)
    assert answer.item() == 1.
    assert [value.item() for value in gradients] == [0., 0., 0., 0., 1., 0.]
