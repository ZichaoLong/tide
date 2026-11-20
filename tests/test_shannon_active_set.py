"""Real training cap cancellation and independent bounded entropy witnesses."""

from decimal import Decimal, localcontext
import os

import pytest
import torch

from tide.allocation import allocate


DEVICE = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
if DEVICE.startswith("npu"):
    import torch_npu  # noqa: F401


def reference(values, cotangent):
    """Decimal dual solve and analytic solution VJP, independent of production."""
    with localcontext() as context:
        context.prec = 80
        s, w, mu, lower, upper = [[Decimal.from_float(float(x)) for x in row]
                                 for row in values[:5]]
        k = Decimal.from_float(float(values[5]))
        high = min(k, sum(upper))
        low = max(sum(lower), min(Decimal(1), sum(w), high))

        def response(multiplier):
            return [max(l, min(u, (score - 1 - multiplier).exp()))
                    for score, l, u in zip(s, lower, upper)]

        zero = response(Decimal(0))
        total = sum(zero)
        target = max(low, min(high, total))
        result = zero
        if total != target:
            left, right = Decimal(-4096), Decimal(4096)
            for _ in range(256):
                middle = (left + right) / 2
                if sum(response(middle)) > target:
                    left = middle
                else:
                    right = middle
            result = response((left + right) / 2)
        slopes = [c if l < c < u else Decimal(0) for c, l, u in zip(result, lower, upper)]
        roots = [Decimal.from_float(float(x)) for x in cotangent]
        average = (sum(c * g for c, g in zip(slopes, roots)) / sum(slopes)
                   if total != target and sum(slopes) else Decimal(0))
        ds = [c * (g - average) for c, g in zip(slopes, roots)]
        return [float(x) for x in result], [float(x) for x in ds]


def check(values, root=None):
    args = [torch.tensor(v, dtype=torch.float32, device=DEVICE, requires_grad=True) for v in values]
    if root is None:
        root = [1.] + [0.] * (len(values[0]) - 1)
    cotangent = torch.tensor(root, dtype=torch.float32, device=DEVICE)
    output = allocate(*args, kind="shannon")
    gradient, = torch.autograd.grad(output, (args[0],), cotangent)
    expected, ds = reference([x.detach().cpu().tolist() for x in args], root)
    torch.testing.assert_close(output.cpu().double(), torch.tensor(expected, dtype=torch.float64),
                               rtol=3e-5, atol=2.**-149)
    torch.testing.assert_close(gradient.cpu().double(), torch.tensor(ds, dtype=torch.float64),
                               rtol=3e-5, atol=2.**-149)
    assert bool((output[args[4] > 0] > 0).all())
    assert bool(((output >= args[3]) & (output <= args[4])).all())
    return output, gradient


@pytest.mark.parametrize("gap", [16.647329807281494, 18., 40., 90.])
def test_positive_weak_sender_and_coupled_gradient_survive_rounded_cap(gap):
    check(([6.040277004241943, 6.040277004241943 - gap], [1., 1.], [1., 1.],
           [0., 0.], [1., 1.], 1.))


@pytest.mark.parametrize("budget", [1, 2, 4, 8])
@pytest.mark.parametrize("gap", [16.647329807281494, 40.])
def test_sixteen_candidates_all_four_budgets_preserve_tail_and_vjp(budget, gap):
    scores = [6.] * budget + [6. - gap] * (16 - budget)
    check((scores, [1.] * 16, [1.] * 16, [0.] * 16, [1.] * 16, float(budget)))


@pytest.mark.parametrize("budget", [1, 2, 4, 8])
def test_sixteen_candidates_with_actual_caps_and_floors(budget):
    scores = [8., 5., 3., 2., .8, .2, -.4, -1., -2., -3., -4., -5., -6., -7., -8., -9.]
    upper = [.15, .25, .45, .65] + [1.] * 12
    lower = [.02, 0., .03, 0.] + [0.] * 12
    check((scores, [x / 2 for x in upper], upper, lower, upper, float(budget)),
          [1., -.5, .25, -.75] * 4)


@pytest.mark.parametrize("gap", [16.647329807281494, 30., 40.])
def test_bound_mass_subtraction_retains_compensation(gap):
    # FP32(.1)+FP32(.9) is below one in real arithmetic. Losing the residual's
    # low part would misclassify the .9 cap and erase the 2.235e-8 sender.
    check(([6., 6., 6. - gap], [1., 1., 1.], [1., 1., 1.],
           [0., 0., 0.], [.1, .9, 1.], 1.), [0., 1., 0.])


def test_exact_interior_to_cap_tie_keeps_boundary_vjp():
    scores = torch.tensor([2., 2.], device=DEVICE, requires_grad=True)
    upper = torch.tensor([.5, 1.], device=DEVICE, requires_grad=True)
    output = allocate(scores, torch.ones_like(scores), torch.ones_like(scores), 0., upper, 1., "shannon")
    gs, gu = torch.autograd.grad(output, (scores, upper), torch.tensor([1., 0.], device=DEVICE))
    torch.testing.assert_close(output, torch.tensor([.5, .5], device=DEVICE), rtol=0, atol=0)
    torch.testing.assert_close(gs, torch.zeros_like(gs), rtol=0, atol=0)
    torch.testing.assert_close(gu, torch.tensor([1., 0.], device=DEVICE), rtol=0, atol=0)


def test_floor_sum_rounded_to_target_still_has_positive_remaining_mass():
    check(([6., 6., 6.], [1., 1., 1.], [1., 1., 1.],
           [.1, .9, 0.], [1., 1., 1.], 1.), [0., 0., 1.])


def test_near_exhausted_floors_do_not_hide_true_positive_sender_underflow():
    scores = torch.tensor([6., 6., 6., -120.], device=DEVICE)
    with pytest.raises(ArithmeticError, match="Shannon allocation underflow would erase a positive sender"):
        allocate(scores, torch.ones_like(scores), torch.ones_like(scores),
                 torch.tensor([.1, .9, 0., 0.], device=DEVICE), torch.ones_like(scores), 1., "shannon")


@pytest.mark.parametrize("budget", [1, 2, 4, 8])
def test_truly_unrepresentable_sender_keeps_original_guard(budget):
    scores = torch.tensor([6.] * budget + [-120.] * (16 - budget), device=DEVICE)
    with pytest.raises(ArithmeticError, match="Shannon allocation underflow would erase a positive sender"):
        allocate(scores, torch.ones_like(scores), torch.ones_like(scores), 0., torch.ones_like(scores),
                 float(budget), "shannon")


def test_shared_score_alias_and_repeated_backward_keep_version_guard():
    scores = torch.tensor([[6., -10.6473]], device=DEVICE, requires_grad=True)
    mu = torch.ones((3, 2), device=DEVICE)
    output = allocate(scores, mu, mu, 0., mu, 1., "shannon")
    root = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], device=DEVICE)
    first, = torch.autograd.grad(output, (scores,), root, retain_graph=True)
    second, = torch.autograd.grad(output, (scores,), root, retain_graph=True)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first[0, 0] > 0 and first[0, 1] < 0
    with torch.no_grad():
        scores.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()
