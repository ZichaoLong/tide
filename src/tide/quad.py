"""Mass comparisons and local face recovery for the bounded quadratic projection."""
import math
from fractions import Fraction
from typing import NamedTuple

import torch
from .scaled_arithmetic import compensated_sum, two_sum


def mass_parts(values):
    if values.shape[-1] == 2:
        a, b = values[..., :1], values[..., 1:]
        high = a + b
        return high, torch.minimum(a, b) - (high - torch.maximum(a, b))
    return compensated_sum(values)


class _MassSum(torch.autograd.Function):
    """The derivative of the real total is one for every term."""
    @staticmethod
    def forward(ctx, values):
        if values.shape[-1] == 2:
            high, tail = mass_parts(values)
            lost = None
            ctx.mark_non_differentiable(tail)
        else:
            high, tail, lost = compensated_sum(values, track_roundoff=True)
            ctx.mark_non_differentiable(tail, lost)
        ctx.shape = values.shape
        return high, tail, lost

    @staticmethod
    def backward(ctx, grad, *unused):
        return grad.expand(ctx.shape)


def less(a, b):
    return (a[0] < b[0]) | ((a[0] == b[0]) & (a[1] < b[1]))


def select(mask, a, b):
    return torch.where(mask, a[0], b[0]), torch.where(mask, a[1], b[1])


class Totals(NamedTuple):
    floor: tuple
    cap: tuple
    low: tuple
    high: tuple
    initial: tuple | None
    exact: object | None


class _ExactRows(NamedTuple):
    mask: torch.Tensor
    values: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    choices: tuple

    def replace(self, tensor, replacement):
        result = tensor.clone()
        result[self.mask.squeeze(-1)] = replacement
        return result


def total_bounds(scores, w, lower, upper, budget):
    raw_floor, raw_cap, raw_mass = (_MassSum.apply(x) for x in (lower, upper, w))
    floor, cap, mass = raw_floor[:2], raw_cap[:2], raw_mass[:2]
    initial = exact = None
    if scores.shape[-1] > 2:
        with torch.no_grad():
            summed = compensated_sum(scores.clamp(min=lower, max=upper), track_roundoff=True)
            initial = summed[:2]
            lost = raw_floor[2] | raw_cap[2] | raw_mass[2] | summed[2]
            if bool(lost.any()):
                width = scores.shape[-1]
                packed = torch.cat((scores,w,lower,upper,budget), -1)[lost.squeeze(-1)]
                rows = [_recover(row, width, scores.dtype, totals=True) for row in packed.cpu().tolist()]
                values, lf, uf, choices = scores.new_tensor(rows).split((width,width,width,6), -1)
                exact = _ExactRows(lost, values, lf.bool(), uf.bool(), choices.bool().split(1, -1))

    def choose(first, second, index):
        order = less(first, second)
        return exact.replace(order, exact.choices[index]) if exact is not None else order

    zero = torch.zeros_like(budget)
    limit = budget, zero
    unit = torch.ones_like(budget), zero
    # Keep the source tensor of every min/max. At a true tie the first
    # operand owns its derivative, even when the two stored heads coincide.
    high = select(choose(cap, limit, 0), cap, limit)
    invalid = less(high, floor)
    if exact is not None:
        invalid = invalid & ~exact.mask  # Exact recovery checked feasibility from the input descriptors.
    if bool(invalid.any()):
        raise ValueError("role floors exceed the available budget")
    received = select(choose(mass, unit, 1), mass, unit)
    requested = select(choose(high, received, 2), high, received)
    low = select(choose(floor, requested, 3), requested, floor)
    return Totals(floor, cap, low, high, initial, exact)


def disputed(scores, lower, upper, c, target, solve, tolerance):
    """Screen feasibility and boundary ambiguity. Only exact recovery selects a new face."""
    if c.shape[-1] == 2:
        head, tail = mass_parts(c)
        lost = False
    else:
        head, tail, lost = compensated_sum(c, track_roundoff=True)
    residual = (head - target[0]) + (tail - target[1])
    inside = (c > lower) & (c < upper)
    has_inside = inside.any(-1, keepdim=True)
    eps = 8 * torch.finfo(c.dtype).eps
    near_endpoint = inside & (torch.minimum(c - lower, upper - c) <= eps * target[0])
    different_total = ((residual.abs() > tolerance) | (~has_inside & (residual != 0)) |
                       (target[1] != 0) | lost | ~torch.isfinite(c).all(-1, keepdim=True))
    if c.shape[-1] == 2:
        first, second = c[..., :1], c[..., 1:]
        gap = scores[..., :1] - scores[..., 1:]
        margin = gap - (first - second)
        uncertainty = eps * (gap.abs() + first + second)
        increase = (first < upper[..., :1]) & (second > lower[..., 1:])
        decrease = (first > lower[..., :1]) & (second < upper[..., 1:])
        # At a corner a feasible direction must have a strictly worse
        # marginal score to certify it without recomputing the true face.
        ambiguous = torch.where(increase & decrease, margin.abs() > uncertainty,
                                (increase & (margin >= -uncertainty)) | (decrease & (margin <= uncertainty)))
        return solve & (different_total | ambiguous | near_endpoint.any(-1, keepdim=True))
    # An interior rounded to a box endpoint can change the Jacobian even when
    # the stored total passes. Compare KKT inequalities against a stored
    # interior using local score differences (a common score offset cancels).
    index = inside.to(torch.int32).argmax(-1, keepdim=True)
    reference = c.gather(-1, index)
    gap = scores - scores.gather(-1, index)
    margin = gap - (c - reference)
    uncertainty = eps * (gap.abs() + c + reference)
    free = lower < upper
    near_face = free & (((c <= lower) & (margin >= -uncertainty)) |
                        ((c >= upper) & (margin <= uncertainty)))
    # On a flat face compare the two marginal endpoints. Center first to
    # retain small score differences when all scores share a large origin.
    centered = scores - torch.where(free, scores, -torch.inf).amax(-1, keepdim=True)
    left = torch.where(free & (c <= lower), centered - lower, -torch.inf).amax(-1, keepdim=True)
    right = torch.where(free & (c >= upper), centered - upper, torch.inf).amin(-1, keepdim=True)
    flat_uncertain = (torch.isfinite(left) & torch.isfinite(right) &
                      (left - right >= -eps * (left.abs() + right.abs() + 1)))
    uncertain = torch.where(has_inside, (near_endpoint | near_face).any(-1, keepdim=True), flat_uncertain)
    return solve & (different_total | uncertain)


def _round_mass(value, dtype):
    """Round the exact rational once, including subnormals and halfway ties."""
    if value == 0:
        return 0.
    n, d = value.as_integer_ratio()
    exponent = n.bit_length() - d.bit_length()
    if (n << max(0, -exponent)) < (d << max(0, exponent)):
        exponent -= 1
    bits, smallest = (24, -149) if dtype == torch.float32 else (53, -1074)
    shift = max(smallest, exponent - bits + 1)
    units, rem = divmod(n << max(0, -shift), d << max(0, shift))
    divisor = d << max(0, shift)
    units += 2 * rem > divisor or (2 * rem == divisor and units % 2)
    return math.ldexp(units, shift)


def _recover(row, width, dtype, *, totals=False):
    # Dyadic inputs are exact in Fraction. Reconstruct the selected total from
    # the public source values, so a capacity sum never becomes a scalar budget.
    s, w, lo, hi = ([Fraction(x) for x in row[i:i+width]] for i in range(0, 4*width, width))
    budget = Fraction(row[-1])
    capacity, floors, mass = sum(hi), sum(lo), sum(w)
    cap = min(budget, capacity)
    if floors > cap:
        raise ValueError("role floors exceed the available budget")
    received = min(1, mass)
    floor = max(floors, min(received, cap))
    c = [max(l, min(u, v)) for l, u, v in zip(lo, hi, s)]
    initial = sum(c)
    target = min(cap, max(floor, initial))
    if initial != target:
        free = [i for i in range(width) if lo[i] != hi[i]]
        fixed_mass = sum(lo[i] for i in range(width) if lo[i] == hi[i])
        # On each pass clipping the unconstrained face proves at least one
        # bound permanent: excess mass pegs lower bounds, a deficit pegs caps.
        while free:
            multiplier = (sum(s[i] for i in free) - (target - fixed_mass)) / len(free)
            for i in free:
                c[i] = max(lo[i], min(hi[i], s[i] - multiplier))
            delta = sum(c) - target
            if delta == 0:
                break
            pegged = [i for i in free if c[i] == (lo[i] if delta > 0 else hi[i])]
            if not pegged:
                raise ArithmeticError("quad face could not meet its feasible target")
            fixed_mass += sum(c[i] for i in pegged)
            removed = set(pegged)
            free = [i for i in free if i not in removed]
    if sum(c) != target:
        raise ArithmeticError("quad face did not conserve its target")
    result = ([_round_mass(v, dtype) for v in c] +
            [l != u and v == l for l, u, v in zip(lo, hi, c)] +
            [l != u and v == u for l, u, v in zip(lo, hi, c)])
    if totals:
        result.extend((capacity < budget, mass < 1, cap < received, floors < min(received,cap),
                       initial < floor, initial > cap))
    return result


def recover_rows(scores, w, lower, upper, budget, mask, target):
    """Recover disputed rows in one packed transfer each way, including pre-storage faces."""
    width = scores.shape[-1]
    row_ids = mask.squeeze(-1)
    packed = torch.cat((scores, w, lower, upper, budget), -1)[row_ids]
    if width == 2:
        # A score gap > 1 exceeds the entire box width. Fill the higher
        # scoring node up to its cap before giving the other any excess over
        # its floor. The true target is between the aggregate endpoints here.
        s, _, lo, hi = packed[..., :4*width].split(width, -1)
        strong = s >= s.flip((-1,))
        corner = torch.where(strong, hi, lo)
        threshold = mass_parts(corner)
        total = target[0][row_ids], target[1][row_ids]
        below, above = less(total, threshold), less(threshold, total)
        difference, error = two_sum(total[0], -corner.flip((-1,)))
        remainder = difference + (error + total[1])
        interior = torch.where(strong, below, above) & (lo < hi)
        values = torch.where(interior, remainder, corner)
        lf, uf = (~strong & ~interior & (lo < hi)), (strong & ~interior & (lo < hi))
        uncertain_tail = (interior & (error != 0) & (total[1] != 0)).any(-1)
        exact = ((s[..., 0] - s[..., 1]).abs() <= 1) | uncertain_tail
        if not bool(exact.any()):
            return row_ids, values, lf, uf
        source = packed[exact]
    else:
        source = packed
    results = [_recover(row, width, scores.dtype) for row in source.detach().cpu().tolist()]
    c, low, high = scores.new_tensor(results).split(width, -1)
    if width == 2:
        values[exact], lf[exact], uf[exact] = c, low.bool(), high.bool()
        return row_ids, values, lf, uf
    return row_ids, c, low.bool(), high.bool()
