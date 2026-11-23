"""Received-bound role allocation from the fixed Tide region definitions.

Candidate axes are last; all other axes are independent regional calls.
Temperature is one and scores already include any fixed threshold.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from . import quad
from .pool_support import store_pool
from .scaled_arithmetic import binary_parts, compensated_sum as _compensated_sum, two_sum as _two_sum
from .weighted_mean import _difference_parts, _reduce_parts, _restore, _sum_parts
from .shannon_support import FLOOR, refine_rows


def _average_parts(weights, values):
    """Keep the scale of a weighted average until its composite VJP consumes it.

    The mean itself may be too small to store while a later division by a weak
    prior restores a representable derivative. A large common cotangent must
    likewise not overflow the numerator before normalization.
    """
    peak = weights.amax(-1, keepdim=True)
    peak = torch.where(peak > 0, peak, 1)
    total = (weights / peak).sum(-1, keepdim=True)
    total = torch.where(total > 0, total, 1)
    wm, we = binary_parts(weights)
    pm, pe = binary_parts(peak)
    tm, te = binary_parts(total)
    vm, ve = binary_parts(values)
    return _sum_parts(((wm / pm) / tm) * vm, we - pe - te + ve, -1)


def _exp_product_parts(log_scale, fraction, exponent):
    """Combine a log scale with binary parts without materializing exp(scale).

    Exponents beyond +/-16384 cannot yield a representable FP32/FP64 result
    with these finite cotangents. The bound avoids float-to-int overflow, not
    clipping of the returned value: restoration still yields Inf or zero.
    """
    power = torch.floor(log_scale / math.log(2)).clamp(-16384, 16384)
    ordinary = torch.isfinite(log_scale) & (log_scale.abs() < 8192)
    remainder = torch.where(ordinary, log_scale - power * math.log(2), 0)
    return fraction * remainder.exp(), exponent + power.to(torch.int32)


def _finite_vjp(label, values, needs):
    required = [torch.isfinite(value).all() for value, need in zip(values, needs) if need]
    if required and not bool(torch.stack(required).all()):
        raise ArithmeticError(f"{label} VJP exceeds the requested dtype range")


def _first_min(a: Tensor, b: Tensor) -> Tensor:
    return torch.where(a <= b, a, b)


def _first_max(a: Tensor, b: Tensor) -> Tensor:
    return torch.where(a >= b, a, b)


def _response(scores, multiplier, lower, upper, kind):
    x = scores - multiplier
    if kind == "shannon":
        # Compare with the positive cap in log space before exponentiation.
        log_cap = torch.where(upper > 0, upper, 1).log()
        exponent = x - 1
        # exp(log(cap)) can round below cap. Preserve the selected upper face
        # exactly so reconstruction and the VJP do not treat it as interior.
        x = torch.where(exponent >= log_cap, upper,
                        torch.exp(torch.minimum(exponent, log_cap)))
    return torch.where(x <= lower, lower, torch.where(x >= upper, upper, x))


def _save_solution(ctx, c, scores, mu, lower, upper, low_active, high_active, kind, high_total, originals,
                   solved_faces=None):
    refine = None
    if kind == "shannon":
        floor, tail = _compensated_sum(lower)
        remaining, error = _two_sum(high_total, -floor)
        has_positive_room = remaining + (error - tail) > 0
        if ctx.floor_policy:
            sensitive = (lower < upper) & (c < torch.finfo(c.dtype).tiny) & has_positive_room
            refine = (sensitive | ~torch.isfinite(c)).any(-1, keepdim=True)
            if ctx.unresolved is not None:
                refine = refine | ctx.unresolved
        elif bool(((upper > 0) & (c == 0) & has_positive_room).any()):
            raise ArithmeticError("Shannon allocation underflow would erase a positive sender")
    fixed = lower == upper
    fixed_mu = fixed & (upper == mu)
    lower_face = ~fixed & (c <= lower)
    upper_face = ~fixed & ~lower_face & (c >= upper)
    if solved_faces is not None:
        selected, selected_lower, selected_upper = solved_faces
        lower_face = torch.where(selected, selected_lower, lower_face)
        upper_face = torch.where(selected, selected_upper, upper_face)
    if ctx.quad_exact is not None:
        exact = ctx.quad_exact
        c = exact.replace(c, exact.values)
        lower_face = exact.replace(lower_face, exact.lower)
        upper_face = exact.replace(upper_face, exact.upper)
        low_active = exact.replace(low_active, exact.choices[4])
        high_active = exact.replace(high_active, exact.choices[5])
        ctx.quad_exact = None
    inside = ~fixed & ~lower_face & ~upper_face
    slope = inside.to(c.dtype) if kind == "quad" else torch.where(inside, c, 0)
    extended_slope = ()
    if refine is not None and bool(refine.any()):
        rows = refine.reshape(-1)
        values, mantissa, exponent, lf, uf, la, ua = refine_rows(
            scores, lower, upper, originals[4], high_total, rows)

        def merge(base, replacement):
            expanded = base.expand(*c.shape[:-1], replacement.shape[-1])
            flat = expanded.reshape(-1, replacement.shape[-1]).clone()
            flat[rows] = replacement
            return flat.reshape(expanded.shape)

        c = merge(c, values)
        lower_face, upper_face = merge(lower_face, lf), merge(upper_face, uf)
        low_active, high_active = merge(low_active, la), merge(high_active, ua)
        sm, se = binary_parts(slope)
        extended_slope = merge(sm, mantissa), merge(se, exponent)
    # Retain source versions as well as the face description used by the VJP.
    ctx.source_shapes = tuple(value.shape for value in originals)
    ctx.kind = kind
    ctx.save_for_backward(slope, lower_face, upper_face, fixed, fixed_mu,
                          low_active, high_active, *originals, *extended_slope)
    if ctx.return_lower_face:
        ctx.mark_non_differentiable(lower_face)
        return c, lower_face
    return c


def _quad_face_bounds(shifted, lower, upper, target, c):
    """Check the final clamped face's multiplier bounds, in the input dtype.

    For interior set I and remaining mass R, lambda solves
    sum_I(shifted-lambda)=R. Lower faces require lambda>=L=max(shifted-lower),
    upper faces lambda<=U=min(shifted-upper). Compare the corresponding sums
    with R directly: constructing and rounding lambda can erase a tiny gap.
    Fixed coordinates constrain mass, not lambda; an empty I needs L<=U.

    This supplements the total-residual check. It neither adds a support epsilon
    nor certifies arbitrary floating-point boundary cases; a rejected face uses
    the existing same-device safeguarded solver.
    """
    free = lower < upper
    inside = (c > lower) & (c < upper)
    low_face = free & (c <= lower)
    high_face = free & ~low_face & (c >= upper)
    low = torch.where(low_face, shifted - lower, -torch.inf).amax(-1, keepdim=True)
    high = torch.where(high_face, shifted - upper, torch.inf).amin(-1, keepdim=True)
    has_low, has_high = low_face.any(-1, keepdim=True), high_face.any(-1, keepdim=True)
    remainder = target - torch.where(inside, 0, c).sum(-1, keepdim=True)
    # Finite dummy origins avoid inf-inf for absent boundary sets.
    at_low = torch.where(inside, shifted - torch.where(has_low, low, 0), 0).sum(-1, keepdim=True)
    at_high = torch.where(inside, shifted - torch.where(has_high, high, 0), 0).sum(-1, keepdim=True)
    feasible = (~has_low | (at_low >= remainder)) & (~has_high | (at_high <= remainder))
    return torch.where(inside.any(-1, keepdim=True), feasible, low <= high)


def _quad_sorted(shifted, lower, upper, target):
    """Finite breakpoint solve: S(lambda)=intercept-count*lambda per interval.

    At shifted-upper a coordinate enters the interior (+1, +breakpoint);
    at shifted-lower it leaves (-1, -breakpoint). Fixed coordinates emit zero
    events. Prefix sums after the last equal breakpoint represent the whole
    tie group, so only positive-width intervals can supply a candidate.
    """
    free = lower < upper
    enter = torch.where(free, shifted - upper, 0)
    leave = torch.where(free, shifted - lower, 0)
    events, order = torch.cat((enter, leave), -1).sort(-1)
    signs = torch.cat((free.to(shifted.dtype), -free.to(shifted.dtype)), -1).gather(-1, order)
    count = signs.cumsum(-1)
    intercept = upper.sum(-1, keepdim=True) + (events * signs).cumsum(-1)
    following = torch.cat((events[..., 1:], torch.full_like(events[..., :1], torch.inf)), -1)
    candidates = (intercept - target) / torch.where(count > 0, count, 1)
    intervals = ((count > 0) & (events < following) & torch.isfinite(candidates)
                 & (candidates >= events) & (candidates <= following))
    index = intervals.to(torch.int32).argmax(-1, keepdim=True)
    multiplier = candidates.gather(-1, index)
    c = _response(shifted, multiplier, lower, upper, "quad")
    # Exact ties use their declared boundary face, including rounding in the
    # subtraction that constructs the breakpoint itself.
    c = torch.where(free & (multiplier == enter), upper, c)
    c = torch.where(free & (multiplier == leave), lower, c)
    # Reconstruct the selected face from its exact remaining mass. A rounded
    # multiplier beside a flat interval can otherwise invent a positive sender
    # even when the other coordinates already consume the entire target.
    inside = (c > lower) & (c < upper)
    count = inside.sum(-1, keepdim=True).clamp_min(1)
    remainder = target - torch.where(inside, 0, c).sum(-1, keepdim=True)
    origin = torch.where(inside, shifted, -torch.inf).amax(-1, keepdim=True)
    offsets = torch.where(inside, shifted - origin, 0)
    reconstructed = offsets - offsets.sum(-1, keepdim=True) / count + remainder / count
    # These exact residual equalities identify whole boundary faces. The local
    # score origin also protects genuine tiny mass among equal interior scores.
    inside_lower = torch.where(inside, lower, 0).sum(-1, keepdim=True)
    inside_upper = torch.where(inside, upper, 0).sum(-1, keepdim=True)
    reconstructed = torch.where(remainder == inside_lower, lower,
                                torch.where(remainder == inside_upper, upper, reconstructed))
    c = torch.maximum(lower, torch.minimum(upper, torch.where(inside, reconstructed, c)))
    tolerance = 32 * torch.finfo(shifted.dtype).eps * target.abs().clamp_min(torch.finfo(shifted.dtype).tiny)
    # A tiny cap at a large score can collapse its two floating breakpoints.
    # Its lost mass may hide inside a large total's residual tolerance.
    resolved = (~free | (torch.isfinite(enter) & torch.isfinite(leave) & (enter < leave))).all(-1, keepdim=True)
    valid = (resolved & intervals.any(-1, keepdim=True) & torch.isfinite(c).all(-1, keepdim=True)
             & _quad_face_bounds(shifted, lower, upper, target, c)
             & ((c.sum(-1, keepdim=True) - target).abs() <= tolerance))
    return c, valid


def _bisect(shifted, lower, upper, target, solve, kind):
    """Original safeguarded solve, also the numerical fallback for quad."""
    free = lower < upper
    sum_lower = lower.sum(-1, keepdim=True)
    if kind == "quad":
        left = torch.where(free, shifted - upper, torch.inf).amin(-1, keepdim=True)
        right = torch.where(free, shifted - lower, -torch.inf).amax(-1, keepdim=True)
    else:
        log_upper = torch.where(upper > 0, upper, 1).log()
        left = torch.where(free, shifted - 1 - log_upper, torch.inf).amin(-1, keepdim=True)
        residual = torch.where(target > sum_lower, target - sum_lower, 1)
        count = free.sum(-1, keepdim=True).clamp_min(1)
        right = (torch.where(free, shifted - 1, -torch.inf).amax(-1, keepdim=True)
                 - residual.log() + count.log())
    left, right = torch.where(solve, left, 0), torch.where(solve, right, 0)
    for _ in range(64 if shifted.dtype == torch.float64 else 40):
        middle = left * 0.5 + right * 0.5
        above = _response(shifted, middle, lower, upper, kind).sum(-1, keepdim=True) > target
        left, right = torch.where(above, middle, left), torch.where(above, right, middle)
    multiplier = left * 0.5 + right * 0.5
    c = _response(shifted, multiplier, lower, upper, kind)
    for _ in range(3):
        inside = (c > lower) & (c < upper)
        slope = (inside.to(c.dtype) if kind == "quad" else torch.where(inside, c, 0)).sum(-1, keepdim=True)
        correction = (c.sum(-1, keepdim=True) - target) / torch.where(slope > 0, slope, 1)
        multiplier = torch.maximum(left, torch.minimum(right, multiplier + correction))
        c = _response(shifted, multiplier, lower, upper, kind)
    if kind == "shannon":
        # Near a subnormal response step, the rounded midpoint can land on an
        # all-cap face with zero Newton slope. A bracket endpoint may already
        # meet the requested total exactly. Prefer only a strictly smaller
        # residual; no tolerance or support floor is introduced.
        for endpoint in (left, right):
            alternative = _response(shifted, endpoint, lower, upper, kind)
            better = ((alternative.sum(-1, keepdim=True) - target).abs()
                      < (c.sum(-1, keepdim=True) - target).abs())
            c = torch.where(better, alternative, c)
    # Preserve the original weak-mass reconstruction on the selected face.
    inside = (c > lower) & (c < upper)
    count = inside.sum(-1, keepdim=True).clamp_min(1)
    remainder = target - torch.where(inside, 0, c).sum(-1, keepdim=True)
    if kind == "quad":
        average = torch.where(inside, shifted, 0).sum(-1, keepdim=True) / count
        reconstructed = shifted - average + remainder / count
    else:
        logits = torch.where(inside, shifted, -torch.inf)
        logits = torch.where(inside.any(-1, keepdim=True), logits, 0)
        log_probability = logits - torch.logsumexp(logits, -1, keepdim=True)
        rm, re = binary_parts(remainder)
        # The residual total can exceed one. Its multiplication can rescue a
        # representable positive sender even when softmax alone rounds to zero.
        value_m, value_e = _exp_product_parts(log_probability, rm, re)
        reconstructed = _restore(value_m, value_e)
    return torch.maximum(lower, torch.minimum(upper, torch.where(inside, reconstructed, c)))


def _two_product(a, b):
    # Shannon's normalized factors are <= 1 and totals <= candidate count;
    # splitting these factors cannot overflow for any materializable tensor.
    splitter = 4097. if a.dtype == torch.float32 else 134217729.
    def split(x):
        scaled = splitter * x
        high = scaled - (scaled - x)
        return high, x - high
    ah, al = split(a)
    bh, bl = split(b)
    product = a * b
    error = ((ah * bh - product) + ah * bl + al * bh) + al * bl
    return product, error


def _shannon_bound_delta(amount, amount_correction, weights, denominator, correction, bound):
    """Sign of amount*weight-bound*sum(weight), retaining a small tail."""
    left, left_error = _two_product(amount, weights)
    right, right_error = _two_product(bound, denominator)
    difference, error = _two_sum(left, -right)
    return difference + (error + left_error + amount_correction * weights - right_error - bound * correction)


def _shannon_solve(scores, lower, upper, target, solve, refine_unresolved=False):
    """Finite bounded exponential water filling, with explicit solution faces.

    At an unconstrained normalization, clipping below-floor entries increases
    mass and clipping above-cap entries decreases it. If clipped mass is too
    large, every below-floor entry is certainly on its lower face at the root;
    the converse fixes upper faces. Thus each unfinished iteration removes at
    least one coordinate without guessing a face from a rounded multiplier.
    A mathematically interior sender may round to its cap in storage. Keep the
    selected face separately for its VJP; never force it inward or floor it.
    """
    free = (lower < upper) & solve
    fixed = torch.where(free, 0, lower)
    lower_face = torch.zeros_like(free)
    upper_face = torch.zeros_like(free)
    answer = fixed.clone()
    pending = solve.clone()
    unresolved = torch.zeros_like(solve)
    for _ in range(scores.shape[-1] + 1):
        if not bool(pending.any()):
            return answer, lower_face, upper_face, unresolved
        fixed_high, fixed_low = _compensated_sum(fixed)
        residual, error = _two_sum(target, -fixed_high)
        amount, amount_correction = _two_sum(residual, error - fixed_low)
        exhausted = pending & (amount <= 0)
        if refine_unresolved:
            unresolved = unresolved | exhausted
            pending = pending & ~exhausted
        elif bool(exhausted.any()):
            raise ArithmeticError("Shannon selected faces exhaust positive remaining mass")
        amount = torch.where(pending, amount, 1)
        amount_correction = torch.where(pending, amount_correction, 0)
        logits = torch.where(free, scores, -torch.inf)
        peak = logits.amax(-1, keepdim=True)
        peak = torch.where(free.any(-1, keepdim=True), peak, 0)
        logits = logits - peak
        weights = torch.where(free, logits.exp(), 0)
        denominator, correction = _compensated_sum(weights)
        denominator = torch.where(pending, denominator, 1)
        am, ae = binary_parts(amount)
        value_m, value_e = _exp_product_parts(logits, am, ae)
        value_m = value_m / denominator
        # Subtract the small denominator correction before exponent restore;
        # materializing 1+tail first would lose exactly the real failure's tail.
        relative_correction = (amount_correction / amount - correction / denominator)
        relative_correction = relative_correction / (1 + correction / denominator)
        value_m = value_m + value_m * relative_correction
        candidate = torch.where(free, _restore(value_m, value_e), fixed)
        lower_delta = _shannon_bound_delta(amount, amount_correction, weights, denominator, correction, lower)
        upper_delta = _shannon_bound_delta(amount, amount_correction, weights, denominator, correction, upper)
        low_tie = free & (lower > 0) & (candidate == lower) & (lower_delta == 0)
        high_tie = free & (candidate == upper) & (upper_delta == 0)
        below = free & ((candidate < lower) | ((candidate == lower) & (lower > 0) & (lower_delta < 0)))
        above = free & ((candidate > upper) | ((candidate == upper) & (upper_delta > 0)))
        clipped = torch.where(below | low_tie, lower, torch.where(above | high_tie, upper, candidate))
        total, tail = _compensated_sum(clipped)
        difference, error = _two_sum(total, -target)
        excess = difference + (error + tail)
        violations = (below | above).any(-1, keepdim=True)
        done = pending & (~violations | (excess == 0))
        answer = torch.where(done, clipped, answer)
        lower_face = lower_face | (done & (below | low_tie))
        upper_face = upper_face | (done & (above | high_tie))
        pending = pending & ~done
        bind_lower = pending & (excess > 0) & below
        bind_upper = pending & (excess < 0) & above
        progress = (bind_lower | bind_upper).any(-1, keepdim=True)
        if refine_unresolved:
            unresolved = unresolved | (pending & ~progress)
            pending = pending & progress
        elif bool((pending & ~progress).any()):
            raise ArithmeticError("Shannon bounded solve could not certify a remaining face")
        fixed = torch.where(bind_lower, lower, torch.where(bind_upper, upper, fixed))
        lower_face = lower_face | bind_lower
        upper_face = upper_face | bind_upper
        free = free & ~(bind_lower | bind_upper)
    if refine_unresolved:
        return answer, lower_face, upper_face, unresolved | pending
    raise ArithmeticError("Shannon bounded solve exceeded its finite face count")


def _shannon_centered_parts(weights, values, extended=()):
    """Center at a real cotangent before averaging, retaining small VJPs."""
    if extended:
        wm, we = extended
        largest = torch.where(wm > 0, we, -16384).amax(-1, keepdim=True)
        candidates = torch.where(we == largest, wm, 0)
        anchor_index = candidates.argmax(-1, keepdim=True)
        peak_m = wm.gather(-1, anchor_index)
        pm = torch.where(peak_m > 0, peak_m, .5)
        pe = torch.where(peak_m > 0, largest, 1)
        scaled = _restore(wm / pm, we - pe)
        total = scaled.sum(-1, keepdim=True)
    else:
        anchor_index = weights.argmax(-1, keepdim=True)
        peak = weights.amax(-1, keepdim=True)
        peak = torch.where(peak > 0, peak, 1)
        total = (weights / peak).sum(-1, keepdim=True)
        wm, we = binary_parts(weights)
        pm, pe = binary_parts(peak)
    tm, te = binary_parts(torch.where(total > 0, total, 1))
    factor_m, factor_e = (wm / pm) / tm, we - pe - te
    anchor = values.gather(-1, anchor_index)
    anchor_m, anchor_e = binary_parts(anchor)
    delta_m, delta_e = _difference_parts(values, anchor_m, anchor_e)
    mean_m, mean_e = _sum_parts(factor_m * delta_m, factor_e + delta_e, -1)
    average_m, average_e = _sum_parts(torch.stack((anchor_m.squeeze(-1), mean_m), -1),
                                      torch.stack((anchor_e.squeeze(-1), mean_e), -1), -1)
    centered_m, centered_e = _sum_parts(
        torch.stack((delta_m, -mean_m[..., None].expand_as(delta_m)), -1),
        torch.stack((delta_e, mean_e[..., None].expand_as(delta_e)), -1), -1)
    return average_m, average_e, centered_m, centered_e


class _Constrained(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, mu, lower, upper, low_total, high_total, kind, quad_inputs, return_lower_face):
        ctx.floor_policy = kind == FLOOR
        ctx.unresolved = None
        ctx.return_lower_face = return_lower_face
        ctx.quad_exact = None
        if ctx.floor_policy:
            kind = "shannon"
        originals = scores, mu, lower, upper, low_total, high_total
        shape = torch.broadcast_shapes(*(value.shape for value in originals))
        scores, mu, lower, upper = (value.expand(shape) for value in originals[:4])
        c0 = _response(scores, 0, lower, upper, kind)
        if quad_inputs is not None:
            totals, received, budget = quad_inputs
            initial = totals.initial if totals.initial is not None else quad.mass_parts(c0)
            ctx.quad_exact = totals.exact
            s0 = initial[0]
            low_active, high_active = quad.less(initial, totals.low), quad.less(totals.high, initial)
        else:
            s0 = c0.sum(-1, keepdim=True)
            low_active, high_active = s0 < low_total, s0 > high_total
        if kind == "shannon":
            total, tail = _compensated_sum(c0)
            low_difference, low_error = _two_sum(total, -low_total)
            high_difference, high_error = _two_sum(total, -high_total)
            low_active = low_difference + (low_error + tail) < 0
            high_active = high_difference + (high_error + tail) > 0
        active = low_active | high_active
        target = torch.where(low_active, low_total, torch.where(high_active, high_total, s0))
        if quad_inputs is not None:
            target_tail = torch.where(low_active, totals.low[1], totals.high[1])
            at_lower = ~quad.less(totals.floor, (target, target_tail))
            at_upper = ~quad.less((target, target_tail), totals.cap)
        else:
            sum_lower, sum_upper = lower.sum(-1, keepdim=True), upper.sum(-1, keepdim=True)
            at_lower, at_upper = target <= sum_lower, target >= sum_upper
        if kind == "shannon":
            floor, floor_tail = _compensated_sum(lower)
            cap, cap_tail = _compensated_sum(upper)
            floor_difference, floor_error = _two_sum(target, -floor)
            cap_difference, cap_error = _two_sum(target, -cap)
            at_lower = floor_difference + (floor_error - floor_tail) <= 0
            at_upper = cap_difference + (cap_error - cap_tail) >= 0
        solve = active & ~at_lower & ~at_upper
        if ctx.quad_exact is not None:
            solve = solve & ~ctx.quad_exact.mask
        if scores.shape[-1] == 1 or not bool(solve.any()):
            c = torch.where(active, target if scores.shape[-1] == 1
                            else torch.where(at_lower, lower, upper), c0)
            return _save_solution(ctx, c, scores, mu, lower, upper, low_active, high_active, kind, high_total, originals)
        free = lower < upper
        # Centering is legal on the fixed-total solve and prevents cancellation
        # when every score has the same large common offset.
        center = torch.where(free, scores, -torch.inf).amax(-1, keepdim=True)
        center = torch.where(free.any(-1, keepdim=True), center, 0)
        shifted = scores - center
        solved_faces = None
        if kind == "quad":
            c, valid = _quad_sorted(shifted, lower, upper, target)
            fallback = solve & ~valid
            if bool(fallback.any()):
                recovered = _bisect(shifted, lower, upper, target, fallback, kind)
                c = torch.where(fallback, recovered, c)
        else:
            # When the peak binds, re-center the remaining free scores from
            # their originals: a very large removed score can erase their gap.
            active_scores = scores if ctx.floor_policy else shifted
            c, selected_lower, selected_upper, ctx.unresolved = _shannon_solve(
                active_scores, lower, upper, target, solve, refine_unresolved=ctx.floor_policy)
            solved_faces = solve, selected_lower, selected_upper
        c = torch.where(at_lower, lower, torch.where(at_upper, upper, c))
        c = torch.where(active, c, c0)
        tolerance = 32 * torch.finfo(scores.dtype).eps * target.abs().clamp_min(torch.finfo(scores.dtype).tiny)
        if quad_inputs is not None:
            suspect = quad.disputed(scores, lower, upper, c, (target, target_tail), solve, tolerance)
            if bool(suspect.any()):
                rows, values, lf, uf = quad.recover_rows(scores, received, lower, upper, budget, suspect,
                                                         (target, target_tail))
                c = c.clone()
                c[rows] = values
                low_face, high_face = torch.zeros_like(c, dtype=torch.bool), torch.zeros_like(c, dtype=torch.bool)
                low_face[rows], high_face[rows] = lf, uf
                solved_faces = suspect, low_face, high_face
        else:
            missed = active & ((c.sum(-1, keepdim=True) - target).abs() > tolerance)
            if ctx.floor_policy:
                ctx.unresolved = missed if ctx.unresolved is None else ctx.unresolved | missed
            elif bool(missed.any()):
                raise ArithmeticError("allocation multiplier did not meet the total constraint")
        return _save_solution(ctx, c, scores, mu, lower, upper, low_active, high_active, kind, high_total, originals,
                              solved_faces)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, *unused):
        saved = ctx.saved_tensors
        slope, lower_face, upper_face, fixed, fixed_mu, low_active, high_active = saved[:7]
        extended = saved[13:]
        positive = extended[0] if extended else slope
        equality = (low_active | high_active) & (positive > 0).any(-1, keepdim=True)
        if ctx.kind == "shannon":
            am, ae, dm, de = _shannon_centered_parts(slope, grad, extended)
            gm, ge = binary_parts(grad)
            dm, de = torch.where(equality, dm, gm), torch.where(equality, de, ge)
            am, ae = torch.where(equality, am[..., None], 0), ae[..., None]
        else:
            am, ae = _average_parts(slope, grad)
            am = torch.where(equality, am[..., None], 0)
            ae = ae[..., None]
            dm, de = _difference_parts(grad, am, ae)
        sm, se = extended if extended else binary_parts(slope)
        score_m, score_e = sm * dm, se + de
        if ctx.source_shapes[0][-1] == 1:
            # One score source shared by every candidate is a common shift.
            # A fixed-total solve is invariant to it, including when separate
            # expanded adjoints would overflow before cancelling. Slack rows
            # retain their ordinary source derivative.
            score_m = torch.where(equality, 0, score_m)
        # Restore only after multiplying by the slope. In Shannon a tiny
        # slope can make a difference larger than finfo.max representable.
        # Broadcast users are summed before restoring each original root.
        parts = ((score_m, score_e), (torch.where(fixed_mu, dm, 0), de),
                 (torch.where(lower_face | (fixed & ~fixed_mu), dm, 0), de),
                 (torch.where(upper_face, dm, 0), de),
                 (torch.where(low_active, am, 0), ae),
                 (torch.where(high_active, am, 0), ae))
        result = tuple(_reduce_parts(m, e, shape) if need else None
                       for (m, e), shape, need in zip(parts, ctx.source_shapes, ctx.needs_input_grad[:6]))
        _finite_vjp("allocation", result, ctx.needs_input_grad[:6])
        return (*result, None, None, None)


def _as_input(value, scores):
    if isinstance(value, Tensor):
        if value.device != scores.device or value.dtype != scores.dtype:
            raise ValueError("allocation inputs must share scores' dtype and device")
        return value
    return scores.new_tensor(value)


class _Pool(torch.autograd.Function):
    """First-order composite VJP of min(1,M)*softmax(score; prior=mu).

    Keeping the output cotangent unscaled avoids losing M*g before the prior
    derivative. Positive-prior ratios and signed sums retain binary exponents;
    real zero priors retain their right derivative under either storage policy.
    """
    @staticmethod
    def forward(ctx, scores, mu, policy):
        original_scores, original_mu = scores, mu
        scores, mu = torch.broadcast_tensors(scores, mu)
        nonempty = (mu > 0).any(-1, keepdim=True)
        peak = mu.amax(-1, keepdim=True)
        peak = torch.where(nonempty, peak, 1)
        center = torch.where(mu > 0, scores, -torch.inf).amax(-1, keepdim=True)
        center = torch.where(nonempty, center, 0)
        relative = mu / peak
        logits = scores - center + torch.where(mu > 0, relative, 1).log()
        logits = torch.where(mu > 0, logits, -torch.inf)
        weighted_peak = torch.where(nonempty, logits.amax(-1, keepdim=True), 0)
        shifted = logits - weighted_peak
        normalizer = torch.logsumexp(torch.where(nonempty, shifted, 0), -1, keepdim=True)
        pm, pe = _exp_product_parts(shifted - normalizer, torch.ones_like(logits),
                                    torch.zeros_like(logits, dtype=torch.int32))
        pm = torch.where((mu > 0) & nonempty, pm, 0)
        total_m, total_e = _sum_parts(pm, pe, -1)
        pm = pm / torch.where(total_m > 0, total_m, 1)[..., None]
        pe = pe - total_e[..., None]
        probability_sum = _restore(total_m, total_e)[..., None]
        log_z = weighted_peak + normalizer + torch.where(nonempty, probability_sum, 1).log()
        mass = mu.sum(-1, keepdim=True)
        amount = torch.minimum(torch.ones_like(mass), mass)
        alpha_m, alpha_e = binary_parts(amount)
        result = _restore(alpha_m * pm, alpha_e + pe)
        result = store_pool(scores, mu, amount, result, policy)
        ctx.save_for_backward(original_scores, original_mu, pm, pe, mass, peak, center, log_z)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        original_scores, original_mu, pm, pe, mass, peak, center, log_z = ctx.saved_tensors
        scores, mu = torch.broadcast_tensors(original_scores, original_mu)
        gm, ge = binary_parts(grad)
        numerator_m, numerator_e = _sum_parts(pm * gm, pe + ge, -1)
        denominator_m, denominator_e = _sum_parts(pm, pe, -1)
        am = numerator_m / torch.where(denominator_m > 0, denominator_m, 1)
        ae = numerator_e - denominator_e
        am, ae = am[..., None], ae[..., None]
        dm, de = _difference_parts(grad, am, ae)
        amount = torch.minimum(torch.ones_like(mass), mass)
        alpha_m, alpha_e = binary_parts(amount)
        # Use the composite's factors, not the rounded stored sender. Even a
        # nonzero subnormal sender can lose bits needed by its finite VJP.
        cm, ce = alpha_m * pm, alpha_e + pe
        ds = dmu = None
        if ctx.needs_input_grad[0]:
            ds = _reduce_parts(cm * dm, ce + de, original_scores.shape)
        if ctx.needs_input_grad[1]:
            mm, me = binary_parts(torch.where(mu > 0, mu, 1))
            prior_m, prior_e = (cm / mm) * dm, ce - me + de
            log_amount = torch.where(amount > 0, amount, 1).log()
            log_ratio = scores - center - log_z + log_amount - peak.log()
            zero_m, zero_e = _exp_product_parts(log_ratio, dm, de)
            prior_m = torch.where(mu > 0, prior_m, zero_m)
            prior_e = torch.where(mu > 0, prior_e, zero_e)
            # At M=1 the written first-min branch is constant, unlike KV's
            # clamp(max=1) derivative. Empty pool rows keep their zero VJP.
            mass_m = torch.where((mass > 0) & (mass < 1), am, 0).expand_as(prior_m)
            mass_e = ae.expand_as(prior_e)
            prior_m = torch.where(mass > 0, prior_m, 0)
            combined_m, combined_e = _sum_parts(torch.stack((prior_m, mass_m), -1),
                                               torch.stack((prior_e, mass_e), -1), -1)
            dmu = _reduce_parts(combined_m, combined_e, original_mu.shape)
        required = [value for value in (ds, dmu) if value is not None]
        _finite_vjp("pool", required, [True] * len(required))
        return ds, dmu, None


def pool(scores: Tensor, mu: Tensor, *, support_policy: str = "strict") -> Tensor:
    """Total-scaled Attention: min(1,sum(mu)) * softmax(scores; prior=mu).

    There is no per-node mu cap. At sum(mu)=1 the first min branch is the
    constant one. At an all-zero prior the declared VJP is zero. positive-floor-v1
    keeps underflowed positive senders active and differentiates the unrounded weights.
    """
    if scores.dtype not in {torch.float32, torch.float64} or scores.ndim < 1:
        raise TypeError("allocation requires FP32/FP64 and a candidate axis")
    if support_policy not in ("strict", FLOOR):
        raise ValueError("unknown pool support policy")
    mu = _as_input(mu, scores)
    expanded_scores, expanded_mu = torch.broadcast_tensors(scores, mu)
    if bool((~torch.isfinite(expanded_scores) | ~torch.isfinite(expanded_mu)
             | (expanded_mu < 0) | (expanded_mu > 1)).any()):
        raise ValueError("pool requires finite scores and mu in [0,1]")
    if expanded_scores.shape[-1] == 0:
        return expanded_scores * 0 + expanded_mu * 0
    return _Pool.apply(scores, mu, support_policy)


def allocate(scores: Tensor, w: Tensor, mu: Tensor, lower: Tensor, upper: Tensor,
             k: float | Tensor, kind: str = "quad", *, support_policy: str = "strict",
             return_lower_face: bool = False):
    """Solve role-bounded quad/Shannon allocation with the received lower bound.

    For quad, return_lower_face also returns the non-fixed lower-face mask used
    by the VJP, evaluated before a disputed solution is rounded to storage.

    b=min(1,sum(w)), U=min(k,sum(upper)), L=max(sum(lower),min(b,U)).
    Scores, w, mu and bounds broadcast along the candidate-last layout; k
    broadcasts over the batch axes (an optional final singleton is accepted).
    k is finite and >=1. Empty candidate sets return connected empty tensors.

    Boundary VJP: min/max ties take the written first branch; S(0) exactly at
    L/U is slack; clip endpoints use the boundary face. Fixed-through entries
    lower=upper=mu differentiate through mu; other equal bounds through lower.
    With no interior coordinates, backward uses that fixed face, without
    division by zero. These are declared first-order conventions at corners.
    Strict Shannon/pool raise when a positive sender underflows. kind='pool' dispatches to
    pool(scores,mu,support_policy=...), whose contract has no role bounds or k. For Shannon,
    support_policy="positive-floor-v1" refines extreme rows on the host, stores
    lost positive senders at the least subnormal, and differentiates the
    unrounded free responses. The default strict policy rejects support loss.
    """
    if return_lower_face and kind != "quad":
        raise ValueError("return_lower_face requires quad allocation")
    if kind == "pool":
        return pool(scores, mu, support_policy=support_policy)
    if support_policy not in ("strict", FLOOR) or (support_policy == FLOOR and kind != "shannon"):
        raise ValueError("support_policy must be strict or positive-floor-v1 for Shannon")
    if kind not in {"quad", "shannon"}:
        raise ValueError(f"unknown allocation kind: {kind}")
    if scores.dtype not in {torch.float32, torch.float64} or scores.ndim < 1:
        raise TypeError("allocation requires FP32/FP64 and a candidate axis")
    originals = (scores, *(_as_input(x, scores) for x in (w, mu, lower, upper)))
    scores, w, mu, lower, upper = torch.broadcast_tensors(*originals)
    budget = _as_input(k, scores)
    if budget.ndim == scores.ndim and budget.shape[-1] == 1:
        budget = budget.squeeze(-1)
    budget = torch.broadcast_to(budget, scores.shape[:-1]).unsqueeze(-1)
    legal = ((w >= 0) & (w <= mu) & (mu <= 1) & (lower >= 0)
             & (lower <= upper) & (upper <= mu) & ((w > 0) | (mu == 0)))
    finite = torch.isfinite(scores) & torch.isfinite(w) & torch.isfinite(mu) & torch.isfinite(lower) & torch.isfinite(upper)
    if bool((~legal | ~finite).any()) or bool((~torch.isfinite(budget) | (budget < 1)).any()):
        raise ValueError("invalid received weights, role bounds or budget")
    if scores.shape[-1] == 0:
        empty = scores * 0 + w * 0 + mu * 0 + lower * 0 + upper * 0 + budget * 0
        return (empty, torch.zeros_like(empty, dtype=torch.bool)) if return_lower_face else empty
    quad_inputs = None
    if kind == "quad" and scores.shape[-1] > 1:
        totals = quad.total_bounds(scores, w, lower, upper, budget)
        low_total, high_total = totals.low[0], totals.high[0]
        quad_inputs = totals, w, budget
    else:
        sum_lower, sum_upper = lower.sum(-1, keepdim=True), upper.sum(-1, keepdim=True)
        high_total = _first_min(budget, sum_upper)
        if bool((sum_lower > high_total).any()):
            raise ValueError("role floors exceed the available budget")
        mass = w.sum(-1, keepdim=True)
        received = _first_min(torch.ones_like(mass), mass)
        low_total = _first_max(sum_lower, _first_min(received, high_total))
    return _Constrained.apply(originals[0], originals[2], originals[3], originals[4],
                              low_total, high_total, FLOOR if support_policy == FLOOR else kind,
                              quad_inputs, return_lower_face)
