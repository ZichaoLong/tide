"""Received-bound role allocation from the fixed Tide region definitions.

Candidate axes are last; all other axes are independent regional calls.
Temperature is one and scores already include any fixed threshold.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .scaled_arithmetic import binary_parts
from .weighted_mean import _difference_parts, _reduce_parts, _restore, _sum_parts


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


def _save_solution(ctx, c, scores, mu, lower, upper, low_active, high_active, kind, high_total, originals):
    if kind == "shannon" and bool(((upper > 0) & (c == 0)
                                  & (high_total > lower.sum(-1, keepdim=True))).any()):
        raise ArithmeticError("Shannon allocation underflow would erase a positive sender")
    fixed = lower == upper
    fixed_mu = fixed & (upper == mu)
    lower_face = ~fixed & (c <= lower)
    upper_face = ~fixed & ~lower_face & (c >= upper)
    inside = ~fixed & ~lower_face & ~upper_face
    slope = inside.to(c.dtype) if kind == "quad" else torch.where(inside, c, 0)
    # Retain source versions as well as the face description used by the VJP.
    ctx.source_shapes = tuple(value.shape for value in originals)
    ctx.save_for_backward(slope, lower_face, upper_face, fixed, fixed_mu,
                          low_active, high_active, *originals)
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


class _Constrained(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, mu, lower, upper, low_total, high_total, kind):
        originals = scores, mu, lower, upper, low_total, high_total
        shape = torch.broadcast_shapes(*(value.shape for value in originals))
        scores, mu, lower, upper = (value.expand(shape) for value in originals[:4])
        c0 = _response(scores, 0, lower, upper, kind)
        s0 = c0.sum(-1, keepdim=True)
        low_active, high_active = s0 < low_total, s0 > high_total
        active = low_active | high_active
        target = torch.where(low_active, low_total, torch.where(high_active, high_total, s0))
        sum_lower, sum_upper = lower.sum(-1, keepdim=True), upper.sum(-1, keepdim=True)
        at_lower, at_upper = target <= sum_lower, target >= sum_upper
        solve = active & ~at_lower & ~at_upper
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
        if kind == "quad":
            c, valid = _quad_sorted(shifted, lower, upper, target)
            fallback = solve & ~valid
            if bool(fallback.any()):
                recovered = _bisect(shifted, lower, upper, target, fallback, kind)
                c = torch.where(fallback, recovered, c)
        else:
            c = _bisect(shifted, lower, upper, target, solve, kind)
        c = torch.where(at_lower, lower, torch.where(at_upper, upper, c))
        c = torch.where(active, c, c0)
        tolerance = 32 * torch.finfo(scores.dtype).eps * target.abs().clamp_min(torch.finfo(scores.dtype).tiny)
        if bool((active & ((c.sum(-1, keepdim=True) - target).abs() > tolerance)).any()):
            raise ArithmeticError("allocation multiplier did not meet the total constraint")
        return _save_solution(ctx, c, scores, mu, lower, upper, low_active, high_active, kind, high_total, originals)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        slope, lower_face, upper_face, fixed, fixed_mu, low_active, high_active = ctx.saved_tensors[:7]
        equality = (low_active | high_active) & (slope > 0).any(-1, keepdim=True)
        am, ae = _average_parts(slope, grad)
        am = torch.where(equality, am[..., None], 0)
        ae = ae[..., None]
        dm, de = _difference_parts(grad, am, ae)
        sm, se = binary_parts(slope)
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
        return (*result, None)


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
    real zero priors retain their right derivative. No sender floor is added.
    """
    @staticmethod
    def forward(ctx, scores, mu):
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
        if bool(((mu > 0) & (result == 0)).any()):
            raise ArithmeticError("pool underflow would erase a positive sender")
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
        return ds, dmu


def pool(scores: Tensor, mu: Tensor) -> Tensor:
    """Total-scaled Attention: min(1,sum(mu)) * softmax(scores; prior=mu).

    There is no per-node mu cap. At sum(mu)=1 the first min branch is the
    constant one. At an all-zero prior the declared VJP is zero.
    """
    if scores.dtype not in {torch.float32, torch.float64} or scores.ndim < 1:
        raise TypeError("allocation requires FP32/FP64 and a candidate axis")
    mu = _as_input(mu, scores)
    expanded_scores, expanded_mu = torch.broadcast_tensors(scores, mu)
    if bool((~torch.isfinite(expanded_scores) | ~torch.isfinite(expanded_mu)
             | (expanded_mu < 0) | (expanded_mu > 1)).any()):
        raise ValueError("pool requires finite scores and mu in [0,1]")
    if expanded_scores.shape[-1] == 0:
        return expanded_scores * 0 + expanded_mu * 0
    return _Pool.apply(scores, mu)


def allocate(scores: Tensor, w: Tensor, mu: Tensor, lower: Tensor, upper: Tensor,
             k: float | Tensor, kind: str = "quad") -> Tensor:
    """Solve role-bounded quad/Shannon allocation with the received lower bound.

    b=min(1,sum(w)), U=min(k,sum(upper)), L=max(sum(lower),min(b,U)).
    Scores, w, mu and bounds broadcast along the candidate-last layout; k
    broadcasts over the batch axes (an optional final singleton is accepted).
    k is finite and >=1. Empty candidate sets return connected empty tensors.

    Boundary VJP: min/max ties take the written first branch; S(0) exactly at
    L/U is slack; clip endpoints use the boundary face. Fixed-through entries
    lower=upper=mu differentiate through mu; other equal bounds through lower.
    With no interior coordinates, backward uses that fixed face, without
    division by zero. Unrepresentable positive Shannon/pool senders fail
    explicitly instead of silently becoming zero. These are declared first-order conventions, not claims
    of classical differentiability at a corner. kind='pool' dispatches to
    pool(scores,mu), whose contract has no role bounds or k.
    """
    if kind == "pool":
        return pool(scores, mu)
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
    sum_lower, sum_upper = lower.sum(-1, keepdim=True), upper.sum(-1, keepdim=True)
    high_total = _first_min(budget, sum_upper)
    if bool((sum_lower > high_total).any()):
        raise ValueError("role floors exceed the available budget")
    mass = w.sum(-1, keepdim=True)
    received = _first_min(torch.ones_like(mass), mass)
    low_total = _first_max(sum_lower, _first_min(received, high_total))
    if scores.shape[-1] == 0:
        return scores * 0 + w * 0 + mu * 0 + lower * 0 + upper * 0 + budget * 0
    return _Constrained.apply(originals[0], originals[2], originals[3], originals[4],
                              low_total, high_total, kind)
