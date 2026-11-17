"""Received-bound role allocation from the fixed Tide region definitions.

Candidate axes are last; all other axes are independent regional calls.
Temperature is one and scores already include any fixed threshold.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .attention import _weighted_softmax


def _first_min(a: Tensor, b: Tensor) -> Tensor:
    return torch.where(a <= b, a, b)


def _first_max(a: Tensor, b: Tensor) -> Tensor:
    return torch.where(a >= b, a, b)


def _response(scores, multiplier, lower, upper, kind):
    x = scores - multiplier
    if kind == "shannon":
        # Compare with the positive cap in log space before exponentiation.
        log_cap = torch.where(upper > 0, upper, 1).log()
        x = torch.exp(torch.minimum(x - 1, log_cap))
    return torch.where(x <= lower, lower, torch.where(x >= upper, upper, x))


def _save_solution(ctx, c, scores, mu, lower, upper, low_active, high_active, kind, high_total):
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
    ctx.save_for_backward(slope, lower_face, upper_face, fixed, fixed_mu,
                          low_active, high_active, scores, mu, lower, upper)
    return c


class _Constrained(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, mu, lower, upper, low_total, high_total, kind):
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
            return _save_solution(ctx, c, scores, mu, lower, upper, low_active, high_active, kind, high_total)
        free = lower < upper
        # Centering is legal on the fixed-total solve and prevents cancellation
        # when every score has the same large common offset.
        center = torch.where(free, scores, -torch.inf).amax(-1, keepdim=True)
        center = torch.where(free.any(-1, keepdim=True), center, 0)
        shifted = scores - center
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
        for _ in range(64 if scores.dtype == torch.float64 else 40):
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
        # Reconstruct the selected smooth face from its residual mass. This
        # avoids subtracting nearly equal score/multiplier values for weak
        # receiving masses; the solve still decides the active coordinates.
        inside = (c > lower) & (c < upper)
        count = inside.sum(-1, keepdim=True).clamp_min(1)
        remainder = target - torch.where(inside, 0, c).sum(-1, keepdim=True)
        if kind == "quad":
            average = torch.where(inside, shifted, 0).sum(-1, keepdim=True) / count
            reconstructed = shifted - average + remainder / count
        else:
            logits = torch.where(inside, shifted, -torch.inf)
            logits = torch.where(inside.any(-1, keepdim=True), logits, 0)
            reconstructed = remainder * torch.softmax(logits, -1)
        c = torch.maximum(lower, torch.minimum(upper, torch.where(inside, reconstructed, c)))
        c = torch.where(at_lower, lower, torch.where(at_upper, upper, c))
        c = torch.where(active, c, c0)
        tolerance = 32 * torch.finfo(scores.dtype).eps * target.abs().clamp_min(torch.finfo(scores.dtype).tiny)
        if bool((active & ((c.sum(-1, keepdim=True) - target).abs() > tolerance)).any()):
            raise ArithmeticError("allocation multiplier did not meet the total constraint")
        return _save_solution(ctx, c, scores, mu, lower, upper, low_active, high_active, kind, high_total)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        slope, lower_face, upper_face, fixed, fixed_mu, low_active, high_active = ctx.saved_tensors[:7]
        total_slope = slope.sum(-1, keepdim=True)
        equality = (low_active | high_active) & (total_slope > 0)
        average = (slope * grad).sum(-1, keepdim=True) / torch.where(total_slope > 0, total_slope, 1)
        average = torch.where(equality, average, 0)
        adjusted = grad - average
        return (slope * adjusted, torch.where(fixed_mu, adjusted, 0),
                torch.where(lower_face | (fixed & ~fixed_mu), adjusted, 0),
                torch.where(upper_face, adjusted, 0),
                torch.where(low_active, average, 0), torch.where(high_active, average, 0), None)


def _as_input(value, scores):
    if isinstance(value, Tensor):
        if value.device != scores.device or value.dtype != scores.dtype:
            raise ValueError("allocation inputs must share scores' dtype and device")
        return value
    return scores.new_tensor(value)


def pool(scores: Tensor, mu: Tensor) -> Tensor:
    """Total-scaled Attention: min(1,sum(mu)) * softmax(scores; prior=mu).

    There is no per-node mu cap. At sum(mu)=1 the first min branch is the
    constant one. At an all-zero prior the declared VJP is zero.
    """
    if scores.dtype not in {torch.float32, torch.float64} or scores.ndim < 1:
        raise TypeError("allocation requires FP32/FP64 and a candidate axis")
    scores, mu = torch.broadcast_tensors(scores, _as_input(mu, scores))
    if bool((~torch.isfinite(scores) | ~torch.isfinite(mu) | (mu < 0) | (mu > 1)).any()):
        raise ValueError("pool requires finite scores and mu in [0,1]")
    p = _weighted_softmax(scores, mu, torch.ones((), device=scores.device, dtype=torch.bool))
    mass = mu.sum(-1, keepdim=True)
    result = _first_min(torch.ones_like(mass), mass) * p
    if bool(((mu > 0) & (result == 0)).any()):
        raise ArithmeticError("pool underflow would erase a positive sender")
    return result


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
    scores, w, mu, lower, upper = torch.broadcast_tensors(
        scores, *(_as_input(x, scores) for x in (w, mu, lower, upper)))
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
    return _Constrained.apply(scores, mu, lower, upper, low_total, high_total, kind)
