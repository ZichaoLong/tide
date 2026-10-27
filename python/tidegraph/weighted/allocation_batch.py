"""Allocation §4.1/6: simultaneous exact-map forward solves on complete rows.

Rows have the same real candidate count; no padded candidate changes support.
Newton iterates are forward numerics only. The declared VJP is a separate layer.
"""
from dataclasses import dataclass
import torch


ZERO, CAPS, SLACK, LOWER, UPPER, EQUALITY = range(6)
NAMES = ("zero", "caps", "slack", "lower", "upper", "equality")


@dataclass(frozen=True)
class BatchAllocation:
    controls: torch.Tensor
    multiplier: torch.Tensor
    branch: torch.Tensor
    response_input: torch.Tensor
    free: torch.Tensor
    residual: torch.Tensor


def quad(t, caps, target, temperature):
    shift = t.max(1).values
    centered, tau = t-shift[:, None], temperature[:, None]
    points, order = torch.cat((centered-tau*caps, centered), 1).sort(dim=1, stable=True)
    n = t.shape[1]
    enter, index = order < n, order.remainder(n)
    count = torch.where(enter, 1, -1).cumsum(1)
    selected = centered.gather(1, index)/tau
    delta = torch.where(enter, selected-caps.gather(1, index), -selected)
    intercept = caps.sum(1)[:, None]+delta.cumsum(1)
    totals = intercept-count*points/tau
    right = torch.cat((points[:, 1:], points[:, -1:]), 1)
    eligible = (count > 0) & (points < right) & (totals >= target[:, None])
    chosen = torch.where(eligible, torch.arange(2*n, device=t.device), -1).max(1).values
    if (chosen < 0).any():
        raise ArithmeticError("quadratic batch lost feasible breakpoint interval")
    row = torch.arange(len(t), device=t.device)
    lam = (intercept[row, chosen]-target)*temperature/count[row, chosen]
    result = torch.minimum(((centered-lam[:, None])/tau).clamp_min(0), caps)
    free = (result > 0) & (result < caps)
    capped = (result == caps) & ~free
    corrected = (torch.where(free, centered, 0).sum(1)+temperature*(torch.where(capped, caps, 0).sum(1)-target))/free.sum(1).clamp_min(1)
    lam = torch.where(free.any(1), corrected, lam)
    result = torch.minimum(((centered-lam[:, None])/tau).clamp_min(0), caps)
    return result, lam+shift, centered-lam[:, None]


def entropy(t, caps, target, temperature, rule):
    shift, tau = t.max(1).values, temperature[:, None]
    centered = t-shift[:, None]
    points = centered-rule.inverse_bracket((target/caps.sum(1))[:, None], caps, tau)
    positive = caps > 0
    low = points.masked_fill(~positive, torch.inf).min(1).values
    high = points.masked_fill(~positive, -torch.inf).max(1).values
    lam = low+(high-low)/2
    live = torch.ones(len(t), dtype=torch.bool, device=t.device)
    for _ in range(80):
        controls = rule.response(centered-lam[:, None], caps, tau)
        total = controls.sum(1)
        low = torch.where(live & (total > target), lam, low)
        high = torch.where(live & (total <= target), lam, high)
        slope = rule.derivatives(centered-lam[:, None], caps, tau, controls)[0].sum(1)
        divisor = torch.where(slope > 0, slope, torch.ones_like(slope))
        newton = torch.where(slope > 0, lam+(total-target)/divisor, high)
        proposal = torch.where((low < newton) & (newton < high), newton, low+(high-low)/2)
        advancing = live & (proposal != lam) & (low != high)
        lam = torch.where(advancing, proposal, lam)
        live = advancing
        if not live.any():
            break
    choices = torch.stack((low, lam, high), 1)
    values = rule.response(centered[:, None]-choices[..., None], caps[:, None], tau[:, None])
    errors = (values.sum(-1)-target[:, None]).abs()
    chosen = errors.argmin(1)
    lam = choices[torch.arange(len(t), device=t.device), chosen]
    x = centered-lam[:, None]
    return rule.response(x, caps, tau), lam+shift, x


@torch.no_grad()
def solve(scores, caps, lower, upper, temperature, rule, shift=None):
    """Numeric B×N rows, B scalar bounds/temperature, and optional B shifts."""
    if (scores.ndim != 2 or min(scores.shape) < 1 or scores.dtype not in {torch.float32, torch.float64}
            or caps.shape != scores.shape or any(x.shape != (len(scores),) for x in (lower, upper, temperature))):
        raise ValueError("allocation batch requires complete nonempty rows and scalar row bounds")
    tensors = (caps, lower, upper, temperature)+(tuple() if shift is None else (shift,))
    if any((x.dtype, x.device) != (scores.dtype, scores.device) for x in tensors):
        raise ValueError("allocation batch tensor metadata mismatch")
    if shift is not None and shift.shape != lower.shape:
        raise ValueError("allocation batch shift shape mismatch")
    if not all(torch.isfinite(x).all() for x in (scores, *tensors)):
        raise ValueError("nonfinite allocation batch input")
    if (((caps < 0) | (caps > 1)).any() or (lower < 0).any() or (upper < lower).any()
            or (lower > caps.sum(1)).any() or (temperature <= 0).any()):
        raise ValueError("infeasible allocation batch bounds or temperature")
    t = scores if shift is None else scores-shift[:, None]
    if not torch.isfinite(t).all():
        raise ArithmeticError("shifted batch allocation scores exceed requested dtype range")
    mass = caps.sum(1)
    zero = (mass == 0) | (upper == 0)
    full = ~zero & (lower == mass)
    normal = ~zero & ~full
    branch = torch.full((len(t),), SLACK, dtype=torch.int64, device=t.device)
    result = torch.zeros_like(t)
    lam, residual = torch.zeros_like(lower), torch.zeros_like(lower)
    response = t.clone()
    branch[zero], branch[full] = ZERO, CAPS
    result[full] = caps[full]
    if normal.any():
        ids = normal.nonzero().flatten()
        tn, cn, tau = t[ids], caps[ids], temperature[ids]
        controls = rule.response(tn, cn, tau[:, None]); total = controls.sum(1)
        bn = torch.where(total < lower[ids], LOWER, torch.where(total > upper[ids], UPPER, SLACK))
        bn = torch.where(lower[ids] == upper[ids], EQUALITY, bn)
        branch[ids], result[ids] = bn, controls
        constrained = bn != SLACK
        if constrained.any():
            rows = ids[constrained]
            target = torch.where((branch[rows] == LOWER) | (branch[rows] == EQUALITY), lower[rows], upper[rows])
            if rule.kind == "quad":
                values, multipliers, x = quad(t[rows], caps[rows], target, temperature[rows])
            else:
                values, multipliers, x = entropy(t[rows], caps[rows], target, temperature[rows], rule)
            result[rows], lam[rows], response[rows] = values, multipliers, x
        # Scalar quad's residual uses original response coordinates; entropy
        # keeps the centered response point to avoid shifting it twice.
        x = t[ids]-lam[ids, None] if rule.kind == "quad" else response[ids]
        errors = (result[ids]-rule.response(x, caps[ids], temperature[ids, None])).abs().max(1).values
        residual[ids] = torch.stack((errors, (lower[ids]-result[ids].sum(1)).clamp_min(0),
                                    (result[ids].sum(1)-upper[ids]).clamp_min(0)), 1).max(1).values
        tolerance = 64*torch.finfo(t.dtype).eps*t.shape[1]
        if not torch.isfinite(residual).all() or (residual > tolerance).any():
            raise ArithmeticError("allocation batch residual exceeds dtype tolerance")
        if rule.kind != "quad" and (rule.positive_support(x, caps[ids], temperature[ids, None]) & (result[ids] == 0)).any():
            raise ArithmeticError("positive allocation batch support underflowed in requested dtype")
    return BatchAllocation(result, lam, branch, response, (result > 0) & (result < caps), residual)
