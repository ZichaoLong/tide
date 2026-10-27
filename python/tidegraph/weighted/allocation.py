"""Allocation §4.1, §6.1–2, S.2 at upstream 0770593.

The forward solves the strictly concave quadratic problem. Its backward is
the exact solution map on the selected branch, never the solver iteration map
or the legacy HST. Boundary conventions are in docs/weighted-allocation.md.
"""
from dataclasses import dataclass
import torch
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class Allocation:
    controls: torch.Tensor
    multiplier: torch.Tensor
    constraint: str
    free: torch.Tensor
    residual: torch.Tensor
    backward: str = "quad-exact-map-v1"


def _scalar(value, reference, name):
    if not isinstance(value, torch.Tensor):
        value = reference.new_tensor(value)
    if value.shape != () or (value.dtype, value.device) != (reference.dtype, reference.device):
        raise ValueError(f"allocation {name} must be a scalar in the score dtype/device")
    if not torch.isfinite(value):
        raise ValueError(f"nonfinite allocation {name}")
    return value


def _validate(t, mu, lower, upper, temperature):
    if (not isinstance(t, torch.Tensor) or t.ndim != 1 or not t.numel()
            or t.dtype not in {torch.float32, torch.float64}):
        raise ValueError("allocation requires nonempty FP32/FP64 score vector")
    if not isinstance(mu, torch.Tensor) or (mu.shape, mu.dtype, mu.device) != (t.shape, t.dtype, t.device):
        raise ValueError("allocation upper bounds do not match scores")
    if not torch.isfinite(t).all() or not torch.isfinite(mu).all() or ((mu < 0) | (mu > 1)).any():
        raise ValueError("nonfinite scores or invalid allocation upper bounds")
    lower, upper, temperature = [_scalar(x, t, n) for x, n in
                                 ((lower, "lower"), (upper, "upper"), (temperature, "temperature"))]
    if lower < 0 or upper < lower or lower > mu.sum() or temperature <= 0:
        raise ValueError("infeasible allocation bounds or nonpositive temperature")
    return lower, upper, temperature


def _response(t, mu, temperature, multiplier):
    return torch.minimum(torch.clamp_min((t-multiplier)/temperature, 0), mu)


def _solve(t, mu, target, temperature):
    """Sorted breakpoints, O(n log n) work/storage; no numerical trajectory input.

    Center scores first. The integer free count is exact, including tied
    breakpoints. An affine segment determines the multiplier in one division.
    """
    shift = t.max()
    centered = t-shift
    points = torch.cat((centered-temperature*mu, centered))
    points, order = points.sort(stable=True)
    # Each lower breakpoint changes a capped coordinate to free; each upper
    # changes a free coordinate to zero. Zero-cap pairs cancel at their tie.
    enter = order < t.numel()
    index = order.remainder(t.numel())
    count = torch.cumsum(torch.where(enter, 1, -1), 0)
    delta = torch.where(enter, centered[index]/temperature-mu[index], -centered[index]/temperature)
    intercept = mu.sum()+torch.cumsum(delta, 0)
    totals = intercept-count*points/temperature
    right = torch.cat((points[1:], points[-1:]))
    candidates = (count > 0) & (points < right) & (totals >= target)
    # On a flat segment the endpoint solution is still unique. Choose the
    # last nonempty descending segment whose left total includes the target.
    chosen = torch.where(candidates, torch.arange(len(points), device=t.device), -1).max()
    if chosen < 0:
        raise ArithmeticError("quadratic allocation lost its feasible breakpoint interval")
    lam = (intercept[chosen]-target)*temperature/count[chosen]
    c = _response(centered, mu, temperature, lam)
    # Reconstruct from actual responses to reduce prefix-sum cancellation.
    free = (c > 0) & (c < mu)
    if free.any():
        capped = (c == mu) & ~free
        lam = (centered[free].sum()+temperature*(mu[capped].sum()-target))/free.sum()
        c = _response(centered, mu, temperature, lam)
    return c, lam+shift


class _ExactQuad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, mu, target, temperature, c, multiplier, equality):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(t, mu, temperature, c, multiplier)
        ctx.equality = equality
        return c.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return (None,)*7
        t, mu, temperature, c, multiplier = ctx.saved_tensors
        free = (c > 0) & (c < mu)
        a = free.to(c.dtype)/temperature
        # At an exact upper breakpoint choose the saturated-side derivative;
        # zero cap and lower breakpoint choose the zero-side derivative.
        e = ((c == mu) & (mu > 0)).to(c.dtype)
        total = a.sum()
        mean = (a*grad).sum()/total if ctx.equality and total > 0 else grad.new_zeros(())
        centered = grad-mean
        gt = a*centered
        gm = e*centered
        gtemp = (-a*(t-multiplier)/temperature*centered).sum()
        return gt, gm, mean, gtemp, None, None, None


def allocate_quad(scores, caps, lower, upper, temperature=1.0):
    """Solve max s·c - τ/2 ||c||², 0≤c≤caps, lower≤Σc≤upper.

    Returns actual controls, multiplier, selected branch and KKT/feasibility
    residual. All tensor inputs have a first-order VJP. This entry point accepts
    one complete region; joint region packing is a separate execution policy.
    """
    lower, upper, temperature = _validate(scores, caps, lower, upper, temperature)
    with torch.no_grad():
        mass = caps.sum()
        zero = scores.new_zeros(())
        if mass == 0 or upper == 0:
            c, lam, branch = torch.zeros_like(scores), zero, "zero"
        elif lower == mass:
            c, lam, branch = caps.clone(), zero, "caps"
        else:
            c = _response(scores, caps, temperature, zero)
            branch = "lower" if c.sum() < lower else "upper" if c.sum() > upper else "slack"
            if lower == upper:
                branch = "equality"
            target = lower if branch in {"lower", "equality"} else upper
            lam = zero
            if branch != "slack":
                c, lam = _solve(scores, caps, target, temperature)
        free = (c > 0) & (c < caps)
        if branch in {"zero", "caps"}:
            residual = zero
        else:
            response_error = (c-_response(scores, caps, temperature, lam)).abs().max()
            residual = torch.stack((response_error, (lower-c.sum()).clamp_min(0),
                                    (c.sum()-upper).clamp_min(0))).max()
            tolerance = 64*torch.finfo(scores.dtype).eps*max(1, scores.numel())
            if not torch.isfinite(residual) or residual > tolerance:
                raise ArithmeticError("quadratic allocation residual exceeds dtype tolerance")
    if branch == "zero":
        result = c  # Explicit all-zero boundary: no gradient roots.
    elif branch == "caps":
        result = caps  # Singleton feasible set: identity in caps only.
    else:
        target = lower if branch in {"lower", "equality"} else upper
        result = _ExactQuad.apply(scores, caps, target, temperature, c, lam, branch != "slack")
    return Allocation(result, lam, branch, free, residual)
