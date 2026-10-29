"""Allocation §4.1, §6.1–2, S.2 at upstream 0770593.

The forward solves the strictly concave quadratic problem. Its backward is
the exact solution map on the selected branch, never the solver iteration map
or the legacy HST. Boundary conventions are in docs/weighted-allocation.md.
"""
from dataclasses import dataclass
import torch
from torch.autograd.function import once_differentiable
from .quadratic_coordinates import coordinates
from .allocation_failure import scalar_failure


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
    centered, shift, work, cap, remaining = coordinates(t, mu, target, temperature)
    points = torch.cat((work-temperature*cap, work))
    points, order = points.sort(stable=True)
    # Each lower breakpoint changes a capped coordinate to free; each upper
    # changes a free coordinate to zero. Zero-cap pairs cancel at their tie.
    enter = order < t.numel()
    index = order.remainder(t.numel())
    count = torch.cumsum(torch.where(enter, 1, -1), 0)
    delta = torch.where(enter, work[index]/temperature-cap[index], -work[index]/temperature)
    intercept = cap.sum()+torch.cumsum(delta, 0)
    totals = intercept-count*points/temperature
    right = torch.cat((points[1:], points[-1:]))
    candidates = (count > 0) & (points < right) & (totals >= remaining)
    # On a flat segment the endpoint solution is still unique. Choose the
    # last nonempty descending segment whose left total includes the target.
    chosen = torch.where(candidates, torch.arange(len(points), device=t.device), -1).max()
    if chosen < 0:
        raise ArithmeticError("quadratic allocation lost its feasible breakpoint interval")
    lam = (intercept[chosen]-remaining)*temperature/count[chosen]
    c = _response(centered, mu, temperature, lam)
    # Reconstruct from actual responses to reduce prefix-sum cancellation.
    free = (c > 0) & (c < mu)
    if free.any():
        capped = (c == mu) & ~free
        lam = (torch.where(free, centered, 0).sum()+temperature*(torch.where(capped, mu, 0).sum()-target))/free.sum()
        c = _response(centered, mu, temperature, lam)
    return c, lam+shift, centered-lam


class _ExactQuad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, mu, target, temperature, c, multiplier, equality, shift, response):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(t, mu, target, temperature, c, multiplier, shift, response)
        ctx.equality = equality
        return c.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return (None,)*9
        _, mu, _, temperature, c, _, shift, response = ctx.saved_tensors
        free = (c > 0) & (c < mu)
        a = free.to(c.dtype)/temperature
        # At an exact upper breakpoint choose the saturated-side derivative;
        # zero cap and lower breakpoint choose the zero-side derivative.
        e = ((c == mu) & (mu > 0)).to(c.dtype)
        total = a.sum()
        if ctx.equality and total > 0:
            anchor = grad[a.argmax()]
            relative = grad-anchor
            offset = (a*relative).sum()/total
            mean, centered = anchor+offset, relative-offset
        else:
            mean, centered = grad.new_zeros(()), grad
        gt = a*centered
        gm = e*centered
        gtemp = (-a*torch.where(free, response/temperature, 0)*centered).sum()
        gshift = None if shift is None else grad.new_zeros(()) if ctx.equality else -gt.sum()
        return gt, gm, mean, gtemp, None, None, None, gshift, None


def allocate_quad(scores, caps, lower, upper, temperature=1.0, *, score_shift=None):
    """Solve max s·c - τ/2 ||c||², 0≤c≤caps, lower≤Σc≤upper.

    Returns actual controls, multiplier, selected branch and KKT/feasibility
    residual. All tensor inputs have a first-order VJP. This entry point accepts
    one complete region; joint region packing is a separate execution policy.
    """
    lower, upper, temperature = _validate(scores, caps, lower, upper, temperature)
    score_shift = None if score_shift is None else _scalar(score_shift, scores, "score_shift")
    original_scores = scores
    def failure(message, **actual):
        return scalar_failure(message, original_scores, caps, lower, upper, temperature,
                              score_shift=score_shift, **actual)
    with torch.no_grad():
        if score_shift is not None:
            scores = scores-score_shift
            if not torch.isfinite(scores).all():
                raise failure("shifted allocation scores exceed requested dtype range", stage="shift")
        mass = caps.sum()
        zero = scores.new_zeros(())
        response = scores
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
                try:
                    c, lam, response = _solve(original_scores, caps, target, temperature)
                except ArithmeticError as error:
                    raise failure(str(error), stage="solve", branch=branch) from error
                if score_shift is not None:
                    lam = lam-score_shift
        free = (c > 0) & (c < caps)
        if branch in {"zero", "caps"}:
            residual = zero
        else:
            if ((caps > 0) & (response > 0) & (c == 0)).any():
                raise failure("positive quadratic allocation support underflowed in requested dtype",
                              branch=branch, controls=c, multiplier=lam, response=response)
            response_error = (c-_response(response, caps, temperature, zero)).abs().max()
            residual = torch.stack((response_error, (lower-c.sum()).clamp_min(0),
                                    (c.sum()-upper).clamp_min(0))).max()
            tolerance = 64*torch.finfo(scores.dtype).eps*max(1, scores.numel())
            if not torch.isfinite(residual) or residual > tolerance:
                raise failure("quadratic allocation residual exceeds dtype tolerance", branch=branch,
                              controls=c, multiplier=lam, response=response, residual=residual, tolerance=tolerance)
    if branch == "zero":
        result = c  # Explicit all-zero boundary: no gradient roots.
    elif branch == "caps":
        result = caps  # Singleton feasible set: identity in caps only.
    else:
        target = lower if branch in {"lower", "equality"} else upper
        result = _ExactQuad.apply(original_scores, caps, target, temperature, c, lam,
                                  branch != "slack", score_shift, response)
    return Allocation(result, lam, branch, free, residual)
