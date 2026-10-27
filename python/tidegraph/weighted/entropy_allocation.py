"""Allocation §4.1/§6/S.2: bern and fixed-alpha ent exact-map first-order VJP.

Newton steps are bracketed forward numerics, not the differentiation target.
CPU dtype limits that erase a mathematically positive message fail explicitly.
"""
import torch
from torch.autograd.function import once_differentiable
from .allocation import Allocation, _validate, _scalar, allocate_quad
from .regularizers import Regularizer


def _solve(t, caps, target, temperature, rule):
    shift = t.max()
    centered = t-shift
    points = centered-rule.inverse_bracket(target/caps.sum(), caps, temperature)
    positive = caps > 0
    low, high = points[positive].min(), points[positive].max()
    lam = low+(high-low)/2
    for _ in range(80):
        c = rule.response(centered-lam, caps, temperature)
        total = c.sum()
        low, high = (lam, high) if total > target else (low, lam)
        a, _, _ = rule.derivatives(centered-lam, caps, temperature, c)
        slope = a.sum()
        newton = lam+(total-target)/slope if slope > 0 else high
        proposal = newton if low < newton < high else low+(high-low)/2
        if proposal == lam or low == high:
            break
        lam = proposal
    choices = torch.stack((low, lam, high))
    errors = torch.stack([(rule.response(centered-x, caps, temperature).sum()-target).abs() for x in choices])
    lam = choices[errors.argmin()]
    response_input = centered-lam
    return rule.response(response_input, caps, temperature), lam+shift, response_input


class _ExactEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, caps, target, temperature, controls, response_input, equality, rule, shift):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(t, caps, target, temperature, controls, response_input, shift)
        ctx.equality, ctx.rule = equality, rule
        return controls.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return (None,)*9
        _, caps, _, temperature, controls, x, shift = ctx.saved_tensors
        a, e, dtau = ctx.rule.derivatives(x, caps, temperature, controls)
        total = a.sum()
        if ctx.equality and total > 0:
            anchor = grad[a.argmax()]
            relative = grad-anchor
            offset = (a*relative).sum()/total
            mean, centered = anchor+offset, relative-offset
        else:
            mean, centered = grad.new_zeros(()), grad
        scores = a*centered
        gshift = None if shift is None else grad.new_zeros(()) if ctx.equality else -scores.sum()
        return scores, e*centered, mean, (dtau*centered).sum(), None, None, None, None, gshift


def allocate(scores, caps, lower, upper, temperature=1., *, regularizer="quad", ent_alpha=None, score_shift=None):
    """Exact Alloc from §4.1; same input domain and boundary policy as quad.

    Bern/Shannon positive support must remain representable. Underflow to zero
    is an ArithmeticError, never an undeclared removal of active messages.
    """
    rule = Regularizer(regularizer, ent_alpha)
    if rule.kind == "quad":
        return allocate_quad(scores, caps, lower, upper, temperature, score_shift=score_shift)
    lower, upper, temperature = _validate(scores, caps, lower, upper, temperature)
    score_shift = None if score_shift is None else _scalar(score_shift, scores, "score_shift")
    original_scores = scores
    with torch.no_grad():
        if score_shift is not None:
            scores = scores-score_shift
            if not torch.isfinite(scores).all():
                raise ArithmeticError("shifted allocation scores exceed requested dtype range")
        mass, zero = caps.sum(), scores.new_zeros(())
        if mass == 0 or upper == 0:
            c, lam, x, branch = torch.zeros_like(scores), zero, scores, "zero"
        elif lower == mass:
            c, lam, x, branch = caps.clone(), zero, scores, "caps"
        else:
            c = rule.response(scores, caps, temperature)
            branch = "lower" if c.sum() < lower else "upper" if c.sum() > upper else "slack"
            if lower == upper:
                branch = "equality"
            lam, x = zero, scores
            if branch != "slack":
                target = lower if branch in {"lower", "equality"} else upper
                c, lam, x = _solve(scores, caps, target, temperature, rule)
        residual = zero
        if branch not in {"zero", "caps"}:
            residual = torch.stack(((c-rule.response(x, caps, temperature)).abs().max(),
                                    (lower-c.sum()).clamp_min(0), (c.sum()-upper).clamp_min(0))).max()
            tolerance = 64*torch.finfo(scores.dtype).eps*max(1, scores.numel())
            if not torch.isfinite(residual) or residual > tolerance:
                raise ArithmeticError("entropy allocation residual exceeds dtype tolerance")
            if (rule.positive_support(x, caps, temperature) & (c == 0)).any():
                raise ArithmeticError("positive allocation support underflowed in the requested dtype")
        free = (c > 0) & (c < caps)
    if branch == "zero":
        result = c
    elif branch == "caps":
        result = caps
    else:
        target = lower if branch in {"lower", "equality"} else upper
        result = _ExactEntropy.apply(original_scores, caps, target, temperature, c, x, branch != "slack", rule, score_shift)
    name = "bern" if rule.kind == "bern" else "ent"
    return Allocation(result, lam, branch, free, residual, name+"-exact-map-v1")
