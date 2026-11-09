"""Allocation §4.1/§6/S.2: bern and fixed-alpha ent exact-map first-order VJP.

Newton steps are bracketed forward numerics, not the differentiation target.
CPU dtype limits that erase a mathematically positive message fail explicitly.
"""
import torch
from torch.autograd.function import once_differentiable
from .allocation import Allocation, _validate, _scalar, allocate_quad
from .regularizers import Regularizer
from .allocation_failure import scalar_failure
from . import support_policy as support


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
    def forward(ctx, t, caps, target, temperature, controls, response_input, equality, rule, shift, policy):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(t, caps, target, temperature, controls, response_input, shift)
        ctx.equality, ctx.rule = equality, rule
        ctx.policy = policy
        return controls.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return (None,)*10
        _, caps, _, temperature, controls, x, shift = ctx.saved_tensors
        if ctx.policy != "strict":
            scores, cap_bar, mean, tau_bar = support.shannon_vjp(
                x, caps, temperature, grad, grad.new_tensor(ctx.equality, dtype=torch.bool))
            gshift = None if shift is None else grad.new_zeros(()) if ctx.equality else -scores.sum()
            return scores, cap_bar, mean, tau_bar, None, None, None, None, gshift, None
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
        return scores, e*centered, mean, (dtau*centered).sum(), None, None, None, None, gshift, None


def allocate(scores, caps, lower, upper, temperature=1., *, regularizer="quad", ent_alpha=None,
             score_shift=None, support_policy="strict"):
    """Exact Alloc from §4.1; same input domain and boundary policy as quad.

    Bern/Shannon positive support must remain representable. Underflow to zero
    is an ArithmeticError, never an undeclared removal of active messages.
    """
    rule = Regularizer(regularizer, ent_alpha)
    support.validate(support_policy, regularizer, ent_alpha)
    if rule.kind == "quad":
        return allocate_quad(scores, caps, lower, upper, temperature, score_shift=score_shift)
    lower, upper, temperature = _validate(scores, caps, lower, upper, temperature)
    score_shift = None if score_shift is None else _scalar(score_shift, scores, "score_shift")
    original_scores = scores
    def failure(message, **actual):
        return scalar_failure(message, original_scores, caps, lower, upper, temperature,
            regularizer=regularizer, ent_alpha=ent_alpha, score_shift=score_shift, **actual)
    with torch.no_grad():
        if score_shift is not None:
            scores = scores-score_shift
            if not torch.isfinite(scores).all():
                raise failure("shifted allocation scores exceed requested dtype range", stage="shift")
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
                try:
                    c, lam, x = _solve(original_scores, caps, target, temperature, rule)
                except ArithmeticError as error:
                    raise failure(str(error), stage="solve", branch=branch) from error
                if score_shift is not None:
                    lam = lam-score_shift
        residual, repairs = zero, None
        if branch not in {"zero", "caps"}:
            residual = torch.stack(((c-rule.response(x, caps, temperature)).abs().max(),
                                    (lower-c.sum()).clamp_min(0), (c.sum()-upper).clamp_min(0))).max()
            tolerance = 64*torch.finfo(scores.dtype).eps*max(1, scores.numel())
            if not torch.isfinite(residual) or residual > tolerance:
                raise failure("entropy allocation residual exceeds dtype tolerance", branch=branch,
                              controls=c, multiplier=lam, response=x, residual=residual, tolerance=tolerance)
            if support_policy != "strict":
                def reject(mask, message):
                    if mask.any():
                        error = failure(message, branch=branch, stage="support-storage",
                            controls=c, multiplier=lam, response=x, residual=residual, tolerance=tolerance)
                        error.details["support_policy"] = support_policy
                        raise error
                c, repairs = support.repair(c, x, caps, temperature, lower, upper, reject)
            elif (rule.positive_support(x, caps, temperature) & (c == 0)).any():
                raise failure("positive allocation support underflowed in the requested dtype", branch=branch,
                              controls=c, multiplier=lam, response=x, residual=residual, tolerance=tolerance)
        free = (c > 0) & (c < caps)
    if branch == "zero":
        result = c
    elif branch == "caps":
        result = caps
    else:
        target = lower if branch in {"lower", "equality"} else upper
        result = _ExactEntropy.apply(original_scores, caps, target, temperature, c, x, branch != "slack",
                                     rule, score_shift, support_policy)
    name = "bern" if rule.kind == "bern" else "ent"
    backward = name+"-exact-map-v1" if support_policy == "strict" else "ent-positive-floor-v1"
    return Allocation(result, lam, branch, free, residual, backward, repairs)
