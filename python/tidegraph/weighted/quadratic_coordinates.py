"""Allocation §6/S.2: a local coordinate for the quadratic transition band.

Let p be the first descending score whose prefix cap mass reaches target T.
At lambda=s[p], S<=T; at lambda=s[p]-tau*max(mu), S>=T. Center at s[p],
so the root is in [-tau*max(mu),0], regardless of distant capped/zero scores.
This is a coordinate change, not clipping the returned response or support.
"""
import torch


def coordinates(scores, caps, target, temperature):
    order = scores.argsort(dim=-1, descending=True, stable=True)
    prefix = caps.gather(-1, order).cumsum(-1)
    index = (prefix < target[..., None]).sum(-1).clamp_max(scores.shape[-1]-1)
    pivot = scores.gather(-1, order.gather(-1, index[..., None])).squeeze(-1)
    centered = scores-pivot[..., None]
    span = temperature[..., None]*caps.amax(-1, keepdim=True)
    if not (span > 0).all():
        raise ArithmeticError("quadratic transition interval underflowed in requested dtype")
    fixed = centered >= temperature[..., None]*caps
    variable = ~fixed & (centered > -span) & (caps > 0)
    # Only these coordinates can change on the certified interval. Removing
    # constant terms from the breakpoint scan avoids large prefix cancellation.
    work_scores = torch.where(variable, centered, 0)
    work_caps = torch.where(variable, caps, 0)
    remaining = target-torch.where(fixed, caps, 0).sum(-1)
    return centered, pivot, work_scores, work_caps, remaining
