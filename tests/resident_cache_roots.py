"""Independent device loss cotangents for public cache boundaries."""
import torch
from resident_training_cases import roots as ordinary_roots, terms as ordinary_terms


def roots(session, window, mode):
    r = ordinary_roots(session, window, mode)
    if mode == "none":
        return r
    items = []
    for group in window.cache:
        item = {}
        rows = torch.arange(group.key.shape[1], device=group.key.device)
        valid = (rows[None, :] < group.lengths[:, None]) & group.present[:, None]
        for name in ("key", "value", "log_bias"):
            if getattr(group, name) is None:
                continue
            with torch.enable_grad():
                leaf = getattr(group, name).detach().requires_grad_(True)
                mask = valid if name == "log_bias" else valid[:, :, None, None]
                safe = torch.where(mask, leaf, torch.zeros_like(leaf))
                loss = safe.square().sum() * (0 if mode == "zero" else .015625)
                gradient, = torch.autograd.grad(loss, (leaf,))
            item[name] = gradient.detach()
        items.append(item)
    r.cache = session.cotangents(window, cache=items).cache
    return r


def terms(result, continuation, mode):
    ordinary = ordinary_terms(result, continuation, mode)
    if mode == "none":
        return ordinary
    factor = 0 if mode == "zero" else .015625
    return ordinary + [s.slots[name].square().sum() * factor
                       for s in continuation.states.values() for name in ("key", "value", "log_bias") if name in s.slots]
