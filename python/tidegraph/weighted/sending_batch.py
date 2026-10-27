"""Allocation §4.2–4.4/6.3: complete batched sending rows and declared VJPs.

No padding candidates, no hard-forward surrogate, and no threshold for positive
support. Stable rank ties retain int64 node identity.
"""
import torch


def validate(scores, caps, temperature=None):
    if (scores.ndim != 2 or min(scores.shape) < 1 or scores.dtype not in {torch.float32, torch.float64}
            or (caps.shape, caps.dtype, caps.device) != (scores.shape, scores.dtype, scores.device)):
        raise ValueError("sending batch requires matching complete FP32/FP64 rows")
    if not torch.isfinite(scores).all() or not torch.isfinite(caps).all() or (caps < 0).any():
        raise ValueError("invalid batched sending scores or weights")
    if temperature is not None and (temperature.shape != (len(scores),)
            or (temperature.dtype, temperature.device) != (scores.dtype, scores.device)
            or not torch.isfinite(temperature).all() or (temperature <= 0).any()):
        raise ValueError("sending batch temperature requires positive row scalars")


class _Shares(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, base, temperature):
        ctx.set_materialize_grads(False)
        positive = base > 0
        present = positive.any(1)
        maximum = scores.masked_fill(~positive, -torch.inf).max(1).values
        maximum = torch.where(present, maximum, torch.zeros_like(maximum))
        shifted = (scores-maximum[:, None])/temperature[:, None]
        log_z = (shifted+base.log()).masked_fill(~positive, -torch.inf).logsumexp(1)
        log_z = torch.where(present, log_z, torch.zeros_like(log_z))
        log_ratio = shifted-log_z[:, None]
        probabilities = torch.where(positive, (log_ratio+base.log()).exp(), torch.zeros_like(scores))
        if not torch.isfinite(probabilities).all() or (positive & (probabilities == 0)).any():
            raise ArithmeticError("positive batched attention support underflowed in requested dtype")
        shifted = torch.where(present[:, None], shifted, torch.zeros_like(shifted))
        log_ratio = torch.where(present[:, None], log_ratio, torch.zeros_like(log_ratio))
        ctx.save_for_backward(scores, base, temperature, probabilities, shifted, log_ratio)
        return probabilities.clone()

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise ValueError("batched sending supports first-order VJP only")
        if grad is None:
            return None, None, None
        scores, base, temperature, p, shifted, log_ratio = ctx.saved_tensors
        present = (base > 0).any(1)
        delta = grad-(grad*p).sum(1)[:, None]
        ds = p*delta/temperature[:, None]
        da = delta.sign()*(log_ratio+delta.abs().log()).exp()
        da = torch.where((delta != 0) & present[:, None], da, torch.zeros_like(da))
        dtau = -(ds*torch.where(p > 0, shifted, torch.zeros_like(shifted))).sum(1)
        if not all(torch.isfinite(g).all() for g in (ds, da, dtau)):
            raise ArithmeticError("batched attention VJP exceeds requested dtype range")
        return ds, da, dtau


def shares(scores, base, temperature):
    validate(scores, base, temperature)
    return _Shares.apply(scores, base, temperature)


def attention(scores, caps, received, mode, base, temperature):
    validate(scores, caps, temperature); validate(scores, received)
    choices = {"direct_attention": {"mu", "received"}, "unit_attention": {"mu", "uniform"}, "pooled_attention": {"mu"}}
    if (caps > 1).any() or (received > caps).any() or base not in choices.get(mode, set()):
        raise ValueError("invalid batched attention sending binding")
    weights = caps if base == "mu" else received if base == "received" else torch.ones_like(caps)
    p = shares(scores, weights, temperature)
    controls = caps*p if mode == "direct_attention" else caps.sum(1).clamp_max(1)[:, None]*p if mode == "pooled_attention" else p
    if ((weights > 0) & (controls == 0)).any():
        raise ArithmeticError("positive batched attention sending support underflowed in requested dtype")
    return controls


def proportional(caps, k):
    from .config import positive_int
    positive_int("k", k); validate(caps, caps)
    if (caps > 1).any():
        raise ValueError("invalid proportional caps")
    mass = caps.sum(1)
    scale = mass.clamp_max(k)/torch.where(mass > 0, mass, torch.ones_like(mass))
    # The all-zero scalar function is the identity in caps, including its VJP.
    scale = torch.where(mass > 0, scale, torch.ones_like(scale))
    result = caps*scale[:, None]
    if ((caps > 0) & (result == 0)).any():
        raise ArithmeticError("positive proportional batch support underflowed")
    return result


class _Sorted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, caps, target, nodes, k):
        ctx.set_materialize_grads(False)
        ids = nodes.argsort(dim=1, stable=True)
        sorted_scores = scores.gather(1, ids).masked_fill(caps.gather(1, ids) <= 0, -torch.inf)
        order = ids.gather(1, sorted_scores.argsort(dim=1, descending=True, stable=True))
        cs = caps.gather(1, order); ss = scores.gather(1, order)
        result, saturated, before = torch.zeros_like(caps), torch.zeros_like(caps), torch.zeros_like(caps)
        partial = torch.full((len(caps),), -1, dtype=torch.int64, device=caps.device)
        if k is not None:
            positive = (cs > 0) & (ss > 0)
            chosen = positive & (positive.to(torch.int64).cumsum(1) <= k)
            first = (cs > 0) & ((cs > 0).to(torch.int64).cumsum(1) == 1)
            chosen = torch.where(positive.any(1)[:, None], chosen, first)
            result, saturated = torch.where(chosen, cs, 0), chosen.to(caps.dtype)
        else:
            remaining = target.clone()
            for j in range(caps.shape[1]):
                live = (cs[:, j] > 0) & (remaining > 0)
                full = live & (remaining >= cs[:, j]); part = live & ~full
                before = torch.where(part[:, None], saturated, before)
                partial = torch.where(part, torch.full_like(partial, j), partial)
                result[:, j] = torch.where(live, torch.minimum(remaining, cs[:, j]), 0)
                saturated[:, j] = full.to(caps.dtype)
                remaining = torch.where(live, remaining-cs[:, j], remaining)
            complete = target == caps.sum(1)
            result = torch.where(complete[:, None], cs, result)
            saturated = torch.where(complete[:, None], (cs > 0).to(caps.dtype), saturated)
            before = torch.where(complete[:, None], torch.zeros_like(before), before)
            partial = torch.where(complete, torch.full_like(partial, -1), partial)
        inverse = order.argsort(1)
        ctx.hard = k is not None
        ctx.save_for_backward(scores, caps, target, saturated, before, partial, order, inverse)
        return result.gather(1, inverse)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise ValueError("batched sending supports first-order VJP only")
        if grad is None:
            return (None,)*5
        _, _, target, saturated, before, partial, order, inverse = ctx.saved_tensors
        sorted_grad = grad.gather(1, order)
        gtarget = torch.where(partial >= 0, sorted_grad.gather(1, partial.clamp_min(0)[:, None]).squeeze(1), torch.zeros_like(target))
        gcaps = (sorted_grad*saturated-before*gtarget[:, None]).gather(1, inverse)
        return None, gcaps, None if ctx.hard else gtarget, None, None


def sorted_send(scores, caps, nodes, *, target=None, k=None):
    validate(scores, caps)
    if ((caps > 1).any() or nodes.shape != scores.shape or nodes.dtype != torch.int64
            or nodes.device != scores.device or (nodes < 0).any()
            or (nodes.sort(1).values[:, 1:] == nodes.sort(1).values[:, :-1]).any()):
        raise ValueError("invalid batched sorted sending node IDs/caps")
    if k is not None:
        from .config import positive_int
        positive_int("k", k)
        if target is not None:
            raise ValueError("hard sending takes no total")
        target = scores.new_zeros(len(scores))
    elif target is None or (target.shape, target.dtype, target.device) != ((len(scores),), scores.dtype, scores.device):
        raise ValueError("zero-temperature sending requires row totals")
    elif not torch.isfinite(target).all() or (target < 0).any() or (target > caps.sum(1)).any():
        raise ValueError("infeasible batched sending total")
    return _Sorted.apply(scores, caps, target, nodes, k)
