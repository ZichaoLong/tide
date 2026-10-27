"""0770593 allocation §4.2–4.4 / §6.3: exact sending maps, independent of routing.

See docs/weighted-sending.md for first-order and nondifferentiable boundaries.
Scores/caps/target are separate roots. No hard map uses legacy HST.
"""
import torch
from torch.autograd.function import once_differentiable
from .allocation import _scalar
from .config import positive_int


def _vectors(scores, values, *, caps=False):
    if (not isinstance(scores, torch.Tensor) or scores.ndim != 1
            or scores.dtype not in {torch.float32, torch.float64}
            or not isinstance(values, torch.Tensor)
            or (values.shape, values.dtype, values.device) != (scores.shape, scores.dtype, scores.device)):
        raise ValueError("sending requires matching FP32/FP64 vectors")
    if (not torch.isfinite(scores).all() or not torch.isfinite(values).all()
            or (values < 0).any() or caps and (values > 1).any()):
        raise ValueError("invalid sending scores/weights/caps")


class _Shares(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, base, temperature):
        ctx.set_materialize_grads(False)
        positive = base > 0
        if positive.any():
            shifted = (scores-scores[positive].max())/temperature
            log_z = torch.logsumexp(shifted[positive]+base[positive].log(), 0)
            log_ratio = shifted-log_z
            p = torch.zeros_like(scores)
            p[positive] = (log_ratio[positive]+base[positive].log()).exp()
            if not torch.isfinite(p).all() or (p[positive] == 0).any():
                raise ArithmeticError("positive attention support underflowed in requested dtype")
        else:
            p, shifted, log_ratio = (torch.zeros_like(scores) for _ in range(3))
        ctx.save_for_backward(scores, base, temperature, p, shifted, log_ratio)
        return p.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return None, None, None
        scores, base, temperature, p, shifted, log_ratio = ctx.saved_tensors
        if not (base > 0).any():
            # Declared zero VJP at the all-zero branch; not a usual Jacobian.
            return torch.zeros_like(scores), torch.zeros_like(base), temperature.new_zeros(())
        delta = grad-(grad*p).sum()
        ds = p*delta/temperature
        # log form also covers right derivatives at zero base, without an
        # overflowing intermediate exp(log_ratio) times a small cotangent.
        da = delta.sign()*(log_ratio+delta.abs().log()).exp()
        da = torch.where(delta != 0, da, torch.zeros_like(da))
        dtau = -(ds*torch.where(p > 0, shifted, torch.zeros_like(shifted))).sum()
        if not all(torch.isfinite(g).all() for g in (ds, da, dtau)):
            raise ArithmeticError("attention VJP exceeds requested dtype range")
        return ds, da, dtau


def attention_shares(scores, base, temperature=1.):
    """Equation (10), including equation (20)'s zero-base right derivative.

    All entries are real candidates. Padding must be removed by its presence
    mask before this call; a zero base is not a missing candidate.
    """
    _vectors(scores, base)
    temperature = _scalar(temperature, scores, "temperature")
    if temperature <= 0:
        raise ValueError("attention temperature must be positive")
    return _Shares.apply(scores, base, temperature)


def attention_send(scores, caps, received, *, mode, base=None, temperature=1.):
    """Equations (11–13). Return controls, which differ from internal shares."""
    _vectors(scores, caps, caps=True)
    _vectors(scores, received, caps=True)
    if (received > caps).any():
        raise ValueError("received exceeds sending caps")
    choices = {"direct_attention": {"mu", "received"}, "unit_attention": {"mu", "uniform"},
               "pooled_attention": {"mu"}}
    if mode not in choices:
        raise ValueError("unknown attention sending mode")
    base = base or ("uniform" if mode == "unit_attention" else "mu")
    if base not in choices[mode]:
        raise ValueError("inapplicable attention allocation_base")
    weights = caps if base == "mu" else received if base == "received" else torch.ones_like(caps)
    p = attention_shares(scores, weights, temperature)
    c = caps*p if mode == "direct_attention" else caps.sum().clamp_max(1)*p if mode == "pooled_attention" else p
    support = weights > 0
    if (support & (c == 0)).any():
        raise ArithmeticError("positive attention sending support underflowed in requested dtype")
    return c


def proportional(caps, k):
    """Equation (14). Deliberately takes no scores, temperature or regularizer."""
    positive_int("k", k)
    _vectors(caps, caps, caps=True)
    mass = caps.sum()
    if mass == 0:
        # This function is locally c=mu on its nonnegative domain near zero.
        return caps
    c = caps*(mass.clamp_max(k)/mass)
    if ((caps > 0) & (c == 0)).any():
        raise ArithmeticError("positive proportional support underflowed in requested dtype")
    return c


def _order(scores, caps, nodes):
    if (len(nodes) != scores.numel() or any(type(v) is not int or not 0 <= v < 2**63 for v in nodes)
            or len(set(nodes)) != len(nodes)):
        raise ValueError("sending node IDs must be distinct nonnegative int64")
    return sorted((i for i in range(len(nodes)) if caps[i] > 0),
                  key=lambda i: (-float(scores[i].detach()), nodes[i]))


class _SortedSending(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, caps, target, order, k):
        ctx.set_materialize_grads(False)
        c, saturated, before = torch.zeros_like(caps), torch.zeros_like(caps), torch.zeros_like(caps)
        partial = -1
        if k is not None:
            positive = [i for i in order if scores[i] > 0]
            chosen = positive[:k] if positive else order[:1]
            for i in chosen:
                c[i], saturated[i] = caps[i], 1
        elif target == caps.sum():
            c, saturated = caps.clone(), (caps > 0).to(caps.dtype)
        else:
            remaining = target.clone()
            for i in order:
                if remaining <= 0:
                    break
                if remaining >= caps[i]:
                    c[i], saturated[i] = caps[i], 1
                else:
                    c[i], partial, before = remaining, i, saturated.clone()
                remaining = remaining-caps[i]
        ctx.save_for_backward(scores, caps, target, saturated, before)
        ctx.partial, ctx.hard = partial, k is not None
        return c

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return (None,)*5
        _, caps, target, saturated, before = ctx.saved_tensors
        gtotal = grad[ctx.partial] if ctx.partial >= 0 else target.new_zeros(())
        return None, grad*saturated-before*gtotal, None if ctx.hard else gtotal, None, None


def total_zero(scores, caps, total, nodes):
    """§4.4.2: stable score/ID order, fractional fill, fixed-order branch VJP."""
    _vectors(scores, caps, caps=True)
    total = _scalar(total, scores, "total")
    if total < 0 or total > caps.sum():
        raise ValueError("infeasible zero-temperature sending total")
    if scores.device.type != "cpu":
        return _device_sorted(scores, caps, nodes, total, None)
    return _SortedSending.apply(scores, caps, total, _order(scores, caps, nodes), None)


def hard_selection(adjusted_scores, caps, k, nodes):
    """§4.4.1/3: positive top-k or one best nonpositive, cap amplitude VJP."""
    positive_int("k", k)
    _vectors(adjusted_scores, caps, caps=True)
    if adjusted_scores.device.type != "cpu":
        return _device_sorted(adjusted_scores, caps, nodes, None, k)
    return _SortedSending.apply(adjusted_scores, caps, caps.new_zeros(()),
                                _order(adjusted_scores, caps, nodes), k)


def _device_sorted(scores, caps, nodes, total, k):
    """Scalar target dispatch uses device ordering, with the same declared VJP.

    Static IDs can be prepared on the host; actual score ordering and fractional
    fill are tensor operations on the requested selection device.
    """
    from .sending_batch import sorted_send
    if (len(nodes) != scores.numel() or any(type(v) is not int or not 0 <= v < 2**63 for v in nodes)
            or len(set(nodes)) != len(nodes)):
        raise ValueError("sending node IDs must be distinct nonnegative int64")
    ids = torch.tensor([nodes], device=scores.device, dtype=torch.int64)
    return sorted_send(scores.unsqueeze(0), caps.unsqueeze(0), ids,
                       target=None if total is None else total.unsqueeze(0), k=k)[0]
