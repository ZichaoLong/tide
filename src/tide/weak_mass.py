"""Optional Shannon mass coordinates for weak positive senders.

Internal scalar slots store m for ordinary/zero mass and log(m) for m < 2^-60.
The disjoint signs let routing, gathers and cache compaction copy one scalar.
An old, untagged nonnegative entry always denotes its original physical mass.
Only disjoint edge slots may be added as coordinates. Physical sums, role
bounds and normalized reads have explicit conversions below.
"""
import torch

from .scaled_arithmetic import compensated_sum

POLICY = "weak-log-v1"
THRESHOLD = 2.**-60


def weak(x):
    return (x != 0) & (x < THRESHOLD)


class _WideInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.device, ctx.dtype = x.device, x.dtype
        return x.cpu().double()

    @staticmethod
    def backward(ctx, grad):
        from .scaled_arithmetic import GradientRangeError
        result = grad.to(ctx.dtype)
        if not bool(result.isfinite().all()):
            raise GradientRangeError("weak-log input adjoint exceeds its source dtype")
        return result.to(ctx.device)


def wide(x):
    # Some Ascend systems do not implement binary64. The reference path has
    # explicit CPU placement, including both legs of the differentiable copy.
    return _WideInput.apply(x) if x.requires_grad else x.cpu().double()


def decode(x):
    positive_log = x < 0
    return torch.where(positive_log, torch.where(positive_log, x, 0).exp(), x)


def encode(mass, like):
    if bool((~torch.isfinite(mass) | (mass < 0)).any()):
        raise ArithmeticError("invalid physical mass before weak-log encoding")
    small = (mass > 0) & (mass < THRESHOLD)
    # The exceptional KKT solver and bounds are binary64. Never reuse its
    # smallest-subnormal support placeholder as a pre-rounding logarithm.
    if bool((small & (mass < torch.finfo(torch.float64).tiny)).any()):
        raise ArithmeticError("pre-rounding mass exceeds weak-log binary64 solver range")
    coded = torch.where(small, torch.where(small, mass, 1).log(), mass)
    return coded.to(like.dtype).to(like.device)


def physical(x):
    return decode(wide(x)).to(x) if bool((x < 0).any()) else x


@torch.no_grad()
def above_one(weights):
    # Exactly one plus a weak positive can round to one even in binary64.
    head, tail = compensated_sum(weights)
    return ((head > 1) | ((head == 1) & (tail > 0))).squeeze(-1)


def twice_capped(x):
    if not bool(weak(x).any()):
        return torch.where(x >= .5, 1., 2*x)
    m = decode(wide(x))
    return encode(torch.where(m >= .5, 1., 2*m), x)


def control(score, received, roles, budget, profile, partition):
    from .region import region_control
    from .shannon_support import FLOOR
    n = score.shape[0]
    s, r = score.reshape(n, -1), received.reshape(n, -1)
    risky = (weak(r) | (s.abs() > 18)).any(0)

    def solve(scores, amounts, role):
        return region_control(scores[:, None], amounts[:, None], role, budget, profile, partition,
                              FLOOR if profile == "shannon" else "strict")[:, 0]

    # Ordinary rows keep the existing device solve. Check its outputs as well:
    # a near-exhausted role bound alone can make an interior coordinate weak.
    ordinary = (~risky).nonzero().flatten()
    out = torch.zeros_like(s)
    if ordinary.numel():
        result = solve(s[:, ordinary], r[:, ordinary], roles)
        out = out.index_copy(1, ordinary, result)
        risky = risky.index_copy(0, ordinary, weak(result).any(0))
    selected = risky.nonzero().flatten()
    if selected.numel():
        result = solve(wide(s[:, selected]), decode(wide(r[:, selected])), roles.cpu())
        out = out.index_copy(1, selected, encode(result, s))
    return out.reshape_as(score)


def reception_mean(mass, values, mask):
    from .weighted_mean import stable_weighted_mean
    # The unrounded positive coordinate is shared between the physical sum
    # and the normalized read. Only weak rows need the wide feature reduction.
    weights = mass.reshape(-1, mass.shape[-1])
    payload = values.reshape(weights.shape + (values.shape[-1],))
    table = None if mask is None else mask.expand_as(mass).reshape_as(weights)
    flagged = weak(weights).any(-1)
    total = (weights if table is None else weights.masked_fill(~table, 0)).sum(-1)
    result = values.new_zeros((weights.shape[0], values.shape[-1]), dtype=torch.promote_types(mass.dtype, values.dtype))
    for exceptional, indices in ((False, (~flagged).nonzero().flatten()), (True, flagged.nonzero().flatten())):
        if not indices.numel():
            continue
        m, v = weights[indices], payload[indices]
        membership = None if table is None else table[indices]
        if exceptional:
            m = decode(wide(m))
            membership = None if membership is None else membership.cpu()
            # A log-total has physical derivative 1/sum(m). Project absent
            # roots here, before that derivative returns to the source dtype.
            present = m if membership is None else m.masked_fill(~membership, 0)
            total = total.index_copy(0, indices, encode(present.sum(-1), mass))
            mean = stable_weighted_mean(m, wide(v), membership)
        else:
            mean = stable_weighted_mean(m, v, membership)
        result = result.index_copy(0, indices, mean.to(result))
    return total.reshape(mass.shape[:-1]), result.reshape(values.shape[:-2] + (values.shape[-1],))


def attention(q, k, v, weights, mask, backend, constant_zero, mass_scaled):
    """Reference weak read with a direct positive log-prior cotangent."""
    from .attention import _auxiliary, _weighted_softmax
    if backend not in ("auto", "math", "tiled"):
        raise ValueError("weak-log attention requires auto, math or tiled")
    shape = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2]) + (q.shape[-2], k.shape[-2])
    code = _auxiliary(wide(weights), shape, True)
    present = torch.ones((), dtype=torch.bool) if mask is None else _auxiliary(mask.cpu(), shape, False)
    if constant_zero is not None:
        present = present & ~(_auxiliary(constant_zero.cpu(), shape, True) & (code == 0))
    q64, k64, v64 = wide(q), wide(k), wide(v)
    shift = torch.where(code < 0, code, 0)
    # A weak sender gets a unit coefficient and adds its log mass to the
    # logit. Its cotangent is p_i * g.(v_i-u), with no division by m_i.
    coefficient = torch.where(code < 0, 1., code)
    raw = decode(code) if mass_scaled else None
    rows = []
    for start in range(0, shape[-2], 32):
        span = slice(start, start+32)
        def queries(x):
            return x if x.shape[-2] == 1 else x[..., span, :]
        visible = queries(present.reshape((1,)*(len(shape)-present.ndim) + present.shape))
        logits = q64[..., span, :] @ k64.transpose(-2,-1) * q.shape[-1]**-.5
        p = _weighted_softmax(logits + queries(shift), queries(coefficient), visible)
        origin = v64[..., :1, :].detach()
        nonempty = (visible & (queries(code) != 0)).any(-1, keepdim=True)
        read = p @ (v64-origin) + torch.where(nonempty, origin, 0)
        if mass_scaled:
            used = torch.where(visible, queries(raw), 0.)
            total = used.sum(-1, keepdim=True)
            read = read * torch.where(above_one(used)[..., None], 1., total)
        rows.append(read)
    result = torch.cat(rows, -2)
    dtype = torch.promote_types(q.dtype, torch.float32) if mass_scaled else q.dtype
    return result.to(dtype).to(q.device)
