"""Independent core-5 KV/region formulas for the static LH/HB cost anchor.

No production local kernels, allocation solver, scheduler or reference replay.
Leading axes are node, sample, position. Parameter stacking retains aliases and
their summed gradient; it is real work included in the measured forward.
"""
import math
import torch
from torch.nn import functional as F


def norm(x, epsilon):
    return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+epsilon)


def project(x, weights):
    """Different node matrices in one bmm; one shared matrix uses a single GEMM."""
    if all(w is weights[0] for w in weights):
        return F.linear(x, weights[0])
    weight = torch.stack(weights)
    return torch.bmm(x.flatten(1, -2), weight.transpose(1, 2)).reshape(
        *x.shape[:-1], weight.shape[1])


def parameter(nodes, name):
    return torch.stack([getattr(n, name) for n in nodes])[:, None, None]


def head(nodes, x):
    y = F.silu(project(x, [n.matrix for n in nodes])+parameter(nodes, "hidden_bias"))
    return torch.stack([n.bias for n in nodes])[:, None, None]+(y*parameter(nodes, "vector")).sum(-1)


def rotate(x, positions, frequencies):
    if not frequencies:
        return x
    count = len(frequencies)
    phase = positions[..., None]*x.new_tensor(frequencies)
    pairs = x[..., -2*count:].reshape(*x.shape[:-1], count, 2)
    a, b = pairs.unbind(-1)
    tail = torch.stack((a*phase.cos()-b*phase.sin(), a*phase.sin()+b*phase.cos()), -1)
    return torch.cat((x[..., :-2*count], tail.flatten(-2)), -1)


def weighted_attention(score, values, weights):
    """Explicit empty rows; all positive weights participate without clipping."""
    present = weights > 0
    logits = score+torch.where(present, weights, 1).log()
    logits = logits.masked_fill(~present, -torch.inf)
    nonempty = present.any(-1, keepdim=True)
    logits = torch.where(nonempty, logits, 0)
    probabilities = logits.softmax(-1)*present
    return (probabilities@values)*weights.sum(-1).clamp_max(1)[..., None]


def describe(nodes, mass, u, bank, times):
    """KV Mem and cat Describe, including textbook att keys and raw-time RoPE."""
    memories = [n.kernel for n in nodes]
    reads = [n.read_program for n in nodes]
    c, r = memories[0].config, reads[0].config
    normal = norm(u, c.epsilon)
    age = times[:, None]-bank["times"][None]
    visible = (age >= 0) & (age < c.window if c.window else True)
    weights = bank["weights"][:, :, None, :]*bank["valid"][:, :, None, :]*visible
    short = weights*torch.exp(-age.clamp_min(0).to(u.dtype)*c.beta_short)
    totals = [w.sum(-1) for w in (short, weights)]
    if c.memory_read == "sum":
        values = [(w@bank["summaries"])/total.clamp_min(1)[..., None]
                  for w, total in zip((short, weights), totals)]
    else:
        q = project(normal, [m.read_q for m in memories])
        key = project(bank["keys"], [m.read_k for m in memories])
        value = project(bank["values"], [m.read_v for m in memories])
        # Common integer rebasing precedes conversion; absolute int64 clocks
        # are never converted to floating coordinates.
        origin = bank["times"][0]
        q = rotate(q, (times-origin).to(u.dtype)/c.position_scale, c.read_frequencies)
        key = rotate(key, (bank["times"]-origin).to(u.dtype)/c.position_scale, c.read_frequencies)
        score = (q@key.transpose(-1, -2))/math.sqrt(c.summary_width)
        values = [weighted_attention(score, value, w) for w in (short, weights)]
    summary = project(normal, [m.summary for m in memories])
    stats = torch.stack((mass.log1p(), totals[0].log1p(), totals[1].log1p()), -1)
    features = torch.cat((normal, stats[..., :1], *values, stats[..., 1:],
                          summary*values[0], summary*values[1]), -1)
    recv = mass.clamp_max(1)
    fields = dict(w=recv, mu=(r.gamma_max*recv).clamp_max(1),
                  score=head([p.score for p in reads], features))
    if r.region_control_mode == "context":
        evidence = project(features, [p.evidence_weight for p in reads])+parameter(reads, "evidence_bias")
        fields["evidence"] = norm(evidence, c.epsilon)
    if r.region_score == "att":
        fields["x"] = project(normal, [p.current_weight for p in reads])+parameter(reads, "current_bias")
        source = torch.cat((*values, stats[..., 1:]), -1)
        fields["key"] = project(source, [p.key_weight for p in reads])+parameter(reads, "key_bias")
    return fields, visible


class _Quadratic(torch.autograd.Function):
    """Finite affine-interval enumeration plus the exact solution-map VJP.

The O(N²) enumeration is an independent small-region anchor, not an assertion
that this is the best large-region solver. Padding has zero capacity.
"""
    @staticmethod
    def forward(ctx, scores, caps, target, temperature):
        shape = scores.shape
        s, mu, total = scores.flatten(0, -2), caps.flatten(0, -2), target.flatten()
        full, empty = (total >= mu.sum(-1)) & (mu.sum(-1) > 0), total <= 0
        ids = (~full & ~empty).nonzero().flatten()
        result = torch.where(full[:, None], mu, 0)
        if len(ids):
            x, cap, t = s[ids], mu[ids], total[ids]
            x = x-x.amax(-1, keepdim=True)
            points = torch.cat((x-temperature*cap, x), -1).sort(-1).values
            left, right = points[:, :-1], points[:, 1:]
            mid = (left+right)/2
            free = (mid[..., None] > (x-temperature*cap)[:, None]) & (mid[..., None] < x[:, None])
            upper = mid[..., None] <= (x-temperature*cap)[:, None]
            count = free.sum(-1)
            lam = ((free*x[:, None]).sum(-1)+temperature*((upper*cap[:, None]).sum(-1)-t[:, None]))/count.clamp_min(1)
            possible = (count > 0) & (left < right)
            error = torch.maximum((left-lam).clamp_min(0), (lam-right).clamp_min(0)).masked_fill(~possible, torch.inf)
            chosen = error.argmin(-1)
            root = lam.gather(1, chosen[:, None])
            response = torch.minimum(((x-root)/temperature).clamp_min(0), cap)
            if ((cap > 0) & (x-root > 0) & (response == 0)).any():
                raise ArithmeticError("independent LH quadratic positive support underflow")
            result = result.index_copy(0, ids, response)
        tolerance = 64*torch.finfo(scores.dtype).eps*scores.shape[-1]
        if not torch.isfinite(result).all() or ((result.sum(-1)-total).abs() > tolerance).any():
            raise ArithmeticError("independent LH quadratic allocation residual failure")
        interior = (result > 0) & (result < mu)
        saturated = (result >= mu) & (mu > 0)
        ctx.temperature, ctx.shape = temperature, shape
        ctx.save_for_backward(interior, saturated, full, empty)
        return result.reshape(shape)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise ValueError("LH quadratic anchor declares first-order exact-map VJP only")
        interior, saturated, full, empty = ctx.saved_tensors
        g = grad.flatten(0, -2)
        slope = interior.to(g.dtype)/ctx.temperature
        den = slope.sum(-1)
        mean = (slope*g).sum(-1)/den.clamp_min(torch.finfo(g.dtype).tiny)
        centered = g-mean[:, None]
        ds = slope*centered
        dc = saturated*centered
        dc = torch.where(full[:, None], g, torch.where(empty[:, None], 0, dc))
        return ds.reshape(ctx.shape), dc.reshape(ctx.shape), mean.reshape(ctx.shape[:-1]), None


def select(program, fields):
    c = program.config
    recv, caps, scores = (fields[name].movedim(0, -1) for name in ("w", "mu", "score"))
    received, capacity = recv.sum(-1), caps.sum(-1)
    if c.region_score == "att":
        weights = fields["w" if c.query_weight == "received" else "mu"]
        x = (weights[..., None]*fields["x"]).sum(0)/weights.sum(0).clamp_min(1)[..., None]
        q = program.query
        query = F.silu(F.linear(x, q.in_weight, q.hidden_bias))@q.out_weight.t()+q.out_bias
        scores = c.alpha_loc*scores+c.alpha_sel*(fields["key"]*query).sum(-1).movedim(0, -1)/math.sqrt(c.selection_width)
    if c.region_control_mode == "context":
        z = (fields["w"][..., None]*fields["evidence"]).sum(0)/received.clamp_min(1)[..., None]
        context = torch.cat((z, received.log1p()[..., None], capacity.log1p()[..., None]), -1)
        rho = head([program.head], context[None])[0].sigmoid()
    else:
        rho = c.total_fraction
    low, high = received.clamp_max(1), capacity.clamp_max(c.k)
    return _Quadratic.apply(scores, caps, low+rho*(high-low), c.allocation_temperature).movedim(-1, 0)


def full(nodes, u, bank, times, visible, controls):
    """Dense N×B×L Full followed by support masking; actual work is reported."""
    c = nodes[0].kernel.config
    modules = [n.full_program.content for n in nodes]
    n, b, length, d = u.shape
    width = d//c.heads
    q = project(norm(u, c.epsilon), [m.query for m in modules]).reshape(n, b, length, c.heads, width).transpose(2, 3)
    k, v = (bank[name].reshape(n, b, -1, c.heads, width).transpose(2, 3) for name in ("keys", "values"))
    origin = bank["times"][0]
    qt = (times-origin).to(u.dtype)/c.position_scale
    kt = (bank["times"]-origin).to(u.dtype)/c.position_scale
    if any(c.full_frequencies):
        q = torch.stack([rotate(q[:, :, h], qt, f) for h, f in enumerate(c.full_frequencies)], 2)
        k = torch.stack([rotate(k[:, :, h], kt, f) for h, f in enumerate(c.full_frequencies)], 2)
    score = q@k.transpose(-1, -2)/math.sqrt(width)
    if any(c.distance_slopes):
        displacement = (bank["times"][None]-times[:, None]).to(u.dtype)/c.position_scale
        score = score+u.new_tensor(c.distance_slopes)[None, None, :, None, None]*displacement
    weights = (bank["weights"]*bank["valid"])[:, :, None, None, :]*visible
    read = weighted_attention(score, v, weights).transpose(2, 3).reshape_as(u)
    value = u+c.alpha_att*project(read, [m.out for m in modules])
    normal = norm(value, c.epsilon)
    value = value+c.alpha_ff*project(F.silu(project(normal, [m.gate for m in modules]))
        *project(normal, [m.up for m in modules]), [m.down for m in modules])
    return torch.where((controls > 0)[..., None], value, 0)
