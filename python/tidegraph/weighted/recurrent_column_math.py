"""Affine/Delta/GDN/SSD coefficients and exact logical-time convolution columns."""
import math
import torch
from torch.nn import functional as F
from .column_layout import SequenceLayout, gather_column
from .numeric_time import passive_decay
from .tensors import norm
from .normalization import epsilon_l2


def cache_bank(memory, writes, normalized, weight, left):
    c = memory.config
    if c.basic:
        return None, None
    raw = normalized@memory.projection.t()
    if c.memory_family == "ssd":
        raw = raw[:, c.heads*c.value_width:c.heads*c.value_width+c.conv_width]
    values = torch.cat((left.cache_values, weight[:, None]*raw))
    times, offsets, permutation = [], [0], []
    old_count = len(left.cache_layout.times)
    for (a, z), (oa, oz) in zip(writes.spans, left.cache_layout.spans):
        permutation.extend(range(oa, oz)); permutation.extend(range(old_count+a, old_count+z))
        times.extend((left.cache_layout.times[oa:oz], writes.times[a:z]))
        offsets.append(offsets[-1]+oz-oa+z-a)
    return SequenceLayout(torch.tensor(offsets), torch.cat(times)), values.index_select(0, torch.tensor(permutation, device=values.device))


def convolve(conv, queries, raw, bank_layout, bank):
    base = conv.bias+conv.weight[0]*raw
    if conv.size == 1 or not len(bank_layout.times):
        return base
    # Integer slot lookup is host metadata; a single differentiable gather
    # carries all B×L×tap rows and its transpose. Missing slots use a sentinel.
    ids = []
    for (a, z), (oa, oz) in zip(queries.spans, bank_layout.spans):
        taps = queries.times[a:z, None]-conv.stride*torch.arange(1, conv.size, dtype=torch.int64)
        stamps = bank_layout.times[oa:oz]
        found = torch.searchsorted(stamps, taps)
        safe = torch.cat((stamps, stamps.new_tensor([-1])))
        present = (found < len(stamps)) & (safe[found] == taps) & (taps >= 0)
        ids.append(torch.where(present, found+oa, len(bank_layout.times)))
    history = gather_column(bank, torch.cat(ids).to(bank.device))
    return base+(conv.weight[1:]*history).sum(1)


def features(memory, layout, u, cache_layout, cache_values):
    c = memory.config
    raw = norm(u, c.epsilon)@memory.projection.t()
    gate = delta = None
    if c.memory_family == "ssd":
        gate, raw, delta = raw.split((c.heads*c.value_width, c.conv_width, c.heads), -1)
        delta = F.softplus(delta+memory.delta_bias)
    values = F.silu(convolve(memory.conv, layout, raw, cache_layout, cache_values))
    if c.memory_family == "gated_delta":
        q, k, v = values.reshape(-1, c.heads, 2*c.key_width+c.value_width).split((c.key_width, c.key_width, c.value_width), -1)
        return epsilon_l2(q, c.qk_epsilon)/math.sqrt(c.key_width), epsilon_l2(k, c.qk_epsilon), v, gate, delta
    v, k, q = values.split((c.heads*c.value_width, c.groups*c.key_width, c.groups*c.key_width), -1)
    k, q = (x.reshape(-1, c.groups, c.key_width).repeat_interleave(c.heads//c.groups, 1) for x in (k, q))
    return q, k, v.reshape(-1, c.heads, c.value_width), gate, delta


def coefficients(memory, layout, u, x, weight, gap, cache_layout, cache_values):
    c = memory.config
    if c.basic:
        k = x@memory.key.t()
        k = k/k.norm(dim=-1, keepdim=True).clamp_min(1)
        value = x@memory.value.t()
        retain = torch.exp(-weight*F.softplus(x@memory.retain+memory.retain_bias))
        decay = passive_decay(gap[:, None], (c.lambda_short, 0.), x)*retain[:, None]
        gain = weight
        matrix = c.memory_family == "delta"
        if matrix:
            gain = gain*torch.sigmoid(x@memory.write+memory.write_bias)
            a = (torch.eye(c.key_width, device=x.device, dtype=x.dtype)-gain[:, None, None]*k[:, :, None]*k[:, None, :])[:, None]
            a = a*decay[..., None, None]
        else:
            a = decay[..., None, None]
        b = (gain[:, None, None]*value[:, :, None]*k[:, None, :])[:, None].expand(-1, 2, -1, -1)
        return a, b, matrix
    _, k, value, _, delta = features(memory, layout, u, cache_layout, cache_values)
    decay = passive_decay(gap[:, None], c.matrix_rates, x)
    if c.memory_family == "gated_delta":
        delta = F.softplus(x@memory.delta.t()+memory.delta_bias)
        retain = torch.exp(-weight[:, None]*memory.log_rate.exp()*delta)
        gain = weight[:, None]*torch.sigmoid(x@memory.write.t()+memory.write_bias)
        a = torch.eye(c.key_width, device=x.device, dtype=x.dtype)-gain[..., None, None]*k[..., :, None]*k[..., None, :]
        a = a*(decay*retain)[..., None, None]
        b = gain[..., None, None]*value[..., :, None]*k[..., None, :]
        return a, b, True
    step = weight[:, None]*delta
    return ((decay*torch.exp(-memory.log_rate.exp()*step))[..., None, None],
            step[..., None, None]*value[..., :, None]*k[..., None, :], False)


def snapshots(window, rows, mode, initial):
    layout = window.layout.select(rows)
    if mode == "0":
        ids = layout.owners().to(window.content.device)
        return layout, {n: getattr(initial, n).index_select(0, ids) for n in ("H", "Z", "W")}, initial.last_times.clamp_min(0)[ids.cpu()], initial.cache_layout, initial.cache_values
    selected = []
    b = layout.samples
    for owner, ((a, z), (wa, wz)) in enumerate(zip(layout.spans, window.write_layout.spans)):
        count = torch.searchsorted(window.write_layout.times[wa:wz], layout.times[a:z], right=mode == "+")
        selected.append(torch.where(count > 0, b+wa+count-1, owner))
    ids = torch.cat(selected)
    return layout, {n: x.index_select(0, ids.to(x.device)) for n, x in window.bank.items()}, window.times[ids], window.cache_layout, window.cache_values


def effective(memory, layout, state, times):
    c = memory.config
    delta = (layout.times-times).clamp_min(0).to(state["H"].device)
    rates = (c.lambda_short, 0.) if c.basic else c.matrix_rates
    H = passive_decay(delta[:, None], rates, state["H"])[..., None, None]*state["H"]
    decay = passive_decay(delta[:, None], (c.lambda_short, 0.), state["Z"])
    return H, decay[..., None]*state["Z"], decay*state["W"]
