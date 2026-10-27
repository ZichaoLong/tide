"""KV/recurrent Mem: batches of actual snapshots at original int64 query times.

This layer computes local queries only. It neither advances state nor invents
missing cache entries; same-size buckets retain each row's complete visibility.
"""
from collections import defaultdict
import math
import torch
from .config import KVConfig
from .convolution import ConvCache
from .packing import project
from .tensors import norm, rotate_relative
from .attention_mean import attention_mean
from .normalization import epsilon_l2
from .numeric_time import passive_decay, time_values


def weighted_read(logits, values, weights, totals):
    groups, ids = defaultdict(list), []
    for i, row in enumerate(weights):
        ids.append(torch.nonzero(row > 0).flatten())
        groups[ids[-1].numel()].append(i)
    result = [None]*len(weights)
    for length, rows in groups.items():
        if length == 0:
            for i in rows:
                result[i] = values.new_zeros(values.shape[-1])
            continue
        selected = torch.stack([ids[i] for i in rows])
        owners = torch.tensor(rows, dtype=torch.int64, device=weights.device)
        l = logits[owners[:, None], selected]
        w = weights[owners[:, None], selected]
        v = values[owners[:, None], selected]
        reads = totals[owners].clamp(max=1)[:, None]*attention_mean(l+w.log(), v)
        for i, value in zip(rows, reads.unbind()):
            result[i] = value
    return torch.stack(result)


def kv(memory, states, times, u):
    c, groups, visible = memory.config, defaultdict(list), []
    for i, (state, time) in enumerate(zip(states, times)):
        mask, age = memory.visible(state, time)
        visible.append(({k: v[mask] for k, v in state.slots.items()}, age[mask]))
        groups[int(mask.sum())].append(i)
    result = [None]*len(states)
    for length, rows in groups.items():
        s = {k: torch.stack([visible[i][0][k] for i in rows]) for k in visible[rows[0]][0]}
        age = torch.stack([visible[i][1] for i in rows])
        weights = (s["weights"]*passive_decay(age, c.beta_short, u), s["weights"])
        totals = tuple(w.sum(1) for w in weights)
        if c.memory_read == "sum":
            values = tuple((w[..., None]*s["summaries"]).sum(1)/total.clamp(min=1)[:, None]
                           for w, total in zip(weights, totals))
        else:
            q = project(norm(u[rows], c.epsilon), memory.read_q)
            k = project(s["keys"].reshape(-1, c.width), memory.read_k).reshape(len(rows), length, c.summary_width)
            k = rotate_relative(k, -time_values(age, u)/c.position_scale, c.read_frequencies)
            logits = (k*q[:, None]).sum(-1)/math.sqrt(c.summary_width)
            v = project(s["values"].reshape(-1, c.width), memory.read_v).reshape(len(rows), length, c.summary_width)
            values = tuple(weighted_read(logits, v, w, total) for w, total in zip(weights, totals))
        for j, i in enumerate(rows):
            result[i] = tuple(t[j] for t in (*values, *totals))
    return tuple(torch.stack([r[k] for r in result]) for k in range(4))


def effective(memory, states, times):
    c = memory.config
    old = {k: torch.stack([s.slots[k] for s in states]) for k in ("H", "Z", "W", "time")}
    # No float conversion before subtraction, including times above 2**53.
    delta = (old["time"].new_tensor(times)-old["time"]).clamp_min(0)
    matrix_rates = (c.lambda_short, 0.) if c.basic else c.matrix_rates
    matrix_decay = passive_decay(delta[:, None], matrix_rates, old["H"])
    summary_decay = passive_decay(delta[:, None], (c.lambda_short, 0.), old["Z"])
    return (matrix_decay[..., None, None]*old["H"],
            summary_decay[..., None]*old["Z"], summary_decay*old["W"])


def convolve(conv, states, times, raw):
    groups, result = defaultdict(list), [None]*len(states)
    for i, (state, time) in enumerate(zip(states, times)):
        cache = ConvCache(state.slots["conv_times"], state.slots["conv_values"])
        conv._validate(cache, time, raw[i])
        groups[len(cache.times)].append(i)
    for length, rows in groups.items():
        base = conv.bias+conv.weight[0]*raw[rows]
        if length and conv.size > 1:
            stamps = torch.stack([states[i].slots["conv_times"] for i in rows])
            values = torch.stack([states[i].slots["conv_values"] for i in rows])
            taps = stamps.new_tensor([times[i] for i in rows])[:, None]-conv.stride*torch.arange(
                1, conv.size, dtype=torch.int64, device=raw.device)
            matches = (taps[..., None] == stamps[:, None]).to(raw.dtype)
            history = matches@values
            base = base+(conv.weight[1:]*history).sum(1)
        for i, value in zip(rows, base.unbind()):
            result[i] = value
    return torch.stack(result)


def features(memory, states, times, u):
    c = memory.config
    projection = project(norm(u, c.epsilon), memory.projection)
    if c.memory_family == "ssd":
        gate, raw, delta = projection.split((c.heads*c.value_width, c.conv_width, c.heads), -1)
    else:
        raw = projection
    projected = torch.nn.functional.silu(convolve(memory.conv, states, times, raw))
    if c.memory_family == "gated_delta":
        q, k, value = projected.reshape(-1, c.heads, 2*c.key_width+c.value_width).split(
            (c.key_width, c.key_width, c.value_width), -1)
        q = epsilon_l2(q, c.qk_epsilon)/math.sqrt(c.key_width)
        k = epsilon_l2(k, c.qk_epsilon)
        return raw, q, k, value
    value, key, query = projected.split((c.heads*c.value_width, c.groups*c.key_width, c.groups*c.key_width), -1)
    key = key.reshape(-1, c.groups, c.key_width).repeat_interleave(c.heads//c.groups, dim=1)
    query = query.reshape(-1, c.groups, c.key_width).repeat_interleave(c.heads//c.groups, dim=1)
    return raw, gate, value.reshape(-1, c.heads, c.value_width), key, query, torch.nn.functional.softplus(delta+memory.delta_bias)


def matrix_read(memory, states, times, u):
    H, _, _ = effective(memory, states, times)
    f = features(memory, states, times, u)
    query = f[1] if memory.config.memory_family == "gated_delta" else f[4]
    return (H@query.unsqueeze(-1)).squeeze(-1)


def memory_read(memory, states, times, u):
    if isinstance(memory.config, KVConfig):
        return kv(memory, states, times, u)
    H, Z, W = effective(memory, states, times)
    c = memory.config
    values = Z/W.clamp(min=1)[..., None]
    if c.memory_read == "state":
        if c.basic:
            query = project(norm(u, c.epsilon), memory.read_query)
            reads = (H@query[:, None, :, None]).squeeze(-1)
            values = (memory.read_out@reads.unsqueeze(-1)).squeeze(-1)
        else:
            long = project(matrix_read(memory, states, times, u).flatten(1), memory.read_out)
            return values[:, 0], long, W[:, 0], W[:, 1]
    return values[:, 0], values[:, 1], W[:, 0], W[:, 1]
