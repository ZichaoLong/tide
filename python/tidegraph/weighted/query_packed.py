"""KV Mem/Full with batched snapshot gathers and exact positive support.

No dynamic per-row nonzero/index kernels. Numeric zero padding never becomes
an atom; structural all-empty branches keep disconnected parameters absent.
"""
import math
import torch
from .query_layout import Layout, chunks
from .packing import project
from .tensors import norm, rotate_relative
from .numeric_time import passive_decay, time_values
from .attention_mean import attention_mean


def read(logits, values, weights, totals):
    positive = weights > 0
    active = positive.any(-1)
    if not bool(active.any()):
        return values.new_zeros((values.shape[0], values.shape[-1]))
    safe = torch.where(positive, weights, torch.ones_like(weights))
    masked = (logits+safe.log()).masked_fill(~positive, -torch.inf)
    # Empty rows only coexist on no-grad calls; differentiable query groups
    # partition long/short positive support before constructing their VJP.
    masked = torch.where(active[:, None], masked, torch.zeros_like(masked))
    # Keep the scalar first-positive centering anchor. An expired/padded value
    # must not cause cancellation (or 0*inf) in the centered Attention VJP.
    first = positive.to(torch.int64).argmax(-1)[:, None, None]
    anchor = values.gather(1, first.expand(-1, 1, values.shape[-1]))
    safe_values = torch.where(positive[..., None], values, anchor)
    value = attention_mean(masked, safe_values)*totals.clamp(max=1)[:, None]
    return torch.where(active[:, None], value, torch.zeros_like(value))


def memory(memory, states, times, u):
    c, output = memory.config, [None]*len(states)
    queries = project(norm(u, c.epsilon), memory.read_q) if c.memory_read == "qry" else None
    for ids in chunks(states, c.width):
        layout = Layout([states[i] for i in ids], [times[i] for i in ids], c.window)
        long = layout.weights()
        short = long*passive_decay(layout.age, c.beta_short, u)
        totals = (short.sum(1), long.sum(1))
        if c.memory_read == "sum":
            summaries = layout.visible_slot("summaries")
            values = tuple((w[..., None]*summaries).sum(1)/t.clamp(min=1)[:, None]
                           for w, t in zip((short, long), totals))
        else:
            n, length = layout.shape
            k = layout.project("keys", memory.read_k)
            k = rotate_relative(k, -time_values(layout.age, u)/c.position_scale, c.read_frequencies)
            q = queries.index_select(0, torch.tensor(ids, dtype=torch.int64, device=u.device))
            logits = (k*q[:, None]).sum(-1)/math.sqrt(c.summary_width)
            v = layout.project("values", memory.read_v)
            values = tuple(read(logits, v, w, t) for w, t in zip((short, long), totals))
        # One transpose node per field; per-row indexing would create one
        # whole-batch SelectBackward zero/assembly for every returned row.
        rows = zip(*(t.unbind() for t in (*values, *totals)))
        for i, row in zip(ids, rows):
            output[i] = row
    return tuple(torch.stack([row[j] for row in output]) for j in range(4))


def full(program, memory, states, times, u):
    c, output = memory.config, [None]*len(states)
    d = c.width//c.heads
    queries = project(norm(u, c.epsilon), program.query).reshape(-1, c.heads, d)
    for ids in chunks(states, c.width):
        layout = Layout([states[i] for i in ids], [times[i] for i in ids], c.window)
        n, length = layout.shape
        weights = layout.weights()
        total = weights.sum(1)
        keys, values = (layout.visible_slot(name).reshape(n, length, c.heads, d) for name in ("keys", "values"))
        q = queries.index_select(0, torch.tensor(ids, dtype=torch.int64, device=u.device))
        displacement = -time_values(layout.age, u)/c.position_scale
        reads = []
        for h in range(c.heads):
            k = rotate_relative(keys[:, :, h], displacement, c.full_frequencies[h])
            logits = (k*q[:, h, None]).sum(-1)/math.sqrt(d)+c.distance_slopes[h]*displacement
            reads.append(read(logits, values[:, :, h], weights, total))
        mixed = torch.cat(reads, -1)
        for i, row in zip(ids, mixed.unbind()):
            output[i] = row
    return project(torch.stack(output), program.out)


def branches(memory, weights, requests, kind):
    from ..region_metadata import positive
    states = [r.state if kind == "read" else r.comparison for r in requests]
    states = [memory.initial(weights) if s is None else s for s in states]
    result, flags, positions = [None]*len(states), [], []
    with torch.no_grad():
        for ids in chunks(states, memory.config.width):
            layout = Layout([states[i] for i in ids], [requests[i].time for i in ids], memory.config.window)
            writes = layout.weights()
            long = (writes > 0).any(-1)
            rows = [long]
            if kind == "read":
                rows.append((writes*passive_decay(layout.age, memory.config.beta_short, writes) > 0).any(-1))
            flags.extend(torch.stack(rows, -1).flatten().unbind())
            positions.extend(ids)
        values = positive(flags)
    stride = 2 if kind == "read" else 1
    for j, i in enumerate(positions):
        result[i] = tuple(values[j*stride:(j+1)*stride])
    return result
