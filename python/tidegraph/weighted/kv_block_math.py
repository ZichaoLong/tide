"""Position-free KV bank queries, B and query time contracted together.

Inputs are private column leaves supplied by kv_block_query. No graph scheduler,
snapshot decoding, or reference execution belongs in these numerical functions.
"""
import torch
from torch.nn import functional as F
from .block_attention import attention
from .tensors import norm
from .numeric_time import passive_decay
from .dense_describe import columns


def banks(old, values, summaries, contents, lengths, maximum):
    """Join each raw left state with actual prepared writes once per field."""
    result = {name: [] for name in ('keys', 'values', 'summaries', 'weights')}
    offset = 0
    for prior, count in zip(old, lengths):
        key, value = values[offset:offset+count].chunk(2, -1)
        fresh = dict(keys=key, values=value, summaries=summaries[offset:offset+count],
                     weights=contents[offset:offset+count, 0].clamp(max=1))
        for name in result:
            joined = torch.cat((prior[name], fresh[name]), 0)
            pad = [0, 0] * joined.ndim
            pad[-1] = maximum - joined.shape[0]
            result[name].append(F.pad(joined, pad))
        offset += count
    return {name: torch.stack(rows) for name, rows in result.items()}


def gather(rows, indices):
    padded = torch.cat((rows, rows.new_zeros((1, *rows.shape[1:]))))
    return padded.index_select(0, indices.flatten()).reshape(*indices.shape, *rows.shape[1:])


def evaluate(program, memory, payload, controls, raw, indices, visible, ages, kind):
    c = memory.config
    h = gather(payload, indices)
    mass, u = h[..., 0], h[..., 1:]
    weight = torch.where(visible, raw['weights'][:, None], 0)
    used = (weight > 0).any(1)
    total = weight.sum(-1)
    k = torch.where(used[..., None], raw['keys'], 0)
    v = torch.where(used[..., None], raw['values'], 0)
    if kind == 'read':
        short = weight * passive_decay(ages, c.beta_short, u)
        totals = short.sum(-1), total
        if c.memory_read == 'qry':
            q = norm(u, c.epsilon) @ memory.read_q.t()
            k, v = k @ memory.read_k.t(), v @ memory.read_v.t()
            values = tuple(attention(q[:, None], k[:, None], v[:, None], w)[:, 0]
                           * t.clamp(max=1)[..., None] for w, t in zip((short, weight), totals))
        else:
            summary = torch.where(used[..., None], raw['summaries'], 0)
            values = tuple((w @ summary) / t.clamp(min=1)[..., None]
                           for w, t in zip((short, weight), totals))
        return columns(program, memory, mass, u, (*values, *totals))
    p = program.content
    b, count, _ = u.shape
    q = (norm(u, c.epsilon) @ p.query.t()).reshape(b, count, c.heads, -1).transpose(1, 2)
    k, v = (x.reshape(b, -1, c.heads, c.width//c.heads).transpose(1, 2) for x in (k, v))
    mixed = attention(q, k, v, weight).transpose(1, 2).reshape(b, count, c.width)
    x = u + c.alpha_att * ((mixed * total.clamp(max=1)[..., None]) @ p.out.t())
    n = norm(x, c.epsilon)
    fresh = x + c.alpha_ff * ((F.silu(n @ p.gate.t()) * (n @ p.up.t())) @ p.down.t())
    return {'value': torch.cat((gather(controls, indices)[..., None], fresh), -1)}
