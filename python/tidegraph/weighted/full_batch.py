"""KV §4 / recurrent §3: selected Full batches on complete comparison snapshots."""
from collections import defaultdict
import math
import torch
from ..full import FullResult
from .config import KVConfig
from .packing import project
from .query_batch import effective, features, weighted_read
from .tensors import norm, rotate_relative
from .normalization import scaled_rms
from .numeric_time import time_values


def kv(program, memory, states, times, u):
    c, groups, visible = program.config, defaultdict(list), []
    for i, (state, time) in enumerate(zip(states, times)):
        mask, age = memory.visible(state, time)
        visible.append((mask, age[mask]))
        groups[int(mask.sum())].append(i)
    result = [None]*len(states)
    queries = project(norm(u, c.epsilon), program.query).reshape(-1, c.heads, c.width//c.heads)
    for length, rows in groups.items():
        s = {k: torch.stack([states[i].slots[k][visible[i][0]] for i in rows]) for k in ("keys", "values", "weights")}
        keys = s["keys"].reshape(len(rows), length, c.heads, c.width//c.heads)
        values = s["values"].reshape_as(keys)
        weights, total = s["weights"], s["weights"].sum(1)
        displacement = -time_values(torch.stack([visible[i][1] for i in rows]), u)/c.position_scale
        reads = []
        for h in range(c.heads):
            key = rotate_relative(keys[:, :, h], displacement, c.full_frequencies[h])
            logits = (key*queries[rows, h][:, None]).sum(-1)/math.sqrt(c.width//c.heads)
            logits = logits+c.distance_slopes[h]*displacement
            reads.append(weighted_read(logits, values[:, :, h], weights, total))
        mixed = project(torch.cat(reads, -1), program.out)
        for i, value in zip(rows, mixed.unbind()):
            result[i] = value
    return torch.stack(result)


def recurrent(program, memory, states, times, u):
    c = program.config
    H, _, _ = effective(memory, states, times)
    if c.basic:
        q = project(norm(u, c.epsilon), program.query)
        reads = (H@q[:, None, :, None]).squeeze(-1)
        return (program.out@reads.unsqueeze(-1)).squeeze(-1).sum(1)
    f = features(memory, states, times, u)
    query = f[1] if c.memory_family == "gated_delta" else f[4]
    read = (H@query.unsqueeze(-1)).squeeze(-1)
    if c.memory_family == "gated_delta":
        rms = scaled_rms(read, program.rho, c.output_epsilon)
        gate = torch.nn.functional.silu(project(norm(u, c.epsilon), program.output_gate))
        return project(rms.flatten(1)*gate, program.out)
    y = (read+program.direct[:, None]*f[2]).flatten(1)*torch.nn.functional.silu(f[1])
    return project(scaled_rms(y, program.rho, c.output_epsilon), program.out)


def evaluate(program, weights, requests, slots, mode, zeta):
    if mode != "hard":
        raise ValueError("weighted Full has its own exact VJP; legacy Emit mode must be hard")
    if not requests:
        return []
    content, memory = program.content, weights.kernel
    c = content.config
    states, times = [r.comparison for r in requests], [r.time for r in requests]
    u = torch.stack([r.content.value[1:] for r in requests])
    if isinstance(c, KVConfig):
        mix, alpha = kv(content, memory, states, times, u), c.alpha_att
    else:
        mix, alpha = recurrent(content, memory, states, times, u), c.alpha_mem
    x = u+alpha*mix
    normal = norm(x, c.epsilon)
    fresh = x+c.alpha_ff*project(torch.nn.functional.silu(project(normal, content.gate))*project(normal, content.up), content.down)
    values = torch.cat((torch.stack([r.control for r in requests])[:, None], fresh), -1)
    return [FullResult(value, {i: value for i in range(slots)}) for value in values.unbind()]
