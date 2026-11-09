"""Architecture §3: batched Mem, per-row finite descriptor operation order.

RMS-normalized heads are sensitive to projection rounding in their near-zero
VJPs. Keep their scalar arithmetic while batching complete memory queries.
"""
import torch
from .query_batch import memory_read
from .tensors import norm


def evaluate(program, weights, requests):
    if not requests:
        return []
    memory = weights.kernel
    from .query_execution import dense, packed, dense_heads
    states = [memory.initial(weights) if r.state is None else r.state for r in requests]
    times = [r.time for r in requests]
    payload = torch.stack([r.content.value for r in requests])
    mass, u = payload[:, 0], payload[:, 1:]
    if packed(program):
        from .query_packed import memory as query
        values = query(memory, states, times, u)
    else:
        values = memory_read(memory, states, times, u, dense=dense(program))
    if dense_heads(program):
        from .dense_describe import evaluate as describe
        return describe(program, memory, mass, u, values, scalar_evidence=packed(program))
    result = []
    for i in range(len(requests)):
        short, long, ws, wl = (value[i] for value in values)
        current = memory.current(u[i])
        cur = norm(u[i], memory.config.epsilon)
        stat = torch.stack((mass[i].log1p(), ws.log1p(), wl.log1p()))
        groups = dict(cur=cur, mem=torch.cat((short, long)), stat=stat)
        features = torch.cat((cur, stat[:1], short, long, stat[1:], current*short, current*long))
        result.append(program.describe(mass[i], features, groups))
    return result
