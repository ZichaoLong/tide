"""Causal-convolution §1–3: joint absolute-time gathers and exact raw caches."""
import torch
from .convolution import ConvCache


def convolve(conv, plan, old, raw):
    if not torch.isfinite(raw).all():
        raise ValueError("invalid convolution query value")
    base = conv.bias+conv.weight[0]*raw
    rows, caches = [], []
    taps = conv.stride*torch.arange(1, conv.size, dtype=torch.int64, device=raw.device)
    for i, state in enumerate(old):
        a, b = plan.offsets[i:i+2]
        if a == b:
            continue
        cache = ConvCache(state.slots["conv_times"], state.slots["conv_values"])
        cache.validate(conv.width, conv.horizon)
        if (cache.times >= plan.times[a]).any():
            raise ValueError("convolution writes require increasing logical time")
        times = torch.cat((state.slots["conv_times"], plan.times[a:b]))
        values = torch.cat((state.slots["conv_values"], plan.weight[a:b, None]*raw[a:b]))
        # A single write with empty left cache has no history dependency in
        # the scalar function. Do not add a spurious zero VJP to empty slots.
        if conv.size > 1 and (b-a > 1 or len(cache.times)):
            queries = plan.times[a:b, None]-taps
            indices = torch.searchsorted(times, queries)
            safe = indices.clamp(max=len(times)-1)
            present = (indices < len(times)) & (times[safe] == queries) & (queries >= 0)
            history = values[safe]*present.unsqueeze(-1)
            rows.append(base[a:b]+(conv.weight[1:]*history).sum(-2))
        else:
            rows.append(base[a:b])
        for j in range(a, b):
            visible = (times >= plan.times[j]-conv.horizon) & (times <= plan.times[j])
            caches.append(dict(conv_times=times[visible], conv_values=values[visible]))
    return torch.cat(rows), caches
