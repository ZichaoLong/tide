"""KV §2/§5: joint write projections and actual ragged prefix caches."""
import torch
from bisect import bisect_left
from .packing import Writes
from .kv_projection import projections


def packed(memory, old, batch):
    plan = Writes(old, batch)
    if not plan.ids:
        return plan.decode(old, batch, []), 0
    c = memory.config
    values, summaries = projections(memory, plan, batch)
    key, value = values.chunk(2, -1)
    entries = dict(times=plan.times, keys=key, values=value,
                   summaries=summaries, weights=plan.weight)
    # Raw timestamps are sorted and every positive write follows the old state.
    # Only windows that can expire an entry need a device metadata download.
    needs = [i for i in range(len(old)) if plan.offsets[i] < plan.offsets[i+1]
             and c.window and batch.times[plan.ids[plan.offsets[i+1]-1]] >= c.window]
    stamps, offset = {}, 0
    if needs:
        host = torch.cat([old[i].slots["times"] for i in needs]).cpu().tolist()
        for i in needs:
            n = old[i].slots["times"].numel()
            stamps[i] = host[offset:offset+n]
            offset += n
    slots = []
    for i, state in enumerate(old):
        a, b = plan.offsets[i:i+2]
        if a == b:
            continue
        # Concatenate once per owner. Every result is a causal index selection
        # from these actual projections, never a reference recurrence.
        joined = {name: torch.cat((state.slots[name], v[a:b])) for name, v in entries.items()}
        n = state.slots["times"].numel()
        clock = stamps.get(i, [])+[batch.times[plan.ids[j]] for j in range(a, b)]
        for j in range(a, b):
            end = n+j-a+1
            start = bisect_left(clock, batch.times[plan.ids[j]]-c.window+1, 0, end) if i in stamps else 0
            slots.append({name: v[start:end] for name, v in joined.items()})
    return plan.decode(old, batch, slots), 1
