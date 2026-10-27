"""KV §2/§5: joint write projections and actual ragged prefix caches."""
import torch
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
    slots = []
    for i, state in enumerate(old):
        a, b = plan.offsets[i:i+2]
        if a == b:
            continue
        # Concatenate once per owner. Every result is a causal index selection
        # from these actual projections, never a reference recurrence.
        joined = {name: torch.cat((state.slots[name], v[a:b])) for name, v in entries.items()}
        for j in range(a, b):
            age = plan.times[j]-joined["times"]
            visible = age >= 0
            if c.window:
                visible = visible & (age < c.window)
            slots.append({name: v[visible] for name, v in joined.items()})
    return plan.decode(old, batch, slots), 1
