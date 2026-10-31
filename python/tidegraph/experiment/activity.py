"""Untimed host metadata from the candidate's already decoded complete result."""
from collections import Counter
import torch


def activity(result):
    count = result.continuation.batch_size
    rows = [dict(sample=i, candidates=0, full=0, received_atoms=0, messages=0,
                 outputs=0, pending=0, stored_kv_owners=0) for i in range(count)]
    caches = [Counter() for _ in rows]
    for event in result.trace:
        row = rows[event["batch"]]
        row["candidates"] += 1
        row["full"] += int(event["active"])
        row["received_atoms"] += len(event.get("fiber", ()))
    for name, atoms in (("messages", result.messages), ("pending", result.continuation.pending)):
        for atom in atoms:
            rows[atom.batch][name] += 1
    for sample, *_ in result.outputs:
        rows[sample]["outputs"] += 1
    for (sample, _), state in result.continuation.states.items():
        times = state.slots.get("times")
        if times is None:
            continue
        if times.device.type != "cpu" or times.dtype != torch.int64 or times.ndim != 1:
            raise ValueError("activity observation requires decoded CPU int64 KV timestamps")
        caches[sample][times.numel()] += 1
        rows[sample]["stored_kv_owners"] += 1
    total = Counter()
    for row, cache in zip(rows, caches):
        row["raw_kv_cache_lengths"] = {str(n): number for n, number in sorted(cache.items())}
        total.update(cache)
    return dict(scope="decoded complete return; raw stored KV lengths, not visible query buckets; outside window timing",
                samples=rows, raw_kv_cache_lengths={str(n): number for n, number in sorted(total.items())})
