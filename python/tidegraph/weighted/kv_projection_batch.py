"""KV §1.2 / FC §6.2: batch private maps after exact shared-base preparation."""
from collections import defaultdict
from .row_linear import linear_rows


def prepare(memories, shared, stats):
    groups = defaultdict(list)
    for i, m in enumerate(memories):
        key = tuple((tuple(p.shape), p.dtype, p.device) for p in (m.adapter_v, m.adapter_u, m.summary))
        groups[key].append(i)
    result = [shared]*len(memories)
    for ids in groups.values():
        if len(ids) < 2:
            continue
        normal = [shared["normal"]]*len(ids)
        low = linear_rows(normal, [memories[i].adapter_v for i in ids])
        values = linear_rows(low, [memories[i].adapter_u for i in ids])
        summaries = linear_rows(normal, [memories[i].summary for i in ids])
        for j, i in enumerate(ids):
            result[i] = {"values": shared["base"]+values[j], "summaries": summaries[j]}
        stats["state_private_projection_calls"] = stats.get("state_private_projection_calls", 0)+3
        stats["state_private_projection_rows"] = stats.get("state_private_projection_rows", 0)+3*len(ids)
        stats["max_state_private_projection_batch"] = max(stats.get("max_state_private_projection_batch", 0), len(ids))
    return result
