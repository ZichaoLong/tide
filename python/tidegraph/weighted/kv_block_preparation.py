"""KV §1.2 / FC §6.2: batch both sample and time without copying weight banks.

Only input and parameter identity license shared computation. Same-parameter
rows use matrix multiplication; singleton workloads retain the existing BMM
path. Positive-write metadata is transferred once per device/dtype group.
"""
from collections import defaultdict
import torch
from .preparation_rows import prepare_rows

PACK_BYTES = 8*1024**2


def _private(memory):
    return tuple(id(p) for p in (memory.adapter_v, memory.adapter_u, memory.summary))


def prepare(memories, contents, stats):
    repeats = defaultdict(int)
    for memory in memories:
        repeats[_private(memory)] += 1
    if not repeats or max(repeats.values()) < 2:
        return None
    metadata = defaultdict(list)
    for i, h in enumerate(contents):
        metadata[h.device, h.dtype].append(i)
    positive = []
    for ids in metadata.values():
        flags = (torch.stack([contents[i][0] for i in ids]).detach() > 0).cpu().tolist()
        positive.extend(i for i, flag in zip(ids, flags) if flag)
    positive.sort()
    common, private = {}, {}
    for i in positive:
        memory = memories[i]
        common.setdefault((id(memory.base), memory.config.epsilon), {}).setdefault(id(contents[i]), []).append(i)
        private.setdefault(_private(memory), []).append(i)
    operands, result = {}, [{} for _ in memories]
    for group in common.values():
        rows = list(group.values())
        m = memories[rows[0][0]]
        limit = max(1, PACK_BYTES//(4*(3*m.config.width+1)*m.base.element_size()))
        for start in range(0, len(rows), limit):
            chunk = rows[start:start+limit]
            values = prepare_rows("base", [m.base], [(contents[ids[0]],) for ids in chunk], m.config.epsilon)
            stats["state_block_base_projection_calls"] = stats.get("state_block_base_projection_calls", 0)+1
            stats["max_state_block_base_batch"] = max(stats.get("max_state_block_base_batch", 0), len(chunk))
            for ids, pair in zip(chunk, values):
                for i in ids:
                    operands[i] = pair
                if len(ids) > 1:
                    for key, value in (("calls", 1), ("events", len(ids)), ("reused_events", len(ids)-1)):
                        name = "state_preparation_"+key
                        stats[name] = stats.get(name, 0)+value
    for ids in private.values():
        m = memories[ids[0]]
        limit = max(1, PACK_BYTES//(4*(5*m.config.width+m.config.summary_width+m.config.adapter_rank)*m.base.element_size()))
        for start in range(0, len(ids), limit):
            chunk = ids[start:start+limit]
            values = prepare_rows("private", [m.adapter_v, m.adapter_u, m.summary], [operands[i] for i in chunk])
            for i, (v, s) in zip(chunk, values):
                result[i] = dict(values=v, summaries=s)
            for key, value in (("calls", 3), ("rows", 3*len(chunk))):
                name = "state_private_projection_"+key
                stats[name] = stats.get(name, 0)+value
            stats["max_state_private_projection_batch"] = max(stats.get("max_state_private_projection_batch", 0), len(chunk))
            stats["max_state_independent_projection_batch"] = max(stats.get("max_state_independent_projection_batch", 0), len(chunk))
    stats["state_block_preparation_events"] = stats.get("state_block_preparation_events", 0)+len(memories)
    return result
