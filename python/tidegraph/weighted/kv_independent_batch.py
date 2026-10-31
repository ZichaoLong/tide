"""KV §1.2: bounded BMM for distinct content/base rows, without value reuse."""
from collections import defaultdict
from .row_linear import linear_rows
from .tensors import norm
from . import packing_observation as observation

PACK_BYTES = 8 * 1024**2


def prepare(memories, contents, stats, budget=PACK_BYTES):
    if len(memories) != len(contents) or type(budget) is not int or budget < 1:
        raise ValueError("invalid independent KV preparation rows/budget")
    result = [{} for _ in contents]
    groups = defaultdict(list)
    for i, (m, h) in enumerate(zip(memories, contents)):
        if not bool(h[0] > 0):
            continue
        weights = (m.base, m.adapter_v, m.adapter_u, m.summary)
        key = tuple((tuple(p.shape), p.dtype, p.device) for p in weights)
        groups[key].append(i)
    for ids in groups.values():
        m = memories[ids[0]]
        # Bound stacked operands/outputs and their saved first-order buffers.
        # A row exceeding this pack budget keeps its existing single-row path.
        row_bytes = 4 * sum(p.numel()*p.element_size() for p in
                            (m.base, m.adapter_v, m.adapter_u, m.summary))
        limit = budget // max(1, row_bytes)
        if limit < 2:
            observation.skipped("independent_projection_budget", len(ids))
            continue
        for start in range(0, len(ids), limit):
            group = ids[start:start+limit]
            if len(group) < 2:
                observation.skipped("independent_projection_singleton", len(group))
                continue
            normal = [norm(contents[i][1:], memories[i].config.epsilon) for i in group]
            base = linear_rows(normal, [memories[i].base for i in group])
            low = linear_rows(normal, [memories[i].adapter_v for i in group])
            private = linear_rows(low, [memories[i].adapter_u for i in group])
            summary = linear_rows(normal, [memories[i].summary for i in group])
            for j, i in enumerate(group):
                result[i] = dict(values=base[j]+private[j], summaries=summary[j])
            stats["state_independent_projection_calls"] = stats.get("state_independent_projection_calls", 0)+4
            stats["state_independent_projection_rows"] = stats.get("state_independent_projection_rows", 0)+4*len(group)
            stats["max_state_independent_projection_batch"] = max(
                stats.get("max_state_independent_projection_batch", 0), len(group))
    return result
