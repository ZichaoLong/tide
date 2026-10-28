"""KV §1.2 / FC §6.2: base projection reuse with node-private adapters.

Only actual common input/parameter handles license reuse. Scalars and unpacked
execution continue using the textbook combined matrix as independent anchors.
"""
import torch
from .tensors import norm
from .packing import project


def key(memory, content):
    if not bool(content[0] > 0):
        return None
    return id(memory.base), id(content), memory.config.epsilon


def prepare(memory, content):
    x = norm(content[1:], memory.config.epsilon)
    return {"normal": x, "base": memory.base@x}


def projections(memory, plan, batch):
    """Return actual write rows in positive-write order; no state reconstruction."""
    prepared = getattr(batch, "prepared", None)
    common, independent = [], []
    results = [None]*len(plan.ids)
    summaries = [None]*len(plan.ids)
    for k, j in enumerate(plan.ids):
        row = prepared[j] if prepared else {}
        if set(row) == {"values", "summaries"}:
            v, s = row["values"], row["summaries"]
            if v.shape != (2*memory.config.width,) or s.shape != (memory.config.summary_width,):
                raise ValueError("invalid KV private preparation shape")
            if any((t.dtype, t.device) != (plan.content.dtype, plan.content.device) for t in (v, s)):
                raise ValueError("invalid KV private preparation dtype/device")
            results[k], summaries[k] = v, s
        else:
            (common if row else independent).append((k, j))
    if len(independent) == len(plan.ids):
        x = norm(plan.content, memory.config.epsilon)
        return (project(x, memory.base+memory.adapter_u@memory.adapter_v),
                project(norm(plan.content, memory.config.epsilon), memory.summary))
    if independent:
        ids = [k for k, _ in independent]
        x = norm(plan.content[ids], memory.config.epsilon)
        values = project(x, memory.base+memory.adapter_u@memory.adapter_v)
        local = project(norm(plan.content[ids], memory.config.epsilon), memory.summary)
        for r, (k, _) in enumerate(independent):
            results[k], summaries[k] = values[r], local[r]
    if common:
        if any(set(prepared[j]) != {"normal", "base"} for _, j in common):
            raise ValueError("invalid KV preparation fields")
        if any((v.dtype, v.device) != (plan.content.dtype, plan.content.device)
               for _, j in common for v in prepared[j].values()):
            raise ValueError("invalid KV preparation dtype/device")
        x = torch.stack([prepared[j]["normal"] for _, j in common])
        base = torch.stack([prepared[j]["base"] for _, j in common])
        if x.shape != (len(common), memory.config.width) or base.shape != (len(common), 2*memory.config.width):
            raise ValueError("invalid KV preparation shape")
        # FC §6.2: the adapter is a low-rank map, not a dense temporary.
        # The unpacked combined matrix remains an independent numerical anchor.
        values = base+project(project(x, memory.adapter_v), memory.adapter_u)
        local = project(x, memory.summary)
        for r, (k, _) in enumerate(common):
            results[k], summaries[k] = values[r], local[r]
    return torch.stack(results), torch.stack(summaries)
