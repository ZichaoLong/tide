"""Copy publication batches while keeping storage and version counters private."""
import torch


def detached_copies(values):
    # A clone of a whole batch followed by views would couple output versions
    # and resizing. Foreach copies instead own truly independent allocations.
    result, groups = [None]*len(values), {}
    for i, value in enumerate(values):
        if value.layout != torch.strided:
            result[i] = value.detach().clone()
            continue
        groups.setdefault((value.device, value.dtype), []).append(i)
    with torch.no_grad():
        for ids in groups.values():
            sources = [values[i].detach() for i in ids]
            outputs = [torch.empty_like(t) for t in sources]
            torch._foreach_copy_(outputs, sources)
            for i, value in zip(ids, outputs):
                result[i] = value
    return tuple(result)
