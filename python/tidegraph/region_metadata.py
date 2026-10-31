"""Read-only control flags for already complete region batches.

Only detached bool metadata crosses to the host. Floating controls, their
aliases and VJP roots remain untouched; CPU and singleton reads stay scalar.
"""
import torch


def _group_flags(values, *, positive):
    if positive or all(value.numel() == 1 for value in values):
        packed = torch.stack([value.detach().reshape(()) for value in values])
        flags = packed > 0 if positive else torch.isfinite(packed)
    else:
        # Custom selectors may return vector or empty controls. Keep their
        # per-control all-elements predicate, batching only the bool return.
        flags = torch.stack([torch.isfinite(value.detach()).all() for value in values])
    return flags.to(device="cpu").tolist()


def _flags(values, *, positive):
    groups, result = {}, [None]*len(values)
    for index, value in enumerate(values):
        groups.setdefault((value.device, value.dtype), []).append(index)
    for (device, _), ids in groups.items():
        if device.type == "cpu" or len(ids) == 1:
            flags = [bool(values[i].detach() > 0) if positive else bool(torch.isfinite(values[i]).all()) for i in ids]
        else:
            flags = _group_flags([values[i] for i in ids], positive=positive)
        for index, flag in zip(ids, flags):
            result[index] = flag
    return result


def positive(values):
    return _flags(values, positive=True)


def finite(values):
    return _flags(values, positive=False)
