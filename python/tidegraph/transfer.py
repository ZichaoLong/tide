"""Bounded tensor transport with independent public output cotangents."""
import torch

PACK_BYTES = 8 * 1024**2


class _Rows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, destination, *rows):
        ctx.set_materialize_grads(False)
        ctx.source = rows[0].device
        result = tuple(torch.stack(rows).to(destination).unbind())
        ctx.mark_non_differentiable(*(v for v, r in zip(result, rows) if not r.requires_grad))
        return result

    @staticmethod
    def backward(ctx, *bars):
        result = [None] * (len(bars) + 1)
        used = [i for i, g in enumerate(bars) if g is not None and ctx.needs_input_grad[i+1]]
        if used:
            values = copy_rows([bars[i] for i in used], ctx.source)
            for i, value in zip(used, values):
                result[i+1] = value
        return tuple(result)


def copy_rows(rows, destination, budget=PACK_BYTES):
    """Copy equal vector rows in bounded groups, retaining None versus zero VJPs.

    Same-device calls are useful independent anchors; scheduling only invokes
    this packer for actual remote messages. A single row uses ordinary Tensor.to.
    A row larger than the packing budget remains an indivisible scalar copy.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("transfer packing budget must be positive")
    if not rows:
        return []
    first = rows[0]
    if (first.ndim != 1 or first.numel() == 0 or first.layout != torch.strided
            or first.dtype not in (torch.float16, torch.float32, torch.float64)):
        raise ValueError("transfer requires nonempty FP16/FP32/FP64 vector rows")
    if any(r.shape != first.shape or r.device != first.device or r.dtype != first.dtype
           or r.layout != torch.strided for r in rows):
        raise ValueError("transfer row metadata mismatch")
    limit = max(1, budget // (first.numel()*first.element_size()))
    result = []
    for start in range(0, len(rows), limit):
        group = rows[start:start+limit]
        result.extend(_Rows.apply(destination, *group) if len(group) > 1 else [group[0].to(destination)])
    return result


def deliver_remote(atoms, destinations, packed, stats):
    """Assign copied payloads without reordering or coalescing atom identities."""
    groups = {}
    for i, (atom, destination) in enumerate(zip(atoms, destinations)):
        value = atom.value
        if value.device != destination:
            key = (value.device, destination, value.dtype, tuple(value.shape))
            groups.setdefault(key, []).append(i)
    for (_, destination, _, _), ids in groups.items():
        row = atoms[ids[0]].value
        size = row.numel()*row.element_size()
        limit = max(1, PACK_BYTES//size) if packed else 1
        for start in range(0, len(ids), limit):
            selected = ids[start:start+limit]
            values = copy_rows([atoms[i].value for i in selected], destination)
            for i, value in zip(selected, values):
                atoms[i].value = value
            for name, count in (("cross_device_copy_groups", 1), ("cross_device_rows", len(selected)),
                                ("cross_device_bytes", len(selected)*size)):
                stats[name] = stats.get(name, 0)+count
            stats["max_cross_device_batch"] = max(stats.get("max_cross_device_batch", 0), len(selected))
