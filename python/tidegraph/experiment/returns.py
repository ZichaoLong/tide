"""Complete CPU observations; transfer packing never aliases returned fields."""
from dataclasses import fields, is_dataclass, replace
import torch

PACK_BYTES = 8*1024**2


def _map(value, tensor):
    if isinstance(value, torch.Tensor):
        return tensor(value)
    if is_dataclass(value):
        return replace(value, **{f.name: _map(getattr(value, f.name), tensor) for f in fields(value)})
    if isinstance(value, dict):
        return {key: _map(item, tensor) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_map(item, tensor) for item in value)
    return value


def _scalar(value):
    # An accelerator-to-CPU transfer already owns new storage. CPU inputs
    # still need a clone, including repeated occurrences and overlapping views.
    value = value.detach()
    return value.clone() if value.device.type == "cpu" else value.cpu()


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _copy_group(rows, outputs):
    """One compatible, bounded group; CPU callers also exercise exact copying."""
    if len(rows) == 1:
        value, positions = rows[0]
        for position in positions:
            outputs[position] = _scalar(value)
        return
    # All inputs are contiguous and have the same device/dtype. No casts,
    # arithmetic, nonblocking transfer, or persistent device scratch are used.
    flat = torch.cat([value.detach().reshape(-1) for value, _ in rows]).cpu()
    offset = 0
    for value, positions in rows:
        decoded = flat.narrow(0, offset, value.numel()).reshape(value.shape)
        for position in positions:
            outputs[position] = decoded.clone()
        offset += value.numel()


def snapshot(value, *, packing="scalar"):
    """Actual computed fields only; each tensor occurrence owns a CPU clone."""
    if packing not in {"scalar", "packed"}:
        raise ValueError("unknown complete-return packing")
    if packing == "scalar":
        return _map(value, _scalar)
    tensors = list(_tensors(value))
    outputs, groups = [None]*len(tensors), {}
    for position, tensor in enumerate(tensors):
        eligible = (tensor.device.type != "cpu" and tensor.layout == torch.strided
                    and not tensor.is_quantized and tensor.is_contiguous()
                    and not tensor.is_conj() and not tensor.is_neg()
                    and 0 < tensor.numel()*tensor.element_size() <= PACK_BYTES)
        if not eligible:
            outputs[position] = _scalar(tensor)
            continue
        group = groups.setdefault((tensor.device, tensor.dtype), {})
        group.setdefault(id(tensor), (tensor, []))[1].append(position)
    for group in groups.values():
        chunk, size = [], 0
        for tensor, positions in group.values():
            cost = tensor.numel()*tensor.element_size()
            if chunk and size+cost > PACK_BYTES:
                _copy_group(chunk, outputs)
                chunk, size = [], 0
            chunk.append((tensor, positions))
            size += cost
        if chunk:
            _copy_group(chunk, outputs)
    result = iter(outputs)
    return _map(value, lambda _: next(result))
