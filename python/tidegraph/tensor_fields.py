"""Core-5 named finite products: independent tensors and exact integer metadata.

Fields stay separate public gradient roots. Packing is an explicit representation
operation; decoding reads actual candidate buffers, never a reference recursion.
See docs/structured-records.md.
"""
from dataclasses import dataclass
import math
import torch


def validate_fields(fields, layout, dtype, device, *, check_finite=True):
    if not isinstance(fields, dict) or fields.keys() != layout.keys():
        raise ValueError("structured Read field domain mismatch")
    for name, shape in layout.items():
        if (not isinstance(name, str) or not name or not isinstance(shape, tuple)
                or any(type(n) is not int or n < 0 for n in shape)):
            raise ValueError("invalid structured Read field layout")
        value = fields[name]
        if not isinstance(value, torch.Tensor) or (value.shape, value.dtype, value.device) != (shape, dtype, device):
            raise ValueError("structured Read field metadata mismatch")
        if check_finite and not torch.isfinite(value).all():
            raise ValueError("nonfinite structured Read field")


@dataclass(frozen=True)
class PackedFields:
    """One record, separate dtype buffers; optional original gradient anchors.

    `pack_fields` keeps the actually computed original fields as VJP anchors.
    Deserialization may omit them and starts an explicit new gradient segment.
    No floating representation ever contains int64 timestamps or counters.
    """
    layout: tuple[tuple[str, tuple[int, ...], torch.dtype, int, int], ...]
    buffers: tuple[torch.Tensor, ...]
    roots: dict[str, torch.Tensor] | None = None

    def decode(self):
        from .autograd import value
        offsets = [0] * len(self.buffers)
        names = set()
        for name, shape, dtype, bank, offset in self.layout:
            if (not isinstance(name, str) or not name or name in names
                    or type(bank) is not int or not 0 <= bank < len(offsets)
                    or type(offset) is not int or offset != offsets[bank]
                    or not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape)):
                raise ValueError("invalid packed field layout")
            names.add(name); offsets[bank] += math.prod(shape)
        if (any(buffer.ndim != 1 or len(buffer) != end for buffer, end in zip(self.buffers, offsets))
                or self.roots is not None and self.roots.keys() != names):
            raise ValueError("packed field buffer/root domain mismatch")
        result = {}
        for name, shape, dtype, bank, offset in self.layout:
            buffer = self.buffers[bank]
            if buffer.dtype != dtype or buffer.ndim != 1:
                raise ValueError("packed field dtype/shape mismatch")
            field = buffer.narrow(0, offset, math.prod(shape)).reshape(shape)
            result[name] = value(field, self.roots[name]) if self.roots is not None else field
        return result


def pack_fields(fields):
    """Pack computed fields by dtype/device without merging their VJP roots."""
    if not isinstance(fields, dict) or not fields:
        raise ValueError("packing requires nonempty named fields")
    groups, layouts = {}, []
    for name, field in sorted(fields.items()):
        if not isinstance(name, str) or not name or not isinstance(field, torch.Tensor):
            raise ValueError("invalid named tensor field")
        key = field.device, field.dtype
        if key not in groups:
            groups[key] = (len(groups), [], 0)
        bank, rows, offset = groups[key]
        layouts.append((name, tuple(field.shape), field.dtype, bank, offset))
        rows.append(field.reshape(-1)); groups[key] = bank, rows, offset+field.numel()
    # Numeric packed buffers have no incidental dependency on unused fields.
    buffers = tuple(torch.cat(rows).detach() for _, rows, _ in groups.values())
    return PackedFields(tuple(layouts), buffers, dict(fields))
