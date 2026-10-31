"""Exact finite rejection with an explicit bounded diagnostic packing policy."""
import torch
from .returns import _tensors

PACK_BYTES = 8*1024**2


def finite_tensors(value, *, packing="scalar"):
    """Check every floating/complex leaf without changing any mathematical value.

    Packing only joins compatible contiguous diagnostic inputs. Large/special
    tensors keep the scalar check. Flags still cross the host boundary once per
    device; no persistent buffer, owner binding or autograd graph is cached.
    """
    if packing not in {"scalar", "packed"}:
        raise ValueError("finite_check_packing: choose scalar or packed")
    from ..comparison import NonFiniteMismatch
    flags, groups, seen = {}, {}, set()

    def check(tensor):
        flags.setdefault(tensor.device, []).append(torch.isfinite(tensor.detach()).all())

    def flush(rows):
        if len(rows) == 1:
            check(rows[0])
        elif rows:
            check(torch.cat([row.detach().reshape(-1) for row in rows]))

    with torch.autograd.profiler.record_function("tide.finite_check"):
        for tensor in _tensors(value):
            if id(tensor) in seen or not (tensor.is_floating_point() or tensor.is_complex()):
                continue
            seen.add(id(tensor))
            size = tensor.numel()*tensor.element_size()
            eligible = (packing == "packed" and tensor.layout == torch.strided
                and not tensor.is_quantized and tensor.is_contiguous()
                and not tensor.is_conj() and not tensor.is_neg() and 0 < size <= PACK_BYTES)
            if not eligible:
                check(tensor)
                continue
            rows, used = groups.get((tensor.device, tensor.dtype), ([], 0))
            if rows and used+size > PACK_BYTES:
                flush(rows)
                rows, used = [], 0
            rows.append(tensor)
            groups[tensor.device, tensor.dtype] = rows, used+size
        for rows, _ in groups.values():
            flush(rows)
        if any(not bool(torch.stack(rows).all()) for rows in flags.values()):
            raise NonFiniteMismatch("runtime", "nonfinite mathematical tensor")
