"""Fixed identity transport: bounded value guards, no per-scale device reads.

Architecture §2's unit transport does not add a trainable multiplier. This
guard changes only validation granularity; every invocation reads actual values.
Shape/dtype/device binding remains the caller's responsibility. Empty tensors
have the same vacuous all-ones value as ``equal(x, ones_like(x))``.
"""
from collections import defaultdict
import torch


def _bounded_views(value, capacity):
    """Split before reshape so a large strided operand cannot allocate a copy."""
    count = value.numel()
    if not count:
        return
    if count <= capacity:
        yield value
    elif value.is_contiguous():
        flat = value.view(-1)
        for start in range(0, count, capacity):
            yield flat.narrow(0, start, min(capacity, count-start))
    else:
        axis = next(i for i, size in enumerate(value.shape) if size > 1)
        width = max(1, capacity//(count//value.shape[axis]))
        for start in range(0, value.shape[axis], width):
            yield from _bounded_views(value.narrow(axis, start, min(width, value.shape[axis]-start)), capacity)


def validate_unit_scales(values, *, budget=8*1024**2, fixed=True,
                         message="weighted payload transport must use fixed unit scales"):
    """One value comparison per device/dtype chunk; reject every invalid scale.

    ``budget`` bounds explicit temporary tensor bytes: reshape copies, the
    packed operand and its all-ones peer (at most three equal-sized buffers).
    Backend workspace/allocator caching is separate. No state survives this
    call. ``fixed=False`` checks checkpoint values without prescribing whether
    their temporary source tensors require gradients, as before.
    """
    if type(budget) is not int or not 0 < budget < 2**63:
        raise ValueError("unit transport budget must be a positive int64 byte count")
    groups = defaultdict(list)
    for value in values:
        if fixed and value.requires_grad:
            raise ValueError(message)
        groups[value.device, value.dtype].append(value)
    with torch.no_grad():
        for tensors in groups.values():
            capacity = budget//(3*tensors[0].element_size())
            if capacity < 1 and any(t.numel() for t in tensors):
                raise ValueError("unit transport budget cannot hold one element")
            rows, count = [], 0

            def check():
                packed = torch.cat([row.reshape(-1) for row in rows])
                if not torch.equal(packed, torch.ones_like(packed)):
                    raise ValueError(message)

            for tensor in tensors:
                for view in _bounded_views(tensor, capacity):
                    if rows and count+view.numel() > capacity:
                        check()
                        rows, count = [], 0
                    rows.append(view)
                    count += view.numel()
            if rows:
                check()
