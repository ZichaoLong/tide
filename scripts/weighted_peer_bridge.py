"""Explicit diagnostic CPU bridge; never a default transfer substitution.

Each output keeps its own cotangent and original version. Host staging changes
communication/synchronization costs, so passing does not clear peer async.
"""
from contextlib import contextmanager
from unittest.mock import patch
import torch


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, device, dtype, counts):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(value)
        ctx.counts = counts
        counts["forward_calls"] += 1
        counts["forward_bytes"] += value.numel()*value.element_size()
        return value.cpu().to(device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("CPU bridge supports first-order VJP only")
        value, = ctx.saved_tensors
        if bar is None:
            return None, None, None, None
        ctx.counts["backward_calls"] += 1
        ctx.counts["backward_bytes"] += bar.numel()*bar.element_size()
        return bar.cpu().to(value), None, None, None


@contextmanager
def cpu_bridge():
    import tidegraph.transfer as transport
    import tidegraph.weighted.placement_tensors as placed
    import tidegraph.weighted.placement as placement
    counts = dict(forward_calls=0, backward_calls=0, forward_bytes=0, backward_bytes=0)
    def move(value, device, dtype=None):
        if value is None:
            return None
        device = torch.device(device)
        dtype = (dtype or value.dtype) if value.is_floating_point() else value.dtype
        if value.device == device and value.dtype == dtype:
            return value
        # CPU endpoints already break peer dependencies; all explicit copies
        # use the same first-order boundary so versions/None roots are checked.
        return _Copy.apply(value, device, dtype, counts)
    def mapper(values, device, dtype):
        copied = {id(x): move(x, device, dtype) for x in {id(t): t for t in values}.values()}
        return lambda x: copied[id(x)]
    with patch.object(transport, "copy_rows", lambda rows, destination, budget=transport.PACK_BYTES:
                      [move(x, destination) for x in rows]), patch.object(placed, "move", move), \
            patch.object(placed, "mapper", mapper), patch.object(placement, "mapper", mapper):
        yield counts
