"""Fixed-time convolution gather with a collision-free, ordered transpose.

At one lag, strictly increasing query times select each real bank row at most
once. Different lags may reuse it. Invert that metadata once, gather cotangents
by (bank row, lag), then reduce lag in tensor order; no floating atomic scatter.
"""
import torch
from .column_layout import gather_column


class _TapGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, ids, inverse):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(values, inverse)
        return gather_column(values, ids)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("convolution tap gather supports first-order VJP only")
        values, inverse = ctx.saved_tensors
        if bar is None:
            return None, None, None
        flat = bar.flatten(0, 1)
        return gather_column(flat, inverse).sum(1), None, None


def tap_gather(values, ids):
    """``ids``: host int64 R×lag, sentinel=len(values); fields stay on device."""
    if ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.ndim != 2:
        raise ValueError("tap gather requires host int64 query/lag metadata")
    n, taps = len(values), ids.shape[1]
    if taps < 1 or (ids < 0).any() or (ids > n).any():
        raise ValueError("tap gather index outside bank/sentinel")
    if not torch.is_grad_enabled() or not values.requires_grad:
        return gather_column(values, ids.to(values.device))
    # Metadata only: avoid one tiny Tensor operation per query/tap.
    inverse = [ids.numel()]*(n*taps)
    for j, row in enumerate(ids.tolist()):
        for lag, source in enumerate(row):
            if source == n:
                continue
            slot = source*taps+lag
            if inverse[slot] != ids.numel():
                raise ValueError("convolution tap selects a bank row twice at the same lag")
            inverse[slot] = j*taps+lag
    inverse = torch.tensor(inverse, dtype=torch.int64, device=values.device).reshape(n, taps)
    return _TapGather.apply(values, ids.to(values.device), inverse)
