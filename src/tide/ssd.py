"""Tensor primitives for the SSD binding at upstream 4f5b9f20.

The SSD equations are (7), (8), and (12)--(13) of
local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md.
Convolution follows equations (2)--(3) of causal-convolution-learning-note.md.
These functions update every positive reception; selection belongs to the caller.
"""

from typing import NamedTuple

import torch
from torch import Tensor


def _floating_like(reference: Tensor, *others: Tensor) -> None:
    if not reference.is_floating_point():
        raise TypeError("SSD values must be floating point")
    if any(x.dtype != reference.dtype or x.device != reference.device for x in others):
        raise ValueError("SSD values must have the same dtype and device")


def affine_prefix(decay: Tensor, writes: Tensor, initial: Tensor) -> tuple[Tensor, Tensor]:
    """Parallel affine recurrence for vector summaries, including zero decay.

    decay [..., L, H], writes [..., L, H, V], initial [..., H, V]. Compose
    (a, b): x -> a*x+b in log2(L) batched stages. No division, logarithm or
    growing inverse decay is used. This avoids the K=1 summary's pairwise
    block matrices; the full SSD matrix recurrence remains ssd_scan.
    """
    if decay.ndim < 2 or writes.shape[:-1] != decay.shape:
        raise ValueError("expected decay [..., L, H] and writes [..., L, H, V]")
    if initial.shape != writes.shape[:-3] + writes.shape[-2:]:
        raise ValueError("initial affine summary shape disagrees with writes")
    _floating_like(decay, writes, initial)
    length = decay.shape[-2]
    if length == 0:
        return writes, initial
    a, b = decay, writes
    offset = 1
    while offset < length:
        following_a = a[..., offset:, :] * a[..., :-offset, :]
        following_b = b[..., offset:, :, :] + a[..., offset:, :, None] * b[..., :-offset, :, :]
        a = torch.cat((a[..., :offset, :], following_a), -2)
        b = torch.cat((b[..., :offset, :, :], following_b), -3)
        offset *= 2
    values = b + a[..., None] * initial.unsqueeze(-3)
    return values, values[..., -1, :, :]


def ssd_scan(
    decay: Tensor,
    write_value: Tensor,
    B: Tensor,
    C: Tensor,
    initial: Tensor,
    *,
    chunk_size: int = 32,
    backend: str = "block",
) -> tuple[Tensor, Tensor]:
    """Return all matrix reads and the final SSD matrix, without a state per row.

    ``decay`` has shape ``[..., L, H]`` and values in [0, 1]. ``write_value``
    is ``[..., L, H, V]`` and already includes the effective step ``dt``.
    ``B`` and ``C`` are ``[..., L, H, K]``; callers expand group vectors over
    their heads. ``initial`` is ``[..., H, V, K]``. The recurrence is
    ``state[t] = decay[t] * state[t-1] + write_value[t] outer B[t]``.
    Reads have shape ``[..., L, H, V]``; final has the initial shape.

    A block contracts its causal decay matrix with C/B and the write values.
    Only block boundaries carry a V by K state. Pairwise workspace is at most
    32 by 32 per head; Python loops over blocks, never samples or positions.
    Direct products also handle a decay rounded to zero without log(0)/0/0.
    Nonzero passive decay, if selected, is included in ``decay`` by the caller.
    An absent reception uses decay=1 and write_value=0.

    ``block`` retains cumulative products. ``incremental`` uses the direct
    recurrence for L=1 and otherwise retains ``block``. ``doubling`` also uses
    direct L=1, and replaces block cumulative products by multiplication at
    doubling offsets. No variant divides by decay, preserving its derivative
    even at zero. Doubling bounds each stage to the same block, but autograd
    can retain O(chunk_size**2 * log(chunk_size)) products per head per block.
    Candidate choice is explicit; device performance is shape dependent.
    """
    if not isinstance(chunk_size, int) or not 1 <= chunk_size <= 32:
        raise ValueError("chunk_size must be an integer between 1 and 32")
    if backend not in ("block", "incremental", "doubling"):
        raise ValueError("SSD scan backend must be block, incremental or doubling")
    if decay.ndim < 2 or write_value.ndim != decay.ndim + 1:
        raise ValueError("expected decay [..., L, H] and write_value [..., L, H, V]")
    if write_value.shape[:-1] != decay.shape or B.shape != C.shape:
        raise ValueError("SSD sequence shapes disagree")
    if B.ndim != write_value.ndim or B.shape[:-1] != decay.shape:
        raise ValueError("B and C must have shape [..., L, H, K]")
    expected = decay.shape[:-2] + (decay.shape[-1], write_value.shape[-1], B.shape[-1])
    if initial.shape != expected:
        raise ValueError("initial SSD matrix has the wrong shape")
    if min(decay.shape[-1], write_value.shape[-1], B.shape[-1]) < 1:
        raise ValueError("head, value, and state dimensions must be positive")
    _floating_like(decay, write_value, B, C, initial)
    length = decay.shape[-2]
    if length == 0:
        return write_value, initial
    if length == 1 and backend != "block":
        # Settle continuation often supplies one position. Avoid propagation,
        # masks and packing, while keeping node/sample/head axes batched and
        # the incoming state connected to the complete backward graph.
        state = (decay[..., 0, :, None, None] * initial
                 + write_value[..., 0, :, :, None] * B[..., 0, :, None, :])
        reads = (state @ C[..., 0, :, :, None]).squeeze(-1).unsqueeze(-3)
        return reads, state

    # Move heads ahead of time for the block matrix products.
    d, values = decay.transpose(-1, -2), write_value.transpose(-2, -3)
    keys, queries = B.transpose(-2, -3), C.transpose(-2, -3)
    state, outputs = initial, []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        a, v = d[..., start:stop], values[..., start:stop, :]
        b, c = keys[..., start:stop, :], queries[..., start:stop, :]
        order = torch.arange(stop - start, device=decay.device)
        lower = order[:, None] > order[None, :]
        factors = torch.where(lower, a.unsqueeze(-1), 1)
        if backend == "doubling":
            products = factors
            offset = 1
            while offset < stop - start:
                products = torch.cat((products[..., :offset, :],
                                      products[..., offset:, :] * products[..., :-offset, :]), -2)
                offset *= 2
            propagation = products.masked_fill(~(order[:, None] >= order[None, :]), 0)
            prefix = propagation[..., :, 0] * a[..., :1]
        else:
            propagation = factors.cumprod(-2).masked_fill(~(order[:, None] >= order[None, :]), 0)
            prefix = a.cumprod(-1)
        local = (torch.matmul(c, b.transpose(-1, -2)) * propagation) @ v
        carried = torch.matmul(c, state.transpose(-1, -2)) * prefix.unsqueeze(-1)
        outputs.append((local + carried).transpose(-2, -3))
        weighted_b = b * propagation[..., -1, :].unsqueeze(-1)
        state = (state * prefix[..., -1, None, None]
                 + torch.matmul(v.transpose(-1, -2), weighted_b))
    return torch.cat(outputs, dim=-3), state


class ConvState(NamedTuple):
    """A lossless sparse time-slot cache with fixed-width padded rows.

    ``times`` is int64 ``[..., S]``, sorted with unused -1 slots first.
    ``values`` is ``[..., S, P]`` and stores weighted historical projections.
    Slots with times=-1 are padding, not zero-valued messages.
    """

    times: Tensor
    values: Tensor


def logical_conv(
    projected: Tensor,
    mass: Tensor,
    times: Tensor,
    kernel: Tensor,
    bias: Tensor,
    state: ConvState | None = None,
    *,
    stride: int = 1,
    positive=None,
    weighted_writes=None,
) -> tuple[Tensor, ConvState]:
    """Convolve a complete window on its original integer logical clock.

    Inputs are projected ``[..., L, P]``, clipped reception mass ``[..., L]``
    in [0, 1], and strictly increasing nonnegative int64 ``times[..., L]``.
    The window follows the cache's times. Missing positions may be represented
    by mass=0; they are not written. The current projection is unweighted;
    history stores mass*projected. The returned values include zero-mass rows,
    which the caller must exclude from state updates and candidate selection.

    Kernel ``[..., taps, P]`` and bias ``[..., P]`` broadcast over the leading
    batch axes. For node/sample inputs [N,B,L,P], node parameters therefore use
    [N,1,taps,P] and [N,1,P]. Kernel index zero is the current tap.

    Searchsorted performs exact time lookup with O(L*taps) query storage,
    without a pairwise sequence matrix or advancing by reception count.
    The final cache retains the full raw mapping after the last positive write,
    including expired entries when a row received no positive mass this window.
    Its padded width is bounded by (taps-1)*stride+1.
    """
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("convolution stride must be a positive logical-time interval")
    if projected.ndim < 2 or mass.shape != projected.shape[:-1] or times.shape != mass.shape:
        raise ValueError("expected projected [..., L, P] and mass/times [..., L]")
    if times.dtype != torch.int64 or times.device != projected.device:
        raise ValueError("convolution times must be int64 on the input device")
    if kernel.ndim < 2 or kernel.shape[-1] != projected.shape[-1] or kernel.shape[-2] < 1:
        raise ValueError("kernel must have shape [..., taps, P]")
    if bias.ndim < 1 or bias.shape[-1] != projected.shape[-1]:
        raise ValueError("bias must have shape [..., P]")
    batch = projected.shape[:-2]
    try:
        if torch.broadcast_shapes(batch, kernel.shape[:-2], bias.shape[:-1]) != batch:
            raise ValueError("parameter leading dimensions do not broadcast to the input")
    except RuntimeError as exc:
        raise ValueError("parameter leading dimensions do not broadcast to the input") from exc
    _floating_like(projected, mass, kernel, bias)
    if state is None:
        state = ConvState(times[..., :0], projected[..., :0, :])
    if (state.times.shape[:-1] != batch or state.values.shape[:-1] != state.times.shape
            or state.values.shape[-1] != projected.shape[-1]):
        raise ValueError("convolution cache shape disagrees with the input")
    if state.times.dtype != torch.int64 or state.times.device != times.device:
        raise ValueError("cached convolution times must be int64 on the input device")
    _floating_like(projected, state.values)
    if times.shape[-1] == 0:
        return projected, state

    taps, channels = kernel.shape[-2:]
    history_times = torch.cat((state.times, times), dim=-1)
    # The declared boundary VJP follows the no-write branch at mass=0. This
    # also preserves gradients when that slot lies on a continuation boundary.
    positive = mass > 0 if positive is None else positive
    writes = mass.unsqueeze(-1) * projected if weighted_writes is None else weighted_writes
    writes = writes.masked_fill(~positive.unsqueeze(-1), 0)
    history = torch.cat((state.values, writes), dim=-2)
    result = projected * kernel[..., 0, :].unsqueeze(-2) + bias.unsqueeze(-2)
    if taps > 1:
        lags = stride * torch.arange(1, taps, device=times.device, dtype=torch.int64)
        wanted = (times.unsqueeze(-1) - lags).flatten(-2).contiguous()
        index = torch.searchsorted(history_times, wanted).clamp_max(history_times.shape[-1] - 1)
        found = history_times.gather(-1, index)
        valid = (found == wanted) & (wanted >= 0)
        saved = history.gather(-2, index.unsqueeze(-1).expand(*index.shape, channels))
        saved = saved * valid.unsqueeze(-1)
        saved = saved.reshape(*batch, times.shape[-1], taps - 1, channels)
        result = result + (saved * kernel[..., 1:, :].unsqueeze(-3)).sum(-2)

    last = times.masked_fill(~positive, -1).amax(-1)
    horizon = (taps - 1) * stride
    valid = torch.cat((state.times >= 0, positive), dim=-1)
    keep = valid & ((last < 0).unsqueeze(-1) | (history_times >= (last - horizon).unsqueeze(-1)))
    slots = min(horizon + 1, history_times.shape[-1])
    retained_times, retained_index = history_times.masked_fill(~keep, -1).topk(slots, dim=-1)
    retained_times, retained_index = retained_times.flip(-1), retained_index.flip(-1)
    retained = history.gather(-2, retained_index.unsqueeze(-1).expand(*retained_index.shape, channels))
    retained = retained.masked_fill((retained_times < 0).unsqueeze(-1), 0)
    return result, ConvState(retained_times, retained)
