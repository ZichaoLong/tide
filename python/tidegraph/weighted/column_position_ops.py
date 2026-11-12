"""KV §2–4: bounded common-origin RoPE, original int64 visibility/distance.

Rot(q-o)^T Rot(k-o) equals Rot(k-q) over the reals. The finite arithmetic
domain bounds integer conversion and separated-phase error; it is not a proof
of bitwise or route equivalence. Outside that domain use literal relative keys.
"""
from dataclasses import dataclass
import math
import torch
from ..column_policy import POSITION_POLICY
from .column_query_ops import contract
from .tensors import rotate_relative
from .numeric_time import time_values
from .packing_observation import bucket


@dataclass(frozen=True)
class PositionQuery:
    query: torch.Tensor
    key: torch.Tensor
    age: torch.Tensor
    query_layout: object
    bank_layout: object
    frequencies: tuple
    scale: float
    slopes: tuple = ()


def rebase_positions(query_layout, bank_layout, frequencies, scale, dtype):
    """Host metadata only. Return int64 offsets or an explicit finite-domain reason.

    For evaluation epsilon e, require 8*e*max_abs_phase <= sqrt(e). This is a
    conservative phase-roundoff proxy for conversion/division/multiplication on
    both sides, not an output tolerance. Exact-integer spans are also required.
    Half keeps its established relative-key rounding until separately qualified.
    """
    if dtype not in {torch.float32, torch.float64}:
        return None, "half"
    epsilon = torch.finfo(dtype).eps
    integer_limit = 2**(24 if dtype == torch.float32 else 53)
    phase_limit = 1/(8*math.sqrt(epsilon))
    max_frequency = max((abs(f) for head in frequencies for f in head), default=0.)
    qspans, kspans = query_layout.spans, bank_layout.spans
    if len(qspans) != len(kspans):
        raise ValueError("position query/bank sample domains differ")
    qs, ks = query_layout.times.tolist(), bank_layout.times.tolist()
    width_q = max((z-a for a, z in qspans), default=1) or 1
    width_k = max((z-a for a, z in kspans), default=1) or 1
    origins = []
    for (a, z), (ka, kz) in zip(qspans, kspans):
        first = ([qs[a]] if a < z else [])+([ks[ka]] if ka < kz else [])
        last = ([qs[z-1]] if a < z else [])+([ks[kz-1]] if ka < kz else [])
        origin = min(first, default=0)
        span = max(last, default=origin)-origin
        if span > integer_limit:
            return None, "integer_span"
        phase = span/scale*max_frequency
        if not math.isfinite(phase) or phase > phase_limit:
            return None, "phase_span"
        origins.append(origin)
    qt = torch.zeros((len(qspans), width_q), dtype=torch.int64)
    kt = torch.zeros((len(kspans), width_k), dtype=torch.int64)
    for b, ((a, z), (ka, kz), origin) in enumerate(zip(qspans, kspans, origins)):
        qt[b, :z-a] = query_layout.times[a:z]-origin
        kt[b, :kz-ka] = bank_layout.times[ka:kz]-origin
    return (qt, kt), "rebased"


def _relative(request):
    q, k = request.query, request.key
    displacement = -time_values(request.age, q)/request.scale
    logits = []
    for h, frequencies in enumerate(request.frequencies):
        key = rotate_relative(k[:, h, None].expand(-1, q.shape[2], -1, -1), displacement, frequencies)
        head = q[:, h, :, None]
        if q.dtype == torch.float16:
            head, key = head.float(), key.float()
        logits.append((head*key).sum(-1)/math.sqrt(q.shape[-1]))
    return torch.stack(logits, 1)


class _Rotate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, offsets, frequencies, scale):
        ctx.set_materialize_grads(False)
        ctx.frequencies, ctx.scale = frequencies, scale
        # The linear transpose need not read value; keep its version guard.
        ctx.save_for_backward(value, offsets)
        coordinates = time_values(offsets, value)/scale
        return torch.stack([rotate_relative(value[:, h], coordinates, f) for h, f in enumerate(frequencies)], 1)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("column rotary positions support first-order VJP only")
        value, offsets = ctx.saved_tensors
        if bar is None:
            return None, None, None, None
        coordinates = -time_values(offsets, value)/ctx.scale
        gradient = torch.stack([rotate_relative(bar[:, h], coordinates, f) for h, f in enumerate(ctx.frequencies)], 1)
        return gradient, None, None, None


def position_logits_many(requests):
    """Compatible Q/K shapes; actual node roots stay separate through the BMM."""
    result, selected, operands = [None]*len(requests), [], []
    for i, r in enumerate(requests):
        q, k = r.query, r.key
        if (q.ndim != 4 or k.ndim != 4 or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]
                or (q.dtype, q.device) != (k.dtype, k.device) or len(r.frequencies) != q.shape[1]
                or r.age.dtype != torch.int64 or r.age.shape != (q.shape[0], q.shape[2], k.shape[2])):
            raise ValueError("invalid positioned column query metadata")
        if any(r.frequencies):
            coordinates, reason = rebase_positions(r.query_layout, r.bank_layout, r.frequencies, r.scale, q.dtype)
            if coordinates is None:
                result[i] = _relative(r)
                bucket("kv_rope_relative_"+reason, q.shape[2], 1)
                continue
            qt, kt = (t.to(q.device) for t in coordinates)
            q = _Rotate.apply(q, qt, r.frequencies, r.scale)
            k = _Rotate.apply(k, kt, r.frequencies, r.scale)
        else:
            reason = "content_distance"
            if q.dtype == torch.float16:
                q, k = q.float(), k.float()
        bucket("kv_rope_"+reason, q.shape[2], 1)
        selected.append(i)
        operands.append((q.flatten(0, 1), k.flatten(0, 1).transpose(-1, -2)))
    for i, value in zip(selected, contract("matmul", operands)):
        q = requests[i].query
        result[i] = value.reshape(*q.shape[:3], requests[i].key.shape[2])/math.sqrt(q.shape[-1])
    for i, r in enumerate(requests):
        if any(r.slopes):
            displacement = -time_values(r.age, r.query)/r.scale
            result[i] = result[i]+result[i].new_tensor(r.slopes)[None, :, None, None]*displacement[:, None]
    return result


def position_logits(request):
    return position_logits_many([request])[0]
