"""Positive-support storage for exceptional total-scaled attention rows."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import math

import torch

from .shannon_support import FLOOR


@dataclass
class PoolSupportCounts:
    rows: int = 0
    repaired_rows: int = 0
    repaired_coordinates: int = 0


_observer = ContextVar("tide_pool_support_counts", default=None)


@contextmanager
def observe_pool_support(counts=None):
    counts = PoolSupportCounts() if counts is None else counts
    token = _observer.set(counts)
    try:
        yield counts
    finally:
        _observer.reset(token)


def _repair_row(row, width, delta, precision):
    # All public outputs are in [0,1]. Decimal also distinguishes positive
    # weights below the host float's subnormal range from structural zeros.
    scores, priors, stored = (list(map(Decimal.from_float, row[i:i+width]))
                              for i in range(0, 3*width, width))
    target = Decimal.from_float(row[-1])
    positive = [i for i, prior in enumerate(priors) if prior > 0]
    center = max(scores[i] for i in positive)
    weights = [priors[i] * (scores[i] - center).exp() for i in positive]
    denominator = sum(weights)
    if target <= 0 or denominator <= 0:
        raise ArithmeticError("pool lost a positive normalization or total")
    for i, weight in zip(positive, weights):
        if stored[i] == 0:
            if 2 * target * weight > delta * denominator:
                raise ArithmeticError("pool erased a representable positive sender")
            stored[i] = delta

    def spacing(value):
        exponent = math.frexp(float(value))[1]
        # The host double can round up to the next binade at its boundary.
        if value < Decimal(2) ** (exponent - 1):
            exponent -= 1
        return max(delta, Decimal(2) ** (exponent - precision))

    # Change a donor only when the independent stores miss the row budget
    # by more than half an output ULP. In the subnormal range this requires
    # an exact sum of integer storage units. Every positive prior funds one.
    remaining = target - sum(stored)
    tolerance = spacing(target) / 2
    if abs(remaining) > tolerance:
        for i in sorted(positive, key=stored.__getitem__, reverse=True):
            desired = max(delta, stored[i] + remaining)
            unit = spacing(desired)
            adjusted = (desired / unit).to_integral_value(rounding=ROUND_HALF_EVEN) * unit
            remaining -= adjusted - stored[i]
            stored[i] = adjusted
            if abs(remaining) <= tolerance:
                break
        else:
            raise ArithmeticError("pool positive-support storage exceeded the row budget")
    return [float(x) for x in stored]


def store_pool(scores, mu, amount, result, policy):
    erased = (mu > 0) & (result == 0)
    counts = _observer.get() if policy == FLOOR else None
    if counts is not None:
        counts.rows += scores.numel() // scores.shape[-1]
    if not bool(erased.any()):
        return result
    if policy != FLOOR:
        raise ArithmeticError("pool underflow would erase a positive sender")
    width = scores.shape[-1]
    indices = erased.reshape(-1, width).any(-1).nonzero().flatten()
    packed = torch.cat((scores, mu, result, amount), -1).reshape(-1, 3*width+1).index_select(0, indices)
    delta = Decimal.from_float(torch.finfo(scores.dtype).tiny * torch.finfo(scores.dtype).eps)
    precision = 24 if scores.dtype == torch.float32 else 53
    with localcontext(Context(prec=192 if precision == 24 else 1200)):
        repaired = [_repair_row(row, width, delta, precision) for row in packed.detach().cpu().tolist()]
    replacement = scores.new_tensor(repaired)
    if bool(((packed[:, width:2*width] > 0) & (replacement == 0)).any()):
        raise ArithmeticError("pool positive-support storage flushed a positive sender")
    if counts is not None:
        counts.repaired_rows += len(repaired)
        counts.repaired_coordinates += int(erased.sum())
    return result.reshape(-1, width).index_copy(0, indices, replacement).reshape_as(result)
