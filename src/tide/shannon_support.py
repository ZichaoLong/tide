"""Opt-in storage and face refinement for extreme Shannon rows.

The ordinary solve stays on its device. Only rows with a subnormal/erased
response or an unresolved bounded solve are packed to the host. The face
test evaluates the monotone total at each candidate bound. This distinguishes
an interior weight rounded to its cap from a genuinely saturated weight.
The backward consumes binary parts of the *unrounded* free responses.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import math
import struct

import torch


FLOOR = "positive-floor-v1"


@dataclass
class SupportCounts:
    refined_rows: int = 0
    repaired_rows: int = 0
    repaired_coordinates: int = 0


_counts = ContextVar("tide_shannon_support_counts", default=None)


@contextmanager
def observe_shannon_support():
    """Count actual solves in this context, including checkpoint recomputation."""
    counts = SupportCounts()
    token = _counts.set(counts)
    try:
        yield counts
    finally:
        _counts.reset(token)


def _compare_total(positive, target):
    # Consume the budget before summing tiny terms. If the large terms exhaust
    # it exactly, *any* further positive contribution changes the selected face.
    remaining = target
    for value in sorted(positive, reverse=True):
        remaining -= value
        if remaining < 0:
            return 1
    return -1 if remaining > 0 else 0


def _positive_exp(offset, context):
    value = offset.exp()
    # An exponent outside even Decimal's range cannot contribute a public
    # FP32/FP64 product, but its positive sign can decide a breakpoint tie.
    return value if value else context.next_plus(Decimal(0))


def _parts(value):
    if not value:
        return 0., 0
    ordinary = float(value)
    if ordinary >= 2.**-1022:
        return math.frexp(ordinary)
    # Even the largest finite FP64 root and division by its smallest
    # subnormal cannot rescue a score term this small. Keep a positive sign.
    if value.adjusted() < -4800:
        return .5, -16000
    exponent = math.floor(value.adjusted() * math.log2(10))
    fraction, correction = math.frexp(float(value / (Decimal(2) ** exponent)))
    return fraction, exponent + correction


def _solve_row(row, width, dtype, delta, tiny, eps, cap_offset, context):
    data = list(map(Decimal.from_float, row))
    scores, lower, upper = data[:width], data[width:2*width], data[2*width:3*width]
    low, high = data[-2:]
    zero, one = Decimal(0), Decimal(1)

    def at_bound(origin, amount):
        values = []
        for score, lo, hi in zip(scores, lower, upper):
            offset = score - origin
            if lo == hi or offset > cap_offset:
                values.append(hi)
            else:
                raw = amount * _positive_exp(offset, context)
                values.append(max(lo, min(hi, raw if raw else context.next_plus(zero))))
        return values

    values = at_bound(one, one)
    low_active = _compare_total(values, low) < 0
    high_active = _compare_total(values, high) > 0
    target = low if low_active else high
    constrained = low_active or high_active
    at_lower, at_upper = target <= sum(lower), target >= sum(upper)
    if constrained and at_lower:
        values = lower.copy()
    elif constrained and at_upper:
        values = upper.copy()
    elif constrained:
        values = []
        for score, lo, hi in zip(scores, lower, upper):
            if lo == hi or (lo and _compare_total(at_bound(score, lo), target) >= 0):
                values.append(lo)
            elif _compare_total(at_bound(score, hi), target) <= 0:
                values.append(hi)
            else:
                values.append(None)
        free = [i for i, value in enumerate(values) if value is None]
        lower_face = [value == lo and lo != hi for value, lo, hi in zip(values, lower, upper)]
        upper_face = [value == hi and lo != hi for value, lo, hi in zip(values, lower, upper)]
        if free:
            remaining = target - sum(v for v in values if v is not None)
            if remaining <= 0:
                raise ArithmeticError("Shannon positive-support face exhausted its remaining mass")
            peak = max(scores[i] for i in free)
            weights = [_positive_exp(scores[i] - peak, context) for i in free]
            denominator = sum(weights)
            for i, weight in zip(free, weights):
                values[i] = (remaining * weight / denominator) or context.next_plus(zero)
    if not constrained or at_lower or at_upper:
        lower_face = [v == lo and lo != hi for v, lo, hi in zip(values, lower, upper)]
        upper_face = [v == hi and lo != hi for v, lo, hi in zip(values, lower, upper)]

    stored, mantissas, exponents = [], [], []
    repaired = 0
    for i, value in enumerate(values):
        if value < tiny:
            units = (value / delta).to_integral_value(rounding=ROUND_HALF_EVEN)
            rounded = float(units * delta)
        else:
            rounded = float(value)
            if dtype == torch.float32:
                rounded = struct.unpack("f", struct.pack("f", rounded))[0]
        if rounded == 0 and value > 0:
            rounded = float(delta)
            repaired += 1
        if not lower[i] <= Decimal.from_float(rounded) <= upper[i]:
            raise ArithmeticError("Shannon positive-support storage would violate a role bound")
        stored.append(rounded)
        free = lower[i] != upper[i] and not lower_face[i] and not upper_face[i]
        mantissa, exponent = _parts(value if free else zero)
        mantissas.append(mantissa)
        exponents.append(exponent)
    total = sum(map(Decimal.from_float, stored))
    added = repaired * delta
    budget = 64 * width * eps * max(total - added, low)
    if added > budget or total > high + budget or total < low - budget:
        raise ArithmeticError("Shannon positive-support mass perturbation exceeds relative budget")
    if constrained and abs(total - added - target) > 32 * eps * max(abs(target), tiny):
        raise ArithmeticError("Shannon positive-support solve missed its total constraint")
    return (stored, mantissas, exponents, lower_face, upper_face, low_active, high_active, repaired)


def refine_rows(scores, lower, upper, low_total, high_total, rows):
    """One packed host transfer in each direction, independent candidate-last rows."""
    width = scores.shape[-1]
    shape = scores.shape[:-1] + (1,)
    columns = (scores, lower, upper, low_total.expand(shape), high_total.expand(shape))
    packed = torch.cat(columns, -1).reshape(-1, 3*width + 2)[rows]
    source = packed.detach().cpu().tolist()
    info = torch.finfo(scores.dtype)
    delta, tiny, eps = map(Decimal.from_float, (info.tiny * info.eps, info.tiny, info.eps))
    payload, repairs = [], []
    # Exact sums of storage bounds across the public exponent span plus
    # guard digits. Do not inherit a caller's decimal rounding/trap settings.
    with localcontext(Context(prec=192 if scores.dtype == torch.float32 else 1200)) as context:
        cap_offset = -delta.ln() + 2  # all caps <= 1, including for the smallest anchor
        for row in source:
            c, m, e, lf, uf, la, ua, repaired = _solve_row(row, width, scores.dtype, delta, tiny, eps, cap_offset, context)
            payload.append(c + m + e + lf + uf + [la, ua])
            repairs.append(repaired)
    counts = _counts.get()
    if counts is not None:
        counts.refined_rows += len(repairs)
        counts.repaired_rows += sum(n > 0 for n in repairs)
        counts.repaired_coordinates += sum(repairs)
    device_values = scores.new_tensor(payload)
    c, m, e, lf, uf, la, ua = device_values.split((width, width, width, width, width, 1, 1), -1)
    if bool(((c == 0) & (m > 0)).any()):
        raise ArithmeticError("Shannon positive-support storage flushed a positive sender")
    return c, m, e.to(torch.int32), lf.bool(), uf.bool(), la.bool(), ua.bool()
