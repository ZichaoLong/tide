"""Device-local restoration of an FP32/FP64 binary scale.

Some device ldexp implementations materialize 2**exponent before multiplying.
That can overflow for fraction*2**128 even when the FP32 result is finite, or
underflow the power before a non-normalized fraction restores its magnitude.
Normalize the fraction first, keep the first product normal, and let only the
last product round into the subnormal range. No dtype widening or CPU fallback
is used; callers retain their own checks for genuinely unrepresentable results.
"""

import torch


class GradientRangeError(ArithmeticError):
    """A first-order output adjoint exceeded the input tensor's dtype range."""


def two_sum(a, b):
    """Head and rounding tail of a bounded floating-point addition."""
    total = a + b
    virtual = total - a
    return total, (a - (total - virtual)) + (b - virtual)


def compensated_sum(values, *, track_roundoff=False):
    """Pairwise two-term reduction of nonnegative storage values."""
    high, low = values, torch.zeros_like(values)
    lost = torch.zeros_like(values, dtype=torch.bool) if track_roundoff else None
    while high.shape[-1] > 1:
        if high.shape[-1] % 2:
            high = torch.cat((high, torch.zeros_like(high[..., :1])), -1)
            low = torch.cat((low, torch.zeros_like(low[..., :1])), -1)
            if track_roundoff:
                lost = torch.cat((lost, torch.zeros_like(lost[..., :1])), -1)
        total, error = two_sum(high[..., ::2], high[..., 1::2])
        if track_roundoff:
            remainder, e1 = two_sum(error, low[..., ::2])
            remainder, e2 = two_sum(remainder, low[..., 1::2])
            lost = lost[..., ::2] | lost[..., 1::2] | (e1 != 0) | (e2 != 0)
        else:
            remainder = error + low[..., ::2] + low[..., 1::2]
        high, low = two_sum(total, remainder)
    return (high, low, lost) if track_roundoff else (high, low)


def binary_parts(value):
    """Exact frexp parts, with an explicit NPU FP32 device implementation.

    This TorchNPU baseline has no frexp kernel. Decode IEEE binary32 instead;
    multiply subnormals by an exact power before decoding their exponent.
    CPU FP64 remains a diagnostic and uses torch.frexp. No tensor leaves its
    device. Zero (including -0), Inf and NaN keep their original value.
    """
    if value.device.type != "npu":
        return torch.frexp(value)
    if value.dtype != torch.float32:
        raise TypeError("NPU binary parts require FP32 accumulation")
    import torch_npu

    # A dtype view requires a base-format, contiguous storage representation.
    value = torch_npu.npu_format_cast(value, 2).contiguous()
    bits = value.view(torch.int32)
    magnitude = bits & 0x7fffffff
    subnormal = (magnitude > 0) & (magnitude < 0x00800000)
    scaled = torch.where(subnormal, value, 0) * 16777216.0
    working = torch.where(subnormal, scaled, value).contiguous().view(torch.int32)
    field = (working & 0x7fffffff) >> 23
    exponent = field - 126 - subnormal.to(torch.int32) * 24
    fraction_bits = (working & -2139095041) | 0x3f000000
    fraction = fraction_bits.contiguous().view(torch.float32)
    special = (magnitude == 0) | (field == 255)
    return torch.where(special, value, fraction), torch.where(special, 0, exponent)


def restore_binary_scale(fraction, exponent):
    """Evaluate ``fraction * 2**exponent`` without an extreme standalone power.

    Nonzero finite fractions are normalized with frexp, so their absolute value
    is in [0.5, 1). The first exponent is bounded to [-125, 127] for FP32 and
    [-1021, 1023] for FP64: the first product is therefore normal and exact.
    A representable final result needs only a modest remaining exponent. True
    overflow/underflow remains Inf/zero, and zero fractions keep their sign even
    when the supplied exponent is very large. Floating rounding occurs only in
    the final product, avoiding double rounding at the subnormal boundary.
    """
    if fraction.dtype == torch.float32:
        lower, upper, smallest = -125, 127, -149
    elif fraction.dtype == torch.float64:
        lower, upper, smallest = -1021, 1023, -1074
    else:
        raise ValueError("binary scale restoration requires FP32 or FP64 arithmetic")
    fraction, exponent = torch.broadcast_tensors(fraction, exponent)
    normalized, shift = binary_parts(fraction)
    combined = torch.where(normalized != 0, exponent + shift, 0)
    overflow, underflow = combined > upper + 1, combined < smallest
    # True range failures need no oversized vendor power (which can itself
    # request an unsupported intermediate scalar dtype). Preserve Inf/zero
    # for the caller's explicit range policy, including their sign.
    safe = torch.where(overflow | underflow, 0, combined)
    first = safe.clamp(min=lower, max=upper)
    partial = torch.ldexp(normalized, first)
    restored = torch.ldexp(partial, safe - first)
    if fraction.device.type == "npu":
        negative = torch.signbit(normalized)
        infinity = torch.full_like(restored, torch.inf)
        signed_inf = torch.where(negative, -infinity, infinity)
        signed_zero = torch.where(negative, -torch.zeros_like(restored), torch.zeros_like(restored))
    else:
        signed_inf = torch.copysign(torch.full_like(restored, torch.inf), normalized)
        signed_zero = torch.copysign(torch.zeros_like(restored), normalized)
    restored = torch.where(overflow, signed_inf, restored)
    return torch.where(underflow, signed_zero, restored)
