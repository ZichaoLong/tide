"""Device-local restoration of an FP32/FP64 binary scale.

Some device ldexp implementations materialize 2**exponent before multiplying.
That can overflow for fraction*2**128 even when the FP32 result is finite, or
underflow the power before a non-normalized fraction restores its magnitude.
Normalize the fraction first, keep the first product normal, and let only the
last product round into the subnormal range. No dtype widening or CPU fallback
is used; callers retain their own checks for genuinely unrepresentable results.
"""

import torch


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
    normalized, shift = torch.frexp(fraction)
    combined = torch.where(normalized != 0, exponent + shift, 0)
    overflow, underflow = combined > upper + 1, combined < smallest
    # True range failures need no oversized vendor power (which can itself
    # request an unsupported intermediate scalar dtype). Preserve Inf/zero
    # for the caller's explicit range policy, including their sign.
    safe = torch.where(overflow | underflow, 0, combined)
    first = safe.clamp(min=lower, max=upper)
    partial = torch.ldexp(normalized, first)
    restored = torch.ldexp(partial, safe - first)
    restored = torch.where(overflow, torch.copysign(torch.full_like(restored, torch.inf), normalized), restored)
    return torch.where(underflow, torch.copysign(torch.zeros_like(restored), normalized), restored)
