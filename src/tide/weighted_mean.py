"""Reception's normalized mean and centered first-order mass derivative.

Mass is nonnegative on entry. Scaling it before normalization preserves content
even when every mass is subnormal. Products are contracted as binary mantissas
and exponents, so an unrepresentable probability or intermediate product need
not erase a representable result. This does not widen the public dtype or floor
any quantity. A nonfinite public value/adjoint raises instead of being clipped.

The entirely zero row keeps Reception's existing denominator-one convention:
value/content VJP are zero and the mass VJP is dot(value, upstream). This is a
declared branch derivative, not a directional derivative of the normalized map
at the origin. Reception supplies structural membership separately from mass;
the raw negative-mass clamp remains outside this helper.
"""

import torch
from torch.autograd.function import once_differentiable

from .scaled_arithmetic import binary_parts, restore_binary_scale


def _finite(label, *values):
    # Deliberately explicit, including on accelerators. Device qualification
    # and the synchronization cost of these guards are separate observations.
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ArithmeticError(f"reception weighted mean {label} exceeds the requested dtype range")


def _parts(value):
    return binary_parts(value)


def _restore(mantissa, exponent):
    return restore_binary_scale(mantissa, exponent)


def _align(mantissa, exponent, peak):
    # A zero term must not be multiplied by an overflowing power of two.
    shift = torch.where(mantissa != 0, exponent - peak, 0)
    return restore_binary_scale(mantissa, shift)


def _sum_parts(mantissa, exponent, dim):
    """Contract signed terms before restoring their common exponent."""
    nonzero = mantissa != 0
    # FP64 operand products/quotients use exponents within +/- 4096 here.
    # This sentinel only selects a maximum; it never changes a numeric term.
    peak = torch.where(nonzero, exponent, -16384).amax(dim, keepdim=True)
    peak = torch.where(nonzero.any(dim, keepdim=True), peak, 0)
    subtotal = _align(mantissa, exponent, peak).sum(dim)
    fraction, shift = _parts(subtotal)
    return fraction, shift + peak.squeeze(dim)


def _difference_parts(value, mean_m, mean_e):
    """Center before contraction without overflowing opposite finite values."""
    vm, ve = _parts(value)
    um, ue = mean_m, mean_e
    peak = torch.maximum(torch.where(vm != 0, ve, -16384),
                         torch.where(um != 0, ue, -16384))
    peak = torch.where((vm != 0) | (um != 0), peak, 0)
    return _align(vm, ve, peak) - _align(um, ue, peak), peak


def _reduce_parts(mantissa, exponent, shape):
    """Sum broadcast users before converting the public source adjoint."""
    while mantissa.ndim > len(shape):
        mantissa, exponent = _sum_parts(mantissa, exponent, 0)
    for dim, size in enumerate(shape):
        if size == 1 and mantissa.shape[dim] != 1:
            mantissa, exponent = _sum_parts(mantissa, exponent, dim)
            mantissa, exponent = mantissa.unsqueeze(dim), exponent.unsqueeze(dim)
    return _restore(mantissa, exponent)


class _WeightedMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mass, values, mask):
        ctx.set_materialize_grads(False)
        ctx.mass_shape, ctx.value_shape = mass.shape, values.shape
        dtype = torch.promote_types(mass.dtype, values.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        shape = torch.broadcast_shapes(mass.shape, values.shape[:-1])
        m = mass.to(dtype).expand(shape)
        v = values.to(dtype).expand(*shape, values.shape[-1])
        if mask is not None:
            m = m.masked_fill(~mask.expand(shape), 0)
        total = m.sum(-1)
        if shape[-1] == 0:
            mean = v.sum(-2)
            mean_m, mean_e = _parts(mean)
        else:
            peak = m.amax(-1, keepdim=True)
            safe_peak = torch.where(peak > 0, peak, 1)
            scaled_total = (m / safe_peak).sum(-1, keepdim=True)
            normalizer = torch.where(scaled_total > 0, scaled_total, 1)
            mm, me = _parts(m)
            pm, pe = _parts(safe_peak)
            sm, se = _parts(normalizer)
            vm, ve = _parts(v)
            # p = (m / peak) / sum(m / peak). Keep its exponent until p*v
            # has been contracted; a tiny p may have a representable p*v.
            probabilities = (mm / pm) / sm
            exponent = me - pe - se
            mean_m, mean_e = _sum_parts(probabilities[..., None] * vm,
                                       exponent[..., None] + ve, -2)
            mean = _restore(mean_m, mean_e)
        _finite("forward", total, mean)
        # Save the original inputs too: promotion/expansion must not hide a
        # caller's in-place source mutation between forward and backward.
        # Keep the mean's exponent for the composite VJP too. A mean below
        # the output's range can still produce a representable mean*upstream.
        ctx.save_for_backward(mass, values, total, mean_m, mean_e, mask)
        return mean

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return None, None, None
        original_mass, original_values, total, mean_m, mean_e, mask = ctx.saved_tensors
        dtype = mean_m.dtype
        shape = torch.broadcast_shapes(ctx.mass_shape, ctx.value_shape[:-1])
        mass = original_mass.to(dtype).expand(shape)
        values = original_values.to(dtype).expand(*shape, ctx.value_shape[-1])
        if mask is not None:
            mask = mask.expand(shape)
            mass = mass.masked_fill(~mask, 0)
        grad = grad.to(dtype)
        need_mass, need_values = ctx.needs_input_grad[:2]
        dm = dv = None
        if shape[-1] == 0:
            dm = torch.zeros_like(original_mass) if need_mass else None
            dv = torch.zeros_like(original_values) if need_values else None
        else:
            denominator = torch.where(total > 0, total, 1)
            den_m, den_e = _parts(denominator[..., None, None])
            root_m, root_e = _parts(grad[..., None, :])
            if need_mass:
                delta_m, delta_e = _difference_parts(values, mean_m[..., None, :], mean_e[..., None, :])
                dm_m, dm_e = _sum_parts((delta_m * root_m) / den_m,
                                       delta_e + root_e - den_e, -1)
                # A singleton positive mean is independent of its own mass,
                # even with other real zero-mass edges. Those other roots
                # still have their right derivative. All-zero is different.
                singleton = (mass > 0).sum(-1, keepdim=True) == 1
                dm_m = torch.where(singleton & (mass > 0), 0, dm_m)
                if mask is not None:
                    # These roots do not exist. Remove them before exponent
                    # restoration/range checks, not after an overflowing VJP.
                    dm_m = dm_m.masked_fill(~mask, 0)
                dm = _reduce_parts(dm_m, dm_e, ctx.mass_shape).to(original_mass.dtype)
            if need_values:
                mass_m, mass_e = _parts(mass[..., None])
                dv_m, dv_e = (mass_m / den_m) * root_m, mass_e - den_e + root_e
                dv = _reduce_parts(dv_m, dv_e, ctx.value_shape).to(original_values.dtype)
        _finite("VJP", *(value for value in (dm, dv) if value is not None))
        return dm, dv, None


def stable_weighted_mean(mass, values, mask=None):
    """Mean over the final mass axis / penultimate value axis.

    Leading dimensions broadcast normally. FP16/BF16 inputs compute and return
    at least FP32; source adjoints return to each source dtype. Only a first-order
    VJP is declared. A boolean mask denotes structurally absent edges, not zero
    numerical masses. No backend silently copies arithmetic onto the CPU.
    """
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError("weighted mean structural mask must be boolean")
    with torch.autocast(values.device.type, enabled=False):
        return _WeightedMean.apply(mass, values, mask)
