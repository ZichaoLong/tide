"""Normalized weighted attention, with zero rows and weight right derivatives.

The optional mass_scaled entry fuses the Full/Read amplitude min(1,sum(weights)).
All paths implement first-order VJPs; a zero-weight key in a nonempty row retains
its right derivative. At an entirely zero row we choose the zero VJP.
"""

from __future__ import annotations

import math
from functools import wraps

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .scaled_arithmetic import GradientRangeError, binary_parts, restore_binary_scale


def _without_autocast(function):
    @wraps(function)
    def call(ctx, first, *args):
        with torch.autocast(first.device.type, enabled=False):
            return function(ctx, first, *args)
    return call


def _probability_values(logits: Tensor, weights: Tensor, mask: Tensor):
    positive = mask & (weights > 0)
    log_weights = torch.where(weights > 0, weights, 1).log()
    weighted = torch.where(positive, logits + log_weights, -torch.inf)
    nonempty = positive.any(-1, keepdim=True)
    safe = torch.where(nonempty, weighted, 0)
    log_z = torch.logsumexp(safe, -1, keepdim=True)
    p = torch.where(nonempty, (safe - log_z).exp(), 0)
    return p, log_z, nonempty


class _WeightedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, weights, mask):
        p, log_z, nonempty = _probability_values(logits, weights, mask)
        ctx.save_for_backward(logits, p, log_z, mask, nonempty, weights)
        return p

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        logits, p, log_z, mask, nonempty, weights = ctx.saved_tensors
        centered = grad - (p * grad).sum(-1, keepdim=True)
        log_factor = torch.where(mask & nonempty, logits - log_z, -torch.inf)
        log_p = torch.where(mask & nonempty & (weights > 0),
                            log_factor + torch.where(weights > 0, weights, 1).log(), -torch.inf)
        # A normalized derivative can be finite even when exp(log_factor)
        # alone overflows. Retain the scale through the cotangent product.
        return (_signed_exp_product(log_p, centered),
                _signed_exp_product(log_factor, centered), None)


def _weighted_softmax(logits: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
    logits, weights, mask = torch.broadcast_tensors(logits, weights, mask)
    if logits.shape[-1] == 0:
        return logits * 0 + weights * 0
    return _WeightedSoftmax.apply(logits, weights, mask)


def _auxiliary(value: Tensor, shape: tuple, key_prior: bool) -> Tensor:
    """Canonicalize compact key priors or conventional broadcast masks."""
    if value.ndim == 0:
        return value.reshape((1,) * len(shape))
    if key_prior and value.ndim <= len(shape) - 2:
        candidate = value.reshape(*value.shape[:-1], 1, 1, value.shape[-1])
        try:
            torch.broadcast_shapes(candidate.shape, shape)
            value = candidate
        except RuntimeError:
            pass
    try:
        result = torch.broadcast_shapes(value.shape, shape)
        if result != shape:
            raise ValueError("auxiliary tensor expands the attention shape")
    except RuntimeError as exc:
        if value.ndim == len(shape) - 1:
            value = value.unsqueeze(-3)
        elif value.ndim <= len(shape) - 2:
            value = value.reshape(*value.shape[:-1], 1, 1, value.shape[-1])
        else:
            raise ValueError("attention auxiliary shape is not broadcastable") from exc
        if torch.broadcast_shapes(value.shape, shape) != shape:
            raise ValueError("attention auxiliary shape is not broadcastable") from exc
    return value.reshape((1,) * (len(shape) - value.ndim) + value.shape)


def _native_attention(q, k, v, weights, mask, backend, *, unit_prior=False, no_mask=False):
    """Keep native fused autograd, including trainable positive key priors."""
    # BF16 needs the same forward arithmetic in inference and grad-enabled
    # calls. An explicit all-one prior therefore retains its prior path, and
    # NPU key priors always use the same rank-one representation below.
    stable_bf16_prior = q.dtype == torch.bfloat16 and not unit_prior
    unit = unit_prior or (not stable_bf16_prior and not weights.requires_grad and bool((weights == 1).all()))
    visible = mask if unit else mask & (weights > 0)
    log_weights = None if unit else torch.where(weights > 0, weights, 1).log()
    if backend == "sdpa":
        bias = (None if no_mask else mask) if unit else torch.where(visible, log_weights, -torch.inf)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    else:
        import torch_npu  # Vendor registration is never required by CPU entry points.

        fusion = getattr(torch_npu, "npu_fusion_attention_v2", None)
        if fusion is None:
            raise RuntimeError("npu_fusion requires npu_fusion_attention_v2")
        batch = math.prod(q.shape[:-3])
        heads, queries, width = q.shape[-3:]
        keys, value_width, scale = k.shape[-2], v.shape[-1], width ** -0.5
        pse = None
        if weights.requires_grad or stable_bf16_prior:
            # PSE-gradient support differs between vendor releases. Encode a
            # positive key prior as one Q/K feature and use native Q/K VJPs.
            prior = log_weights.expand(q.shape[:-2] + (1, keys)).squeeze(-2).to(q.dtype)
            q = torch.cat((q, q.new_full(q.shape[:-1] + (1,), 1 / scale)), -1)
            k = torch.cat((k, prior.unsqueeze(-1)), -1)
            padded = math.ceil(q.shape[-1] / 16) * 16
            q, k, v = (F.pad(x, (0, padded - x.shape[-1])) for x in (q, k, v))
            width = padded
        elif not unit:
            # A compact key prior stays BN1K; no expansion across Q.
            pse = log_weights.to(q.dtype).expand(q.shape[:-2] + (log_weights.shape[-2], keys))
            pse = pse.reshape(batch, heads, *pse.shape[-2:]).contiguous()
        blocked = None
        if not (no_mask and (unit or bool(visible.all()))):
            blocked = (~visible).expand(q.shape[:-3] + (visible.shape[-3], queries, keys))
            blocked = blocked.reshape(batch, *blocked.shape[-3:]).contiguous()
        output = fusion(
            q.reshape(batch, heads, queries, width).contiguous(),
            k.reshape(batch, heads, keys, width).contiguous(),
            v.reshape(batch, heads, keys, width).contiguous(),
            heads, "BNSD", pse=pse, atten_mask=blocked,
            scale=scale, keep_prob=1.0, sparse_mode=1,
            # CANN requires the default PSE mode when no PSE tensor exists.
            # With an actual log prior, select scale(QK^T) + pse explicitly.
            inner_precise=2, pse_type=1 if pse is None else 0,
        )[0].reshape_as(q)[..., :value_width]
    return output if unit and no_mask else torch.where(visible.any(-1, keepdim=True), output, 0)


class _SingleKeyAttention(torch.autograd.Function):
    """Exact normalized singleton/empty rows with connected zero Q/K/prior VJPs.

    Admission excludes visible trainable zero priors: those retain a right
    derivative and must use the existing tiled path. Mass amplitude is applied
    by the caller, so its legitimate prior derivative is not removed here.
    """

    @staticmethod
    @_without_autocast
    def forward(ctx, q, k, v, weights, index, single):
        index = index.expand(q.shape[:-1] + (v.shape[-1],))
        ctx.save_for_backward(q, k, v, weights, index, single)
        return torch.where(single, v.gather(-2, index), 0)

    @staticmethod
    @once_differentiable
    @_without_autocast
    def backward(ctx, grad):
        q, k, v, weights, index, single = ctx.saved_tensors
        dq, dk, dv, dw = (torch.zeros_like(t) if need else None
                          for t, need in zip((q, k, v, weights), ctx.needs_input_grad[:4]))
        if dv is not None:
            dv.scatter_add_(-2, index, torch.where(single, grad, 0))
        return dq, dk, dv, dw, None, None


def _centered_native_attention(q, k, v, weights, mask, backend, *, unit_prior=False, no_mask=False):
    """FP32 value translation keeps a common offset out of native Q/K VJPs.

    For any constant value vector c, Attention(V-c)+c = Attention(V).
    Treat c as a fixed numerical origin: its two mathematical derivatives
    cancel, while native V derivatives already supply the full probability.
    Per-feature range admission prevents the subtraction from introducing an
    overflow for finite extreme values. BF16 multi-key arithmetic is unchanged.
    Empty rows are handled by the support adapter, outside this translation.
    """
    if q.dtype != torch.float32:
        return _native_attention(q, k, v, weights, mask, backend,
                                 unit_prior=unit_prior, no_mask=no_mask)
    detached = v.detach()
    bounded = detached.abs().amax(-2, keepdim=True) <= torch.finfo(v.dtype).max / 2
    origin = torch.where(bounded, detached[..., :1, :], 0)
    return _native_attention(q, k, v-origin, weights, mask, backend,
                             unit_prior=unit_prior, no_mask=no_mask) + origin


def _support_safe_native_attention(q, k, v, weights, mask, backend, *, unit_prior=False, no_mask=False):
    """Retain fusion for multi-key rows, use the identity for exact support <= 1.

    Metadata stays compact when masks/priors broadcast over queries or heads.
    A mixed call sends zero cotangents for analytic rows into fused backward,
    before shared K/V/prior gradients are reduced; no parameter gradient hook
    or small-probability threshold is used. Explicit NPU fusion uses this same
    semantic adapter. CPU/CUDA public dispatch remains unchanged.
    """
    keys = k.shape[-2]
    if unit_prior and no_mask and keys > 1:
        return _centered_native_attention(q, k, v, weights, mask, backend, unit_prior=True, no_mask=True)
    visible = mask & (weights > 0)
    count = visible.sum(-1, keepdim=True)
    if visible.shape[-1] == 1:
        count = count * keys
    special = count <= 1
    if keys > 1 and not bool(special.any()):
        return _centered_native_attention(q, k, v, weights, mask, backend, unit_prior=unit_prior, no_mask=no_mask)
    index = visible.to(torch.int32).argmax(-1, keepdim=True)
    exact = _SingleKeyAttention.apply(q, k, v, weights, index, count == 1)
    if keys == 1 or bool(special.all()):
        return exact
    # Do not evaluate irrelevant large QK products for analytic rows. The
    # final where also cuts their K and prior contributions at the native root.
    native_q = torch.where(special, 0, q)
    fused = _centered_native_attention(native_q, k, v, weights, mask, backend,
                                       unit_prior=unit_prior, no_mask=no_mask)
    return torch.where(special, exact, fused)


def _single_positive_rows(weights, mask, block, key_tile):
    """Exact support metadata, bounded to the same query/key tiles as the VJP.

    A normalized row with one positive prior is independent of Q/K and of
    that prior's magnitude. Real zero priors still have their right derivative.
    Do not infer this identity from small probabilities or gradient magnitudes.
    """
    count = torch.zeros_like(weights[..., block, :1], dtype=torch.int64)
    for offset in range(0, weights.shape[-1], key_tile):
        keys = slice(offset, offset + key_tile)
        count = count + ((weights[..., block, keys] > 0) & mask[..., block, keys]).sum(-1, keepdim=True)
    return count == 1


class _TiledAttention(torch.autograd.Function):
    """Online key/query tiles; backward never retains a Q-by-K matrix."""

    @staticmethod
    @_without_autocast
    def forward(ctx, q, k, v, weights, mask):
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        qw, kw, vw = q.to(dtype), k.to(dtype), v.to(dtype)
        shape = q.shape[:-1] + (k.shape[-2],)
        w, m = weights.expand(shape), mask.expand(shape)
        pieces, normalizers = [], []
        for start in range(0, q.shape[-2], 64):
            block = slice(start, start + 64)
            qb = qw[..., block, :]
            peak = qb.new_full(qb.shape[:-1] + (1,), -torch.inf)
            mass, numerator = torch.zeros_like(peak), qb.new_zeros(qb.shape[:-1] + (v.shape[-1],))
            for offset in range(0, k.shape[-2], 256):
                kb = slice(offset, offset + 256)
                wb, mb = w[..., block, kb].to(dtype), m[..., block, kb]
                logits = qb @ kw[..., kb, :].transpose(-2, -1) * q.shape[-1] ** -0.5
                weighted = torch.where(mb & (wb > 0), logits + torch.where(wb > 0, wb, 1).log(), -torch.inf)
                next_peak = torch.maximum(peak, weighted.amax(-1, keepdim=True))
                safe_peak = torch.where(torch.isfinite(next_peak), next_peak, 0)
                previous = torch.exp(peak - safe_peak)
                terms = torch.exp(weighted - safe_peak)
                mass = mass * previous + terms.sum(-1, keepdim=True)
                numerator = numerator * previous + terms @ vw[..., kb, :]
                peak = next_peak
            safe_mass = torch.where(mass > 0, mass, 1)
            pieces.append(numerator / safe_mass)
            normalizers.append(peak + safe_mass.log())
        output, log_z = torch.cat(pieces, -2), torch.cat(normalizers, -2)
        ctx.save_for_backward(q, k, v, weights, mask, output, log_z)
        return output.to(q.dtype)

    @staticmethod
    @once_differentiable
    @_without_autocast
    def backward(ctx, grad):
        q, k, v, weights, mask, output, log_z = ctx.saved_tensors
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        qw, kw, vw, go = (x.to(dtype) for x in (q, k, v, grad))
        dq, dk, dv = torch.zeros_like(qw), torch.zeros_like(kw), torch.zeros_like(vw)
        dw = torch.zeros_like(weights, dtype=dtype) if ctx.needs_input_grad[3] else None
        shape = q.shape[:-1] + (k.shape[-2],)
        w, m = weights.expand(shape), mask.expand(shape)
        scale = q.shape[-1] ** -0.5
        anchor = vw[..., :1, :]
        for start in range(0, q.shape[-2], 64):
            block = slice(start, start + 64)
            qb, gb = qw[..., block, :], go[..., block, :]
            normalizer = log_z[..., block, :]
            nonempty = torch.isfinite(normalizer)
            mean = (gb * (output[..., block, :] - anchor)).sum(-1, keepdim=True)
            single = _single_positive_rows(w, m, block, 256)
            for offset in range(0, k.shape[-2], 256):
                kb = slice(offset, offset + 256)
                wb, mb = w[..., block, kb].to(dtype), m[..., block, kb]
                logits = qb @ kw[..., kb, :].transpose(-2, -1) * scale
                log_p = logits + torch.where(wb > 0, wb, 1).log() - normalizer
                p = torch.where(mb & (wb > 0) & nonempty, log_p, -torch.inf).exp()
                positive = mb & (wb > 0)
                # Subtract a common value before the dot products. For a
                # single positive key, apply the exact zero derivative at
                # that key; zero-prior keys remain in the derivative domain.
                centered = gb @ (vw[..., kb, :] - anchor).transpose(-2, -1) - mean
                centered = torch.where(single & positive, 0, centered)
                p = torch.where(single & positive, 1, p)
                dl = p * centered
                dq[..., block, :].add_(dl @ kw[..., kb, :] * scale)
                dk[..., kb, :].add_(dl.transpose(-2, -1) @ qb * scale)
                dv[..., kb, :].add_(p.transpose(-2, -1) @ gb)
                if dw is not None:
                    log_factor = torch.where(mb & nonempty, logits - normalizer, -torch.inf)
                    contribution = _signed_exp_product(log_factor, centered)
                    qs = slice(None) if weights.shape[-2] == 1 else block
                    ks = slice(None) if weights.shape[-1] == 1 else kb
                    target = dw[..., qs, ks]
                    target.add_(contribution.sum_to_size(target.shape))
        if dw is not None:
            dw = dw.to(weights.dtype)
            invalid = ~torch.isfinite(dw)
            if bool(invalid.any()):
                index = tuple(invalid.nonzero()[0].tolist())
                raise ArithmeticError(f"nonfinite tiled attention key-prior VJP for {weights.dtype} at {index}")
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dw, None


def _signed_exp_product(log_scale, value):
    """Form exp(log_scale)*value without an unrepresentable separate scale.

    This is used inside a first-order VJP, not as an autograd decomposition.
    An exact zero cotangent remains zero even if the separate exponential would
    overflow. Final underflow remains a limitation of the requested dtype.
    """
    nonzero = value != 0
    exponent = log_scale + torch.where(nonzero, value.abs(), 1).log()
    combined = value.sign() * torch.where(nonzero, exponent, -torch.inf).exp()
    # Avoid taking log/exp of the cotangent in the ordinary scale range: those
    # extra round trips needlessly lose precision, especially for tiny roots.
    info = torch.finfo(value.dtype)
    ordinary = (log_scale >= math.log(info.tiny)) & (log_scale <= math.log(info.max)-1)
    safe_scale = torch.where(ordinary, log_scale, 0).exp()
    return torch.where(ordinary, safe_scale*value, combined)


def _finite_mass_attention(*values, error_type=ArithmeticError):
    if not bool(torch.stack([torch.isfinite(value).all() for value in values]).all()):
        raise error_type("mass-scaled attention exceeds the arithmetic range; no clipping is applied")


def _log_ratio(numerator, denominator):
    """Take a ratio before log when representable, retaining an extreme tail."""
    ratio = numerator / denominator
    direct = (ratio > 0) & torch.isfinite(ratio)
    return torch.where(direct, torch.where(direct, ratio, 1).log(),
                       torch.where(numerator > 0, numerator, 1).log() - denominator.log())


def _range_product(*values):
    """Multiply finite factors before restoring their combined binary scale.

    This keeps alpha*p*root*feature together without a lossy log(root) round
    trip or an early alpha*root underflow. Terms are restored before reduction;
    arbitrary cancellation between individually unrepresentable terms remains
    outside this helper's range. Binary scaling executes on the selected device.
    """
    fraction, exponent = binary_parts(values[0])
    for value in values[1:]:
        part, shift = binary_parts(value)
        fraction, exponent = fraction * part, exponent + shift
    return restore_binary_scale(fraction, exponent)


def _scaled_weighted_terms(logits, weight, mask, prior_scale, peak):
    """Avoid log(weight)->exp round trips after removing a common prior scale."""
    relative = weight / prior_scale
    shift = logits - torch.where(torch.isfinite(peak), peak, 0)
    direct = _signed_exp_product(shift, relative)
    # A ratio too small/large to store can still become visible after the
    # logit shift. Retain the original log ratio for precisely those entries.
    lost = (relative == 0) | ~torch.isfinite(relative)
    terms = torch.where(lost, (shift+_log_ratio(weight, prior_scale)).exp(), direct)
    return torch.where(mask & (weight > 0), terms, 0)


class _MassScaledAttention(torch.autograd.Function):
    """Bounded explicit composite VJP for a*sum(p*v), a=min(sum(mask*w),1).

    Given the *unscaled* output cotangent g, the prior derivative is
    a*exp(logit-logZ)*(g dot (v-mean)) + 1[M<=1]*(g dot mean).
    Fusing the first product avoids both a huge standalone prior derivative
    and underflow of a*g before that derivative is applied. Weak Q/K/V products
    combine their binary scale before materialization. No positive prior floor
    is introduced. The M=1 derivative is the original clamp(max=1) branch.

    Weak priors use 16-query/64-key tiles (including a feature axis), trading
    extra work for range. Ordinary/zero priors retain 64-query/256-key matrix
    contractions and still fuse the prior VJP. Logit products, mean
    reduction, centered dot products and final sums still have finite range;
    this is not an arbitrary-real-arithmetic implementation. For ordinary
    Attention with weak priors, mass_scaled=False uses the same scaled prior
    and centered VJP with amplitude one and no total-mass derivative.
    """
    @staticmethod
    @_without_autocast
    def forward(ctx, q, k, v, weights, mask, tiled, weak, mass_scaled):
        ctx.weak, ctx.mass_scaled = weak, mass_scaled
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        qw, kw, vw = q.to(dtype), k.to(dtype), v.to(dtype)
        shape = q.shape[:-1] + (k.shape[-2],)
        w, m = weights.to(dtype).expand(shape), mask.expand(shape)
        if weak:
            # Finding the scale must not materialize an extra full Q-by-K
            # tensor on the path whose declared workspace is bounded by tiles.
            scales = []
            for start in range(0, q.shape[-2], 16):
                block = slice(start, start + 16)
                maximum = torch.zeros_like(w[..., block, :1])
                for offset in range(0, k.shape[-2], 64):
                    kb = slice(offset, offset + 64)
                    maximum = torch.maximum(maximum, torch.where(m[..., block, kb], w[..., block, kb], 0).amax(-1, keepdim=True))
                scales.append(maximum)
            prior_scale = torch.cat(scales, -2)
        else:
            prior_scale = torch.ones_like(w[..., :1])
        prior_scale = torch.where(prior_scale > 0, prior_scale, 1)
        if not tiled:
            logits = qw @ kw.transpose(-2, -1) * q.shape[-1] ** -0.5
            positive = m & (w > 0)
            scores = torch.where(positive, logits + _log_ratio(w, prior_scale), -torch.inf)
            nonempty = positive.any(-1, keepdim=True)
            safe = torch.where(nonempty, scores, 0)
            log_z = torch.logsumexp(safe, -1, keepdim=True)
            p = torch.where(nonempty, (safe-log_z).exp(), 0)
            mean = p @ vw
            log_z = torch.where(nonempty, log_z, -torch.inf)
            peak, denominator = torch.zeros_like(log_z), torch.ones_like(log_z)
            if weak:
                peak = scores.amax(-1, keepdim=True)
                terms = _scaled_weighted_terms(logits, w, m, prior_scale, peak)
                denominator = terms.sum(-1, keepdim=True)
                denominator = torch.where(denominator > 0, denominator, 1)
                mean = (terms @ vw) / denominator
                log_z = peak + denominator.log()
            total = torch.where(m, w, 0).sum(-1, keepdim=True)
        else:
            means, normalizers, totals, peaks, denominators = [], [], [], [], []
            query_tile, key_tile = (16, 64) if weak else (64, 256)
            for start in range(0, q.shape[-2], query_tile):
                block = slice(start, start + query_tile)
                qb = qw[..., block, :]
                peak = qb.new_full(qb.shape[:-1] + (1,), -torch.inf)
                denominator, total = torch.zeros_like(peak), torch.zeros_like(peak)
                numerator = qb.new_zeros(qb.shape[:-1] + (v.shape[-1],))
                for offset in range(0, k.shape[-2], key_tile):
                    kb = slice(offset, offset + key_tile)
                    wb, mb = w[..., block, kb], m[..., block, kb]
                    logits = qb @ kw[..., kb, :].transpose(-2, -1) * q.shape[-1] ** -0.5
                    scores = torch.where(mb & (wb > 0), logits + _log_ratio(wb, prior_scale[..., block, :]), -torch.inf)
                    next_peak = torch.maximum(peak, scores.amax(-1, keepdim=True))
                    safe_peak = torch.where(torch.isfinite(next_peak), next_peak, 0)
                    previous = (peak-safe_peak).exp()
                    terms = (_scaled_weighted_terms(logits, wb, mb, prior_scale[..., block, :], next_peak)
                             if weak else (scores-safe_peak).exp())
                    denominator = denominator*previous + terms.sum(-1, keepdim=True)
                    numerator = numerator*previous + terms @ vw[..., kb, :]
                    total = total + torch.where(mb, wb, 0).sum(-1, keepdim=True)
                    peak = next_peak
                safe_denominator = torch.where(denominator > 0, denominator, 1)
                means.append(numerator / safe_denominator)
                normalizers.append(peak + safe_denominator.log())
                totals.append(total)
                peaks.append(peak)
                denominators.append(safe_denominator)
            mean, log_z, total = (torch.cat(parts, -2) for parts in (means, normalizers, totals))
            peak, denominator = (torch.cat(parts, -2) for parts in (peaks, denominators))
        alpha = total.clamp(max=1) if mass_scaled else torch.ones_like(total)
        output = alpha * mean
        _finite_mass_attention(mean, output)
        ctx.save_for_backward(q, k, v, weights, mask, mean, log_z, total, prior_scale, peak, denominator)
        # Mass/state arithmetic is FP32 for BF16 payloads. Do not first round
        # the weak scaled result back to BF16 before the caller consumes it.
        return output if mass_scaled else output.to(q.dtype)

    @staticmethod
    @once_differentiable
    @_without_autocast
    def backward(ctx, grad):
        q, k, v, weights, mask, mean, log_z, total, prior_scale, peak, denominator = ctx.saved_tensors
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        qw, kw, vw, go = (x.to(dtype) for x in (q, k, v, grad))
        _finite_mass_attention(go)
        needs = ctx.needs_input_grad[:4]
        dq, dk, dv = (torch.zeros_like(value) if need else None
                      for value, need in zip((qw, kw, vw), needs[:3]))
        dw = torch.zeros_like(weights, dtype=dtype) if needs[3] else None
        shape = q.shape[:-1] + (k.shape[-2],)
        w, m = weights.to(dtype).expand(shape), mask.expand(shape)
        scale = q.shape[-1] ** -0.5
        anchor = vw[..., :1, :]
        query_tile, key_tile = (16, 64) if ctx.weak else (64, 256)
        for start in range(0, q.shape[-2], query_tile):
            block = slice(start, start + query_tile)
            qb, gb = qw[..., block, :], go[..., block, :]
            normalizer, amount = log_z[..., block, :], total[..., block, :]
            nonempty = torch.isfinite(normalizer)
            alpha = amount.clamp(max=1) if ctx.mass_scaled else torch.ones_like(amount)
            log_alpha = torch.where(nonempty, alpha.log(), -torch.inf)
            log_amplitude_ratio = _log_ratio(alpha, prior_scale[..., block, :])
            single = _single_positive_rows(w, m, block, key_tile)
            # A constant offset in every value should not manufacture a prior
            # gradient by subtracting two large, nearly equal dot products.
            need_center = needs[0] or needs[1] or needs[3]
            if need_center:
                center_mean = (gb * (mean[..., block, :] - anchor)).sum(-1, keepdim=True)
            if needs[3]:
                mass_bar = (gb * mean[..., block, :]).sum(-1, keepdim=True)
                mass_bar = torch.where(nonempty & (amount <= 1) & ctx.mass_scaled, mass_bar, 0)
            for offset in range(0, k.shape[-2], key_tile):
                kb = slice(offset, offset + key_tile)
                wb, mb = w[..., block, kb], m[..., block, kb]
                keys, values = kw[..., kb, :], vw[..., kb, :]
                logits = qb @ keys.transpose(-2, -1) * scale
                if need_center:
                    centered = gb @ (values-anchor).transpose(-2, -1) - center_mean
                    centered = torch.where(single & mb & (wb > 0), 0, centered)
                valid = mb & nonempty
                relative_weight = _log_ratio(wb, prior_scale[..., block, :])
                log_p = torch.where(valid & (wb > 0), logits + relative_weight - normalizer, -torch.inf)
                log_mass = log_alpha + log_p
                if ctx.weak:
                    terms = _scaled_weighted_terms(logits, wb, mb, prior_scale[..., block, :], peak[..., block, :])
                    probability = (terms / denominator[..., block, :])[..., None]
                    probability = torch.where((single & mb & (wb > 0))[..., None], 1, probability)
                    amplitude = alpha[..., None]
                    if needs[0]:
                        dq[..., block, :].add_(_range_product(amplitude, probability, centered[..., None],
                                                            keys.unsqueeze(-3)*scale).sum(-2))
                    if needs[1]:
                        dk[..., kb, :].add_(_range_product(amplitude, probability, centered[..., None],
                                                         qb.unsqueeze(-2)*scale).sum(-3))
                    if needs[2]:
                        dv[..., kb, :].add_(_range_product(amplitude, probability, gb.unsqueeze(-2)).sum(-3))
                else:
                    if needs[0] or needs[1]:
                        dl = _signed_exp_product(log_mass, centered)
                    if needs[0]:
                        dq[..., block, :].add_(dl @ keys * scale)
                    if needs[1]:
                        dk[..., kb, :].add_(dl.transpose(-2, -1) @ qb * scale)
                    if needs[2]:
                        dv[..., kb, :].add_(log_mass.exp().transpose(-2, -1) @ gb)
                if needs[3]:
                    log_ratio = torch.where(valid, logits + (log_amplitude_ratio-normalizer), -torch.inf)
                    contribution = _signed_exp_product(log_ratio, centered) + mass_bar
                    contribution = torch.where(valid, contribution, 0)
                    qs = slice(None) if weights.shape[-2] == 1 else block
                    ks = slice(None) if weights.shape[-1] == 1 else kb
                    target = dw[..., qs, ks]
                    target.add_(contribution.sum_to_size(target.shape))
        result = tuple(value.to(source.dtype) if value is not None else None
                       for value, source in zip((dq, dk, dv, dw), (q, k, v, weights)))
        _finite_mass_attention(*(value for value in result if value is not None), error_type=GradientRangeError)
        return (*result, None, None, None, None)


def weighted_attention(q: Tensor, k: Tensor, v: Tensor, weights: Tensor | None = None,
                       mask: Tensor | None = None, backend: str = "auto", *, mass_scaled: bool = False,
                       _constant_zero_prior: Tensor | None = None) -> Tensor:
    """Compute sum(w exp(qk/sqrt(D)) v) / sum(w exp(qk/sqrt(D))).

    q/k/v have [..., H, Q/K, D/Dv]; batch axes and singleton heads broadcast.
    weights is [..., K] (batch key priors) or broadcastable to [..., H, Q, K].
    Boolean mask uses True for visible positions; [Q,K] follows normal tensor
    broadcasting. Empty rows and empty key sets return connected zeros.

    Backends are math, sdpa, tiled and npu_fusion. auto uses native SDPA on
    CPU/CUDA and native NPU fusion for supported inputs; zero trainable priors
    use tiled to retain their right derivatives. Explicit
    unavailable requests fail. The NPU adapter requires v2 and equal Q/K/V
    head widths in [1,768]; it has no CPU fallback. PSE mode is the default
    one when absent and zero for an actual frozen log prior. Positive
    trainable NPU key priors use a rank-one Q/K extension (padded to 16) and
    retain the original scaling. NPU singleton/empty rows use exact analytic
    VJPs; multi-key rows retain native fused backward, including mixed batches.
    This semantic adapter applies to both auto and explicit npu_fusion.
    FP32 native values are translated by a detached common origin before
    fusion and translated back afterwards to reduce VJP cancellation.
    Query-dependent trainable NPU priors require math/tiled. Tiled uses 64
    queries by 256 keys in forward and analytical first-order backward.
    BF16 auto sends any visible zero prior to tiled, in inference and training.
    Its native explicit key priors use one representation independent of grad
    mode (rank-one NPU Q/K extension, including explicit all-one priors).
    Query-dependent BF16 NPU priors therefore use math/tiled. None still means
    the native unit-prior path. FP32 keeps its established selection policy.

    mass_scaled=True returns min(sum(mask*weights),1) times the mean, in FP32
    (FP64 for FP64 inputs). math/tiled fuse its first-order VJP. For both ordinary
    and mass-scaled auto, a visible positive prior below sqrt(accumulation_dtype.tiny)
    selects the bounded scaled-prior tiled path in inference and training.
    This threshold selects execution only; it never clips priors or changes
    support. Explicit native requests reject that weak range for mass-scaled
    calls, and subnormal priors for ordinary calls. Native calls retain their
    fused kernel (and external amplitude multiply only for mass_scaled=True): extreme logits or
    cotangents outside this bounded admission rule are not globally protected.
    """
    if backend not in {"auto", "math", "sdpa", "tiled", "npu_fusion"}:
        raise ValueError(f"unknown attention backend: {backend}")
    if any(x.ndim < 3 for x in (q, k, v)):
        raise ValueError("q, k, v require head, sequence and feature axes")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
        raise ValueError("q, k, v must have the same dtype and device")
    if q.dtype not in {torch.float64, torch.float32, torch.bfloat16}:
        raise TypeError("attention supports BF16, FP32 and FP64 only")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2] or q.shape[-1] == 0:
        raise ValueError("incompatible attention feature/key dimensions")
    batch_heads = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    q, k, v = (x.expand(batch_heads + x.shape[-2:]) for x in (q, k, v))
    shape = batch_heads + (q.shape[-2], k.shape[-2])
    unit_prior, no_mask = weights is None, mask is None
    if weights is None:
        weights = q.new_ones(())
    elif weights.device != q.device or not (weights.dtype == q.dtype or
            q.dtype == torch.bfloat16 and weights.dtype == torch.float32):
        raise ValueError("weights must share q's dtype/device, or be FP32 with BF16 q/k/v")
    if not unit_prior and bool((~torch.isfinite(weights) | (weights < 0)).any()):
        raise ValueError("weights must be finite and nonnegative")
    if mask is None:
        mask = torch.ones((), dtype=torch.bool, device=q.device)
    elif mask.dtype != torch.bool or mask.device != q.device:
        raise ValueError("mask must be boolean and on q's device")
    weights, mask = _auxiliary(weights, shape, True), _auxiliary(mask, shape, False)
    weak_mass = False
    if not unit_prior:
        accumulation_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        threshold = math.sqrt(torch.finfo(accumulation_dtype).tiny)
        weak = (weights > 0) & (weights < threshold)
        weak_mass = bool(weak.any()) and bool((weak & mask).any())
    need_zero_check = weights.requires_grad or (q.dtype == torch.bfloat16 and backend == "auto" and not unit_prior)
    visible_zero_prior = need_zero_check and bool(((weights == 0) & mask).any())
    zero_prior_grad = weights.requires_grad and visible_zero_prior
    auto_must_tiled = weak_mass or zero_prior_grad or (q.dtype == torch.bfloat16 and visible_zero_prior)
    npu_prior_feature = weights.requires_grad or (q.dtype == torch.bfloat16 and not unit_prior)
    npu_prior_ok = not npu_prior_feature or weights.shape[-2] == 1
    npu_width = q.shape[-1] + int(npu_prior_feature)
    npu_ok = (q.device.type == "npu" and q.dtype != torch.float64
              and q.shape[-1] == v.shape[-1] and npu_width <= 768 and npu_prior_ok)
    if backend == "auto":
        backend = ("tiled" if auto_must_tiled else "npu_fusion" if npu_ok
                   else "sdpa" if q.device.type in {"cpu", "cuda"} else "tiled")
    if _constant_zero_prior is not None:
        # Reception proves these actual zero priors are locally constant at
        # their producer. Keep the usual forward kernel choice. Excluding such
        # keys preserves forward probabilities and avoids forming their VJP.
        constant = _auxiliary(_constant_zero_prior, shape, True)
        mask = mask & ~(constant & (weights == 0))
        no_mask = False
        if zero_prior_grad and backend in {"sdpa", "npu_fusion"}:
            zero_prior_grad = bool(((weights == 0) & mask).any())
    if backend == "npu_fusion" and not npu_ok:
        raise ValueError("npu_fusion requires NPU FP32/BF16 and equal head widths <=768")
    if backend == "sdpa" and q.device.type not in {"cpu", "cuda"}:
        raise ValueError("sdpa is implemented for CPU/CUDA; select an NPU backend explicitly")
    if q.shape[-2] == 0 or k.shape[-2] == 0:
        zero = (q*0).sum() + (k*0).sum() + (v*0).sum() + (weights*0).sum()
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32 if mass_scaled else q.dtype
        return q.new_zeros(q.shape[:-1] + (v.shape[-1],), dtype=dtype) + zero
    if backend in {"sdpa", "npu_fusion"}:
        native_weak = weak_mass and (mass_scaled or bool(((weights > 0)
            & (weights < torch.finfo(accumulation_dtype).tiny) & mask).any()))
        if native_weak:
            label = "mass-scaled " if mass_scaled else ""
            raise ValueError(f"weak {label}priors require auto, math or tiled; native VJP range is not qualified")
        if zero_prior_grad:
            raise ValueError("zero trainable priors require auto, math or tiled for their right derivative")
        with torch.autocast(q.device.type, enabled=False):
            native = _support_safe_native_attention if backend == "npu_fusion" else _native_attention
            output = native(q, k, v, weights, mask, backend,
                            unit_prior=unit_prior, no_mask=no_mask)
            if mass_scaled:
                dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
                amount = torch.where(mask, weights.to(dtype), 0).expand(shape).sum(-1, keepdim=True).clamp(max=1)
                output = output.to(dtype) * amount
            return output
    if mass_scaled or weak_mass:
        return _MassScaledAttention.apply(q, k, v, weights, mask, backend == "tiled", weak_mass, mass_scaled)
    if backend == "tiled":
        return _TiledAttention.apply(q, k, v, weights, mask)
    with torch.autocast(q.device.type, enabled=False):
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        qw, kw, vw = q.to(dtype), k.to(dtype), v.to(dtype)
        logits = qw @ kw.transpose(-2, -1) * q.shape[-1] ** -0.5
        p = _weighted_softmax(logits, weights.to(dtype), mask)
        return (p @ vw).to(q.dtype)
