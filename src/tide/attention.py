"""Normalized weighted attention, with zero rows and weight right derivatives.

The Full/Read amplitude min(1, sum(weights)) belongs to the caller.  All paths
implement first-order VJPs; a zero-weight key in a nonempty row retains its
right derivative.  At an entirely zero row we choose the zero VJP.
"""

from __future__ import annotations

import math
from functools import wraps

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


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
        logits, p, log_z, mask, nonempty, _ = ctx.saved_tensors
        centered = grad - (p * grad).sum(-1, keepdim=True)
        factor = torch.where(mask & nonempty, logits - log_z, -torch.inf).exp()
        return p * centered, factor * centered, None


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
    unit = unit_prior or (not weights.requires_grad and bool((weights == 1).all()))
    visible = mask if unit else mask & (weights > 0)
    log_weights = None if unit else torch.where(weights > 0, weights, 1).log()
    if backend == "sdpa":
        bias = (None if no_mask else mask) if unit else torch.where(visible, log_weights, -torch.inf)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    else:
        import torch_npu  # Vendor registration is never required by CPU entry points.

        fusion = getattr(torch_npu, "npu_fusion_attention_v2", None)
        if fusion is None:
            raise RuntimeError("npu_fusion requires npu_fusion_attention_v2 with pse_type=0")
        batch = math.prod(q.shape[:-3])
        heads, queries, width = q.shape[-3:]
        keys, value_width, scale = k.shape[-2], v.shape[-1], width ** -0.5
        pse = None
        if weights.requires_grad:
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
            inner_precise=2, pse_type=0,
        )[0].reshape_as(q)[..., :value_width]
    return output if unit and no_mask else torch.where(visible.any(-1, keepdim=True), output, 0)


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
        for start in range(0, q.shape[-2], 64):
            block = slice(start, start + 64)
            qb, gb = qw[..., block, :], go[..., block, :]
            normalizer = log_z[..., block, :]
            nonempty = torch.isfinite(normalizer)
            mean = (gb * output[..., block, :]).sum(-1, keepdim=True)
            for offset in range(0, k.shape[-2], 256):
                kb = slice(offset, offset + 256)
                wb, mb = w[..., block, kb].to(dtype), m[..., block, kb]
                logits = qb @ kw[..., kb, :].transpose(-2, -1) * scale
                log_p = logits + torch.where(wb > 0, wb, 1).log() - normalizer
                p = torch.where(mb & (wb > 0) & nonempty, log_p, -torch.inf).exp()
                centered = gb @ vw[..., kb, :].transpose(-2, -1) - mean
                dl = p * centered
                dq[..., block, :].add_(dl @ kw[..., kb, :] * scale)
                dk[..., kb, :].add_(dl.transpose(-2, -1) @ qb * scale)
                dv[..., kb, :].add_(p.transpose(-2, -1) @ gb)
                if dw is not None:
                    factor = torch.where(mb & nonempty, logits - normalizer, -torch.inf).exp()
                    contribution = factor * centered
                    qs = slice(None) if weights.shape[-2] == 1 else block
                    ks = slice(None) if weights.shape[-1] == 1 else kb
                    target = dw[..., qs, ks]
                    target.add_(contribution.sum_to_size(target.shape))
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                None if dw is None else dw.to(weights.dtype), None)


def weighted_attention(q: Tensor, k: Tensor, v: Tensor, weights: Tensor | None = None,
                       mask: Tensor | None = None, backend: str = "auto") -> Tensor:
    """Compute sum(w exp(qk/sqrt(D)) v) / sum(w exp(qk/sqrt(D))).

    q/k/v have [..., H, Q/K, D/Dv]; batch axes and singleton heads broadcast.
    weights is [..., K] (batch key priors) or broadcastable to [..., H, Q, K].
    Boolean mask uses True for visible positions; [Q,K] follows normal tensor
    broadcasting. Empty rows and empty key sets return connected zeros.

    Backends are math, sdpa, tiled and npu_fusion. auto uses native SDPA on
    CPU/CUDA and native NPU fusion for supported inputs; zero trainable priors
    use tiled to retain their right derivatives. Explicit
    unavailable requests fail. The NPU adapter requires v2, pse_type=0 and
    equal Q/K/V head widths in [1,768]; it has no CPU fallback. Positive
    trainable NPU key priors use a rank-one Q/K extension (padded to 16) and
    retain the original scaling. Native fused backward remains in use.
    Query-dependent trainable NPU priors require math/tiled. Tiled uses 64
    queries by 256 keys in forward and analytical first-order backward.
    """
    if backend not in {"auto", "math", "sdpa", "tiled", "npu_fusion"}:
        raise ValueError(f"unknown attention backend: {backend}")
    if any(x.ndim < 3 for x in (q, k, v)):
        raise ValueError("q, k, v require head, sequence and feature axes")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
        raise ValueError("q, k, v must have the same dtype and device")
    if q.dtype not in {torch.float64, torch.float32}:
        raise TypeError("attention supports FP32 and FP64 only")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2] or q.shape[-1] == 0:
        raise ValueError("incompatible attention feature/key dimensions")
    batch_heads = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    q, k, v = (x.expand(batch_heads + x.shape[-2:]) for x in (q, k, v))
    shape = batch_heads + (q.shape[-2], k.shape[-2])
    unit_prior, no_mask = weights is None, mask is None
    if weights is None:
        weights = q.new_ones(())
    elif weights.device != q.device or weights.dtype != q.dtype:
        raise ValueError("weights must share q's dtype and device")
    if not unit_prior and bool((~torch.isfinite(weights) | (weights < 0)).any()):
        raise ValueError("weights must be finite and nonnegative")
    if mask is None:
        mask = torch.ones((), dtype=torch.bool, device=q.device)
    elif mask.dtype != torch.bool or mask.device != q.device:
        raise ValueError("mask must be boolean and on q's device")
    weights, mask = _auxiliary(weights, shape, True), _auxiliary(mask, shape, False)
    zero_prior_grad = weights.requires_grad and bool(((weights == 0) & mask).any())
    npu_prior_ok = not weights.requires_grad or weights.shape[-2] == 1
    npu_width = q.shape[-1] + int(weights.requires_grad)
    npu_ok = (q.device.type == "npu" and q.dtype != torch.float64
              and q.shape[-1] == v.shape[-1] and npu_width <= 768 and npu_prior_ok)
    if backend == "auto":
        backend = ("tiled" if zero_prior_grad else "npu_fusion" if npu_ok
                   else "sdpa" if q.device.type in {"cpu", "cuda"} else "tiled")
    if backend == "npu_fusion" and not npu_ok:
        raise ValueError("npu_fusion requires NPU FP32 and equal head widths <=768")
    if backend == "sdpa" and q.device.type not in {"cpu", "cuda"}:
        raise ValueError("sdpa is implemented for CPU/CUDA; select an NPU backend explicitly")
    if q.shape[-2] == 0 or k.shape[-2] == 0:
        zero = (q.sum() + k.sum() + v.sum() + weights.sum()) * 0
        return q.new_zeros(q.shape[:-1] + (v.shape[-1],)) + zero
    if backend in {"sdpa", "npu_fusion"}:
        if zero_prior_grad:
            raise ValueError("zero trainable priors require auto, math or tiled for their right derivative")
        with torch.autocast(q.device.type, enabled=False):
            return _native_attention(q, k, v, weights, mask, backend,
                                     unit_prior=unit_prior, no_mask=no_mask)
    if backend == "tiled":
        return _TiledAttention.apply(q, k, v, weights, mask)
    with torch.autocast(q.device.type, enabled=False):
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        qw, kw, vw = q.to(dtype), k.to(dtype), v.to(dtype)
        logits = qw @ kw.transpose(-2, -1) * q.shape[-1] ** -0.5
        p = _weighted_softmax(logits, weights.to(dtype), mask)
        return (p @ vw).to(q.dtype)
