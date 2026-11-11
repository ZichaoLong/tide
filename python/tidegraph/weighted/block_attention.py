"""KV §3–4: bank contractions and a bank-sized first-order transpose.

q/k/v use [owner, head, query-or-history, channel]. Each query's visible
positive entries must have the same first positive entry within its owner.
Unlike independent snapshot Attention, K/V are actual shared gradient roots.
"""
import math
import torch


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, weights):
        ctx.set_materialize_grads(False)
        positive = weights > 0
        active = positive.any(-1)
        safe = torch.where(positive, weights, 1)
        logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + safe.log()[:, None]
        logits = logits.masked_fill(~positive[:, None], -torch.inf)
        probabilities = torch.where(active[:, None, :, None], logits, 0).softmax(-1)
        probabilities = torch.where(active[:, None, :, None], probabilities, 0)
        ctx.save_for_backward(q, k, v, weights, probabilities)
        return probabilities @ v

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError('KV block Attention supports first-order VJP only')
        q, k, v, weights, p = ctx.saved_tensors
        if bar is None:
            return None, None, None, None
        # One common positive anchor per bank avoids [B,H,Q,K,D] expansion.
        used = (weights > 0).any(1)
        index = used.to(torch.int64).argmax(-1)[:, None, None, None]
        anchor = v.gather(2, index.expand(-1, v.shape[1], 1, v.shape[-1]))
        delta = torch.where(used[:, None, :, None], v, anchor) - anchor
        mean = p @ delta
        dl = p * (bar @ delta.transpose(-1, -2) - (bar * mean).sum(-1, keepdim=True))
        dq = dl @ k / math.sqrt(q.shape[-1]) if ctx.needs_input_grad[0] else None
        dk = dl.transpose(-1, -2) @ q / math.sqrt(q.shape[-1]) if ctx.needs_input_grad[1] else None
        dv = p.transpose(-1, -2) @ bar if ctx.needs_input_grad[2] else None
        dw = dl.sum(1) / torch.where(weights > 0, weights, 1) if ctx.needs_input_grad[3] else None
        return dq, dk, dv, dw


attention = _Attention.apply
