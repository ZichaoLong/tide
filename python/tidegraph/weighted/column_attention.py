"""KV positional logits with one value-bank transpose, never event objects."""
import torch


class _LogitAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, values, weights):
        ctx.set_materialize_grads(False)
        originals = logits, values, weights
        if values.dtype == torch.float16:
            logits, values, weights = (x.float() for x in originals)
        positive = weights > 0
        active = positive.any(-1)[:, None, :, None]
        l = logits+torch.where(positive, weights, 1).log()[:, None]
        l = l.masked_fill(~positive[:, None], -torch.inf)
        p = torch.where(active, l, 0).softmax(-1)
        p = torch.where(active, p, 0)
        ctx.save_for_backward(*originals, p)
        return (p@values).to(originals[1].dtype)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("KV positional Attention supports first-order VJP only")
        if bar is None:
            return None, None, None
        logits, v, weights, p = ctx.saved_tensors
        originals = logits, v, weights
        if v.dtype == torch.float16:
            v, weights, bar = v.float(), weights.float(), bar.float()
        used = (weights > 0).any(1)
        index = used.to(torch.int64).argmax(-1)[:, None, None, None]
        anchor = v.gather(2, index.expand(-1, v.shape[1], 1, v.shape[-1]))
        delta = torch.where(used[:, None, :, None], v, anchor)-anchor
        dl = p*(bar@delta.transpose(-1, -2)-(bar*(p@delta)).sum(-1, keepdim=True))
        dl = torch.where((weights > 0).sum(-1)[:, None, :, None] > 1, dl, 0)
        bars = (dl if ctx.needs_input_grad[0] else None,
                p.transpose(-1, -2)@bar if ctx.needs_input_grad[1] else None,
                dl.sum(1)/torch.where(weights > 0, weights, 1) if ctx.needs_input_grad[2] else None)
        return tuple(None if g is None else g.to(x.dtype) for g, x in zip(bars, originals))


logit_attention = _LogitAttention.apply
