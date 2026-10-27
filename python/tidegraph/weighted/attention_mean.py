"""KV §3–4 softmax value read, with a centered exact first-order VJP.

The forward probability-weighted sum is unchanged. Centering before the logit
contraction preserves value-translation invariance, even for nearly equal values.
"""
import torch


class _AttentionMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, values):
        ctx.set_materialize_grads(False)
        probabilities = logits.softmax(-1)
        ctx.save_for_backward(logits, values, probabilities)
        return (probabilities.unsqueeze(-1)*values).sum(-2)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("weighted Attention supports first-order VJP only")
        if bar is None:
            return None, None
        _, values, probabilities = ctx.saved_tensors
        delta = values-values[..., :1, :]
        centered = delta-(probabilities.unsqueeze(-1)*delta).sum(-2, keepdim=True)
        logits = probabilities*(centered*bar.unsqueeze(-2)).sum(-1)
        return logits, probabilities.unsqueeze(-1)*bar.unsqueeze(-2)


attention_mean = _AttentionMean.apply
