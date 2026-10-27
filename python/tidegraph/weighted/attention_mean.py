"""KV §3–4 softmax value read, with a centered exact first-order VJP.

The forward probability-weighted sum is unchanged. Centering before the logit
contraction preserves value-translation invariance, even for nearly equal values.
"""
import torch


class _AttentionMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, values):
        ctx.set_materialize_grads(False)
        probabilities = (logits.float() if values.dtype == torch.float16 else logits).softmax(-1)
        ctx.save_for_backward(logits, values, probabilities)
        operand = values.float() if values.dtype == torch.float16 else values
        result = (probabilities.unsqueeze(-1)*operand).sum(-2)
        return result.to(values.dtype) if values.dtype == torch.float16 else result

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("weighted Attention supports first-order VJP only")
        if bar is None:
            return None, None
        original_logits, original_values, probabilities = ctx.saved_tensors
        values = original_values
        if values.dtype == torch.float16:
            values, bar = values.float(), bar.float()
        delta = values-values[..., :1, :]
        centered = delta-(probabilities.unsqueeze(-1)*delta).sum(-2, keepdim=True)
        logits = probabilities*(centered*bar.unsqueeze(-2)).sum(-1)
        return logits.to(original_logits.dtype), (probabilities.unsqueeze(-1)*bar.unsqueeze(-2)).to(original_values.dtype)


attention_mean = _AttentionMean.apply
