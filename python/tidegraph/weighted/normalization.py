"""Architecture §1: epsilon RMS normalization and its normalized first-order VJP.

The forward x/sqrt(epsilon+mean(x²)) is unchanged. Contract the adjoint with
the normalized vector before dividing, instead of forming two inverse powers.
"""
import torch


class _Normalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, epsilon):
        ctx.set_materialize_grads(False)
        denominator = torch.sqrt(epsilon+value.square().mean(-1, keepdim=True))
        normalized = value/denominator
        ctx.save_for_backward(value, normalized, denominator)
        return normalized

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("weighted normalization supports first-order VJP only")
        if bar is None:
            return None, None
        _, normalized, denominator = ctx.saved_tensors
        return (bar-normalized*(bar*normalized).mean(-1, keepdim=True))/denominator, None


norm = _Normalize.apply
