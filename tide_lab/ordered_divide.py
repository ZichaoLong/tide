"""Experiment-owned first-order quotient VJP for explicit v10 source migration.

Forward and zero-denominator policy are unchanged (the caller supplies its
masked denominator). Multiplying the incoming VJP before the second division
avoids the observed unrepresentable Jacobian intermediate. This does not clip
gradients or promise that all magnitudes or higher derivatives are safe.
"""
import torch


class OrderedDivide(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b):
        y=a/b
        ctx.save_for_backward(y,b)
        ctx.a_shape=a.shape;ctx.b_shape=b.shape
        return y

    @staticmethod
    def backward(ctx,g):
        y,b=ctx.saved_tensors
        return (g/b).sum_to_size(ctx.a_shape),(-(g*y)/b).sum_to_size(ctx.b_shape)


def ordered_divide(a,b):
    return OrderedDivide.apply(a,b)
