"""One-row first-order boundary without rebuilding a private autograd graph.

Architecture §2–4: with exactly one fiber/query, there are no other row roots
to disconnect. Keep original version checks and undefined field cotangents.
"""
import torch


class _Boundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, count, *values):
        ctx.set_materialize_grads(False)
        ctx.count = count
        ctx.save_for_backward(*values[:count])
        refs = values[count:]
        outputs = tuple(t.detach().clone() for t in refs)
        ctx.mark_non_differentiable(*(v for v, r in zip(outputs, refs) if not r.requires_grad))
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("weighted singleton supports first-order VJP only")
        _ = ctx.saved_tensors  # Check original versions, including unused fields.
        return (None,)+(None,)*ctx.count+bars


def boundary(originals, refs):
    return _Boundary.apply(len(originals), *originals, *refs)
