"""Independent vector/matrix roots in actual batched linear maps (KV §1.2).

Stacking alone connects unused rows to zero gradients. This first-order boundary
returns None for unused rows, including independent equal-valued inputs/weights.
"""
import torch
from . import packing_observation as observation


class _Rows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.set_materialize_grads(False)
        xs, ws = inputs[::2], inputs[1::2]
        x, w = torch.stack(xs), torch.stack(ws)
        observation.projection("private_bmm", len(xs))
        values = torch.bmm(w, x.unsqueeze(-1)).squeeze(-1)
        outputs = tuple(values.unbind())
        ctx.count = len(inputs)
        ctx.save_for_backward(*inputs, x, w)
        ctx.mark_non_differentiable(*(out for out, a, b in zip(outputs, xs, ws)
                                     if not (a.requires_grad or b.requires_grad)))
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("independent linear rows support first-order VJP only")
        saved = ctx.saved_tensors  # Check every original root's version.
        x, w = saved[-2:]
        result = [None]*ctx.count
        ids = [i for i, bar in enumerate(bars) if bar is not None]
        if ids:
            g = torch.stack([bars[i] for i in ids])
            dx = torch.bmm(w[ids].transpose(1, 2), g.unsqueeze(-1)).squeeze(-1)
            dw = g.unsqueeze(-1)*x[ids].unsqueeze(1)
            for k, i in enumerate(ids):
                result[2*i], result[2*i+1] = dx[k], dw[k]
        return tuple(result)


def linear_rows(xs, weights):
    """y[i]=weights[i]@xs[i]; same shapes/dtype/device, separate root identities."""
    if len(xs) != len(weights):
        raise ValueError("independent linear row count mismatch")
    if not xs:
        return []
    x0, w0 = xs[0], weights[0]
    if x0.ndim != 1 or w0.ndim != 2 or w0.shape[1] != len(x0):
        raise ValueError("independent linear row shape mismatch")
    for x, w in zip(xs, weights):
        if (x.shape != x0.shape or w.shape != w0.shape or
                any((t.dtype, t.device) != (x0.dtype, x0.device) for t in (x, w))):
            raise ValueError("independent linear row metadata mismatch")
    return list(_Rows.apply(*[t for row in zip(xs, weights) for t in row]))
