"""Batched affine work with independent public row VJPs."""
import torch


class _Rows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, *rows):
        ctx.set_materialize_grads(False)
        x = torch.stack(rows)
        result = tuple(torch.nn.functional.linear(x, weight).unbind())
        ctx.mark_non_differentiable(*(v for v, r in zip(result, rows)
                                      if not (r.requires_grad or weight.requires_grad)))
        ctx.save_for_backward(x, weight)
        return result

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("isolated linear supports first-order VJP only")
        result = [None] * (len(bars) + 1)
        used = [i for i, g in enumerate(bars) if g is not None]
        if not used:
            return tuple(result)
        x, weight = ctx.saved_tensors
        dy = torch.stack([bars[i] for i in used])
        if any(ctx.needs_input_grad[i+1] for i in used):
            dx = dy @ weight
            for j, i in enumerate(used):
                if ctx.needs_input_grad[i+1]:
                    result[i+1] = dx[j]
        if ctx.needs_input_grad[0]:
            result[0] = dy.t() @ x[used]
        return tuple(result)


def linear(rows, weight):
    if not rows:
        return []
    if weight.ndim != 2 or weight.device.type not in {"cpu", "cuda", "npu"} or weight.dtype not in (torch.float32, torch.float64):
        raise ValueError("isolated linear requires a supported FP32/FP64 weight matrix")
    if any(r.shape != (weight.shape[1],) or r.device != weight.device or r.dtype != weight.dtype for r in rows):
        raise ValueError("isolated linear row metadata mismatch")
    return list(_Rows.apply(weight, *rows))
