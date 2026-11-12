"""KV §3–4: independent node outputs in real node×B contractions.

Packing occurs under no-grad; the explicit first-order transpose selects only
outputs with a supplied adjoint. An unused node stays disconnected, including
when another node supplies a connected zero. Original versions remain guarded.
"""
import math
import torch
from .row_linear import linear_blocks
from .packing_observation import bucket


def linear(values, matrices):
    if len(values) == 1:
        return [values[0] @ matrices[0].t()]
    rows = [x.reshape(-1, x.shape[-1]) for x in values]
    return [y.reshape(*x.shape[:-1], w.shape[0])
            for y, x, w in zip(linear_blocks(rows, matrices), values, matrices)]


class _Contractions(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kind, *inputs):
        ctx.set_materialize_grads(False)
        ctx.kind, ctx.arity = kind, {"matmul": 2, "attention": 4, "logits": 3}[kind]
        n, a = len(inputs)//ctx.arity, ctx.arity
        ctx.batch = inputs[0].shape[0]
        packed = [torch.cat(inputs[j::a], 0) for j in range(a)]
        if kind == "matmul":
            result, saved = torch.bmm(*packed), []
        else:
            v, w = packed[-2:]
            x = [t.float() for t in packed] if v.dtype == torch.float16 else packed
            logits = x[0] if kind == "logits" else x[0] @ x[1].transpose(-1, -2)/math.sqrt(x[0].shape[-1])
            v, w = x[-2:]
            positive = w > 0
            active = positive.any(-1)[:, None, :, None]
            logits = (logits+torch.where(positive, w, 1).log()[:, None]).masked_fill(~positive[:, None], -torch.inf)
            p = torch.where(active, torch.where(active, logits, 0).softmax(-1), 0)
            result, saved = (p@v).to(packed[-2].dtype), [p]
        ctx.count = len(inputs)
        ctx.save_for_backward(*inputs, *packed, *saved)
        outputs = tuple(result.split(ctx.batch))
        ctx.mark_non_differentiable(*(y for i, y in enumerate(outputs)
                                     if not any(t.requires_grad for t in inputs[i*a:(i+1)*a])))
        bucket("column_"+kind+"_nodes", inputs[0].shape[-2], n)
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("node column contractions support first-order VJP only")
        saved = ctx.saved_tensors  # Includes every original version guard.
        a, b = ctx.arity, ctx.batch
        result = [None]*ctx.count
        ids = [i for i, g in enumerate(bars) if g is not None]
        if not ids:
            return None, *result
        packed = saved[ctx.count:ctx.count+a]
        all_used = len(ids) == len(bars)
        index = None if all_used else torch.tensor([i*b+j for i in ids for j in range(b)], device=packed[0].device)
        take = lambda t: t if all_used else t.index_select(0, index)
        x = [take(t) for t in packed]
        g = torch.cat([bars[i] for i in ids])
        if ctx.kind == "matmul":
            gradients = (torch.bmm(g, x[1].transpose(1, 2)), torch.bmm(x[0].transpose(1, 2), g))
        else:
            if x[-2].dtype == torch.float16:
                x, g = [t.float() for t in x], g.float()
            v, w, p = x[-2], x[-1], take(saved[-1])
            used = (w > 0).any(1)
            anchor_index = used.to(torch.int64).argmax(-1)[:, None, None, None]
            anchor = v.gather(2, anchor_index.expand(-1, v.shape[1], 1, v.shape[-1]))
            delta = torch.where(used[:, None, :, None], v, anchor)-anchor
            dl = p*(g@delta.transpose(-1, -2)-(g*(p@delta)).sum(-1, keepdim=True))
            dl = torch.where((w > 0).sum(-1)[:, None, :, None] > 1, dl, 0)
            dv, dw = p.transpose(-1, -2)@g, dl.sum(1)/torch.where(w > 0, w, 1)
            gradients = (dl, dv, dw) if ctx.kind == "logits" else (
                dl@x[1]/math.sqrt(x[0].shape[-1]), dl.transpose(-1, -2)@x[0]/math.sqrt(x[0].shape[-1]), dv, dw)
        for j, gradient in enumerate(gradients):
            for i, piece in zip(ids, gradient.split(b)):
                if ctx.needs_input_grad[1+i*a+j]:
                    result[i*a+j] = piece.to(saved[i*a+j].dtype)
        return None, *result


def contract(kind, operands):
    """Equal actual operand shapes; callers own configuration and byte grouping."""
    if not operands:
        return []
    arity = {"matmul": 2, "attention": 4, "logits": 3}[kind]
    first = operands[0]
    if len(first) != arity or first[0].shape[0] < 1:
        raise ValueError("invalid node contraction operands")
    for row in operands:
        if len(row) != arity or any((x.shape, x.dtype, x.device) != (y.shape, y.dtype, y.device)
                                   for x, y in zip(row, first)):
            raise ValueError("node contraction shape/dtype/device mismatch")
    if len(operands) == 1:
        if kind == "matmul":
            return [torch.bmm(*first)]
        if kind == "attention":
            from .block_attention import attention
            return [attention(*first)]
        from .column_attention import logit_attention
        return [logit_attention(*first)]
    return list(_Contractions.apply(kind, *(t for row in operands for t in row)))
