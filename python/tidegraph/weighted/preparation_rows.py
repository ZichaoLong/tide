"""KV §1.2: joint preparation with independently addressable row roots.

Parameters occur once per batch. Selecting one output must not connect unused
input rows to zero gradients through the internal stack. Ephemeral outputs are
read-only operands, not public persistent state or parameter owners.
"""
import torch
from .tensors import norm
from .packing import project


class _Rows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kind, epsilon, *inputs):
        ctx.set_materialize_grads(False)
        count, width = (1, 1) if kind == "base" else (3, 2)
        leaves = [t.detach().requires_grad_(t.requires_grad) for t in inputs]
        weights, fields = leaves[:count], leaves[count:]
        with torch.enable_grad():
            if kind == "base":
                x = norm(torch.stack(fields)[:, 1:], epsilon)
                columns = (x, project(x, weights[0]))
            else:
                x, base = torch.stack(fields[::2]), torch.stack(fields[1::2])
                columns = (base+project(project(x, weights[0]), weights[1]),
                           project(x, weights[2]))
            outputs = [t for pair in zip(*(c.unbind() for c in columns)) for t in pair]
        numeric = tuple(t.detach() for t in outputs)
        ctx.mark_non_differentiable(*(t for t, root in zip(numeric, outputs) if not root.requires_grad))
        ctx.count, ctx.width, ctx.size = count, width, len(inputs)
        ctx.save_for_backward(*inputs, *leaves, *outputs)
        return numeric

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("KV preparation supports first-order VJP only")
        saved = ctx.saved_tensors  # Validate every original parameter/input version.
        leaves, outputs = saved[ctx.size:2*ctx.size], saved[2*ctx.size:]
        used = [i for i, bar in enumerate(bars) if bar is not None]
        rows = sorted({i//2 for i in used})
        ids = list(range(ctx.count))
        ids += [ctx.count+r*ctx.width+j for r in rows for j in range(ctx.width)]
        ids = [i for i in ids if leaves[i].requires_grad]
        result = [None]*ctx.size
        if used and ids:
            grads = torch.autograd.grad([outputs[i] for i in used], [leaves[i] for i in ids],
                                        [bars[i] for i in used], retain_graph=True, allow_unused=True)
            for i, grad in zip(ids, grads):
                result[i] = grad
        return None, None, *result


def prepare_rows(kind, weights, rows, epsilon=0.):
    values = _Rows.apply(kind, epsilon, *weights, *[t for row in rows for t in row])
    return list(zip(values[::2], values[1::2]))
