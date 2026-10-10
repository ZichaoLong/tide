"""Independent batched Read VJPs; no scalar autograd replay or row masking."""
import torch


class _Read(torch.autograd.Function):
    @staticmethod
    def forward(ctx, norm, dtype, device, weight, *rows):
        ctx.set_materialize_grads(False)
        x = torch.stack(rows).to(device=device, dtype=dtype)
        w = None if norm else weight.to(device=device, dtype=dtype)
        y = torch.linalg.vector_norm(x, ord=2, dim=-1) if norm else (x*w).sum(-1)
        outputs = y.unbind()
        ctx.mark_non_differentiable(*(out for row, out in zip(rows, outputs)
                                     if not row.requires_grad and (norm or not weight.requires_grad)))
        ctx.norm, ctx.input_device, ctx.input_dtype = norm, rows[0].device, rows[0].dtype
        ctx.save_for_backward(x, w, y, weight, *rows)
        return outputs

    @staticmethod
    def backward(ctx, *grads):
        if torch.is_grad_enabled():
            raise ValueError("batched Read supports first-order VJP only")
        result = [None] * (4 + len(grads))
        used = [i for i, grad in enumerate(grads) if grad is not None]
        if not used:
            return tuple(result)
        x, w, y, weight = ctx.saved_tensors[:4]
        ids = torch.tensor(used, device=x.device, dtype=torch.int64)
        dy = torch.stack([grads[i] for i in used]).unsqueeze(-1)
        x = x.index_select(0, ids)
        if any(ctx.needs_input_grad[4+i] for i in used):
            if ctx.norm:
                length = y.index_select(0, ids).unsqueeze(-1)
                # Match vector_norm's VJP order before casting back to payload.
                # Reassociating x*(dy/length) changes FP32 rounding observable
                # by FP64 payload gradients (including unit-width proposals).
                dx = dy * (x / length).masked_fill(length == 0, 0)
            else:
                dx = dy*w
            dx = dx.to(device=ctx.input_device, dtype=ctx.input_dtype)
            for i, row in zip(used, dx.unbind()):
                if ctx.needs_input_grad[4+i]:
                    result[4+i] = row
        if not ctx.norm and ctx.needs_input_grad[3]:
            result[3] = (dy*x).sum(0).to(device=weight.device, dtype=weight.dtype)
        return tuple(result)


def supported(program):
    from .readout import LinearRead, NormRead, NormFloat32Read
    from .placement_read import PlacedRead
    return type(program) in {LinearRead, NormRead, NormFloat32Read, PlacedRead}


def evaluate(program, weights, requests):
    if not requests:
        return []
    if getattr(program, "identity", False):
        return program.batch(weights, requests)
    rows = [r.content.value if r.state is None else r.state.value for r in requests]
    dtype = {"payload": rows[0].dtype, "float32": torch.float32, "float64": torch.float64}[program.precision]
    return list(_Read.apply(program.profile != "linear-v1", dtype,
                            program.descriptor_device(rows[0].device), weights.read, *rows))
