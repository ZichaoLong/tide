"""Explicit placement copies preserve raw int64 and original VJP versions."""
import torch

PACK_BYTES = 8*1024**2


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, device, dtype):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(value)
        return value.to(device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("weighted placement supports first-order VJP only")
        value, = ctx.saved_tensors
        return (None if bar is None else bar.to(value)), None, None


def move(value, device, dtype=None):
    if value is None:
        return None
    dtype = (dtype or value.dtype) if value.is_floating_point() else value.dtype
    if value.device == device and value.dtype == dtype:
        return value
    return _Copy.apply(value, device, dtype) if value.requires_grad else value.to(device=device, dtype=dtype)


class _Copies(torch.autograd.Function):
    @staticmethod
    def forward(ctx, device, dtype, *rows):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*rows)
        outputs = tuple(torch.stack(rows).to(device=device, dtype=dtype).unbind())
        ctx.mark_non_differentiable(*(v for v, r in zip(outputs, rows) if not r.requires_grad))
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("weighted placement supports first-order VJP only")
        rows = ctx.saved_tensors  # Check every original version before transfer.
        result = [None]*(len(rows)+2)
        ids = [i for i, bar in enumerate(bars) if bar is not None and ctx.needs_input_grad[i+2]]
        if ids:
            copied = torch.stack([bars[i] for i in ids]).to(rows[0]).unbind()
            for i, value in zip(ids, copied):
                result[i+2] = value
        return tuple(result)


def mapper(values, device, dtype):
    """Bounded shape groups, with actual aliases copied once and private roots."""
    unique = {id(t): t for t in values}
    groups, result = {}, {}
    for key, value in unique.items():
        target = dtype if value.is_floating_point() else value.dtype
        if value.device == device and value.dtype == target:
            result[key] = value
        else:
            signature = (value.device, value.dtype, target, tuple(value.shape))
            groups.setdefault(signature, []).append(key)
    for (_, _, target, _), keys in groups.items():
        first = unique[keys[0]]
        limit = max(1, PACK_BYTES//max(1, first.numel()*max(first.element_size(), torch.empty((), dtype=target).element_size())))
        for start in range(0, len(keys), limit):
            ids = keys[start:start+limit]
            rows = [unique[key] for key in ids]
            copies = [move(rows[0], device, target)] if len(rows) == 1 else _Copies.apply(device, target, *rows)
            result.update(zip(ids, copies))
    return lambda t: result[id(t)]


def controls(result, request):
    values = {v: move(c, request.payload_device, request.payload_dtype) for v, c in result.controls.items()}
    # A representable positive activity cannot silently become an absent/zero
    # payload through the explicit scoring-to-payload cast.
    active = {v for v, c in values.items() if c.detach() > 0}
    if active != result.active:
        raise ValueError("weighted control cast changes positive support")
    result.controls = values
    return result
