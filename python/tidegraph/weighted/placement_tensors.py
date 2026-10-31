"""Explicit placement copies preserve raw int64 and original VJP versions."""
import torch
from ..communication import copy as copy_tensor, policy

PACK_BYTES = 8*1024**2


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, device, dtype):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(value)
        ctx.policy = policy()
        return copy_tensor(value, device, dtype, selected=ctx.policy)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("weighted placement supports first-order VJP only")
        value, = ctx.saved_tensors
        return (None if bar is None else copy_tensor(bar, value.device, value.dtype, selected=ctx.policy)), None, None


def move(value, device, dtype=None):
    if value is None:
        return None
    dtype = (dtype or value.dtype) if value.is_floating_point() else value.dtype
    if value.device == device and value.dtype == dtype:
        return value
    return _Copy.apply(value, device, dtype) if value.requires_grad else copy_tensor(value, device, dtype)


class _Copies(torch.autograd.Function):
    @staticmethod
    def forward(ctx, device, dtype, *rows):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*rows)
        ctx.policy = policy()
        packed = copy_tensor(torch.cat([r.reshape(-1) for r in rows]), device, dtype, selected=ctx.policy)
        outputs, offset = [], 0
        for row in rows:
            outputs.append(packed.narrow(0, offset, row.numel()).reshape(row.shape))
            offset += row.numel()
        outputs = tuple(outputs)
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
            copied = copy_tensor(torch.cat([bars[i].reshape(-1) for i in ids]), rows[0].device, rows[0].dtype, selected=ctx.policy)
            offset = 0
            for i in ids:
                result[i+2] = copied.narrow(0, offset, rows[i].numel()).reshape(rows[i].shape)
                offset += rows[i].numel()
        return tuple(result)


def mapper(values, device, dtype):
    """Bounded heterogeneous fields; actual aliases copied once, private roots."""
    unique = {id(t): t for t in values}
    groups, result = {}, {}
    for key, value in unique.items():
        target = dtype if value.is_floating_point() else value.dtype
        if value.device == device and value.dtype == target:
            result[key] = value
        else:
            signature = (value.device, value.dtype, target)
            groups.setdefault(signature, []).append(key)
    for (_, _, target), keys in groups.items():
        first = unique[keys[0]]
        element = max(first.element_size(), torch.empty((), dtype=target).element_size())
        start = 0
        while start < len(keys):
            end, nbytes = start, 0
            while end < len(keys):
                size = max(1, unique[keys[end]].numel())*element
                if end > start and nbytes+size > PACK_BYTES:
                    break
                nbytes += size
                end += 1
            ids = keys[start:end]
            rows = [unique[key] for key in ids]
            copies = [move(rows[0], device, target)] if len(rows) == 1 else _Copies.apply(device, target, *rows)
            result.update(zip(ids, copies))
            start = end
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


def controls_batch(results, requests):
    if len(results) < 2 or all(r.payload_device.type == "cpu" for r in requests):
        return [controls(result, request) for result, request in zip(results, requests)]
    from ..region_metadata import positive
    values = [{v: move(c, request.payload_device, request.payload_dtype) for v, c in result.controls.items()}
              for result, request in zip(results, requests)]
    flags = iter(positive([c for row in values for c in row.values()]))
    for result, row in zip(results, values):
        active = {v for v in row if next(flags)}
        if active != result.active:
            raise ValueError("weighted control cast changes positive support")
        result.controls = row
    return results
