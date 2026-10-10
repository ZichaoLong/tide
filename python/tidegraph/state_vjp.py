"""Batched first-order step graphs with independent row/slot connectivity.

Only declared builtins use this path. Numeric packed sequence values remain
authoritative; backward groups undefined cotangent patterns, never their values.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch
from .records import State
from .content import Content, SourceInput


def supported(kernel):
    from .memory import EMA, DiagonalSSM
    from .matrix_memory import MatrixMemory
    from .attention import Attention
    from .lazy_add import LazyAdd
    from .fiber_attention import FiberAttention
    from .clocked_state import ClockedState
    if type(kernel) is ClockedState:
        return supported(kernel.program)
    return type(kernel) in {EMA, DiagonalSSM, MatrixMemory, Attention, LazyAdd, FiberAttention}


def _rows(rows, transform):
    result = []
    for old, view, time in rows:
        state = State(transform(old.value), old.last_time, old.observations,
                      {k: transform(v) for k, v in sorted(old.slots.items())})
        content = Content(transform(view.value), tuple(
            SourceInput(s.slot, replace(s.atom, value=transform(s.atom.value)), transform(s.scale))
            for s in view.sources), {k: transform(v) for k, v in sorted(view.contributions.items())})
        result.append((state, content, time))
    return result


def _weights(weights, transform):
    return SimpleNamespace(**{k: transform(getattr(weights, k)) for k in ("decay", "weight", "bias", "read")},
                           extra={k: transform(v) for k, v in sorted(weights.extra.items())}, kernel=weights.kernel)


def _values(state):
    return [state.value, *[v for k, v in sorted(state.slots.items())]]


def _signature(row):
    tensors = []
    _rows([row], lambda t: tensors.append((tuple(t.shape), t.device, t.dtype, t.requires_grad)))
    old, view, time = row
    return (time, old.last_time, old.observations, tuple(sorted(old.slots)),
            tuple(s.slot for s in view.sources), tuple(sorted(view.contributions)), tuple(tensors))


class _State(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, numeric, *inputs):
        from .state_batch import step
        ctx.set_materialize_grads(False)
        leaves = [t.detach().requires_grad_(t.requires_grad) for t in inputs]
        it = iter(leaves)
        weights = _weights(plan[0], lambda _: next(it))
        rows = _rows(plan[1], lambda _: next(it))
        old, views, times = zip(*rows)
        with torch.enable_grad():
            states = step(weights.kernel, weights, list(old), list(views), list(times))
        refs = [v for s in states for v in _values(s)]
        packed = [v for s in numeric for v in _values(s)]
        for a, b in zip(states, numeric):
            if (a.last_time, a.observations, sorted(a.slots)) != (b.last_time, b.observations, sorted(b.slots)):
                raise ValueError("State VJP changed state metadata")
        for a, b in zip(refs, packed):
            if (a.shape, a.device, a.dtype) != (b.shape, b.device, b.dtype):
                raise ValueError("State VJP changed tensor metadata")
        outputs = tuple(v.detach().clone() for v in packed)
        ctx.mark_non_differentiable(*(v for v, ref in zip(outputs, refs) if not ref.requires_grad))
        ctx.rows, ctx.weights = len(rows), 4+len(weights.extra)
        ctx.row_inputs, ctx.row_outputs = (len(inputs)-ctx.weights)//len(rows), len(refs)//len(rows)
        ctx.save_for_backward(*inputs, *leaves, *refs)
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("batched State supports first-order VJP only")
        count = ctx.weights + ctx.rows*ctx.row_inputs
        saved = ctx.saved_tensors  # Includes original-version checks.
        leaves, outputs = saved[count:2*count], saved[2*count:]
        result, groups = [None]*count, {}
        for row in range(ctx.rows):
            mask = tuple(j for j in range(ctx.row_outputs) if bars[row*ctx.row_outputs+j] is not None)
            if mask:
                groups.setdefault(mask, []).append(row)
        for mask, rows in groups.items():
            ids = [i for i in range(ctx.weights) if leaves[i].requires_grad]
            ids += [ctx.weights+r*ctx.row_inputs+j for r in rows for j in range(ctx.row_inputs)
                    if leaves[ctx.weights+r*ctx.row_inputs+j].requires_grad]
            if not ids:
                continue
            roots = [r*ctx.row_outputs+j for r in rows for j in mask]
            grads = torch.autograd.grad([outputs[i] for i in roots], [leaves[i] for i in ids],
                                        [bars[i] for i in roots], retain_graph=True, allow_unused=True)
            for i, grad in zip(ids, grads):
                if grad is not None:
                    result[i] = grad if result[i] is None else result[i]+grad
        return None, None, *result


def bind_batch(weights, old, views, times, numeric):
    rows, groups = list(zip(old, views, times)), {}
    for i, row in enumerate(rows):
        groups.setdefault(_signature(row), []).append(i)
    result = list(numeric)
    for ids in groups.values():
        inputs = []
        def capture(t):
            inputs.append(t)
            return None
        plan = (_weights(weights, capture), _rows([rows[i] for i in ids], capture))
        values = iter(_State.apply(plan, [numeric[i] for i in ids], *inputs))
        for i in ids:
            s = numeric[i]
            result[i] = State(next(values), s.last_time, s.observations,
                              {k: next(values) for k in sorted(s.slots)})
    return result


def bind_sequence(weights, old, batch, states):
    current, previous, depth = list(old), [None]*len(states), 0
    while True:
        pairs = [(i, start+depth) for i, start in enumerate(batch.offsets[:-1])
                 if start+depth < batch.offsets[i+1]]
        if not pairs:
            return previous
        for i, j in pairs:
            previous[j] = current[i]
        bound = bind_batch(weights, [current[i] for i, j in pairs], [batch.views[j] for i, j in pairs],
                           [batch.times[j] for i, j in pairs], [states[j] for i, j in pairs])
        for (i, j), s in zip(pairs, bound):
            current[i] = states[j] = s
        depth += 1
