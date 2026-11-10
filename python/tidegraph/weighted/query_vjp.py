"""Independent first-order roots for the actual batched Read/Full computations.

Queries share parameters, not private state. Structural groups preserve the
zero-attention and empty-cache branches; undefined field cotangents select rows.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch
from ..full import FullResult
from ..records import State
from .config import KVConfig
from .parameter_tree import rebind
from .numeric_time import passive_decay
from . import packing_observation as observation


def _rows(requests, kind, transform):
    result = []
    for r in requests:
        original = r.state if kind == "read" else r.comparison
        state = None if original is None else State(transform(original.value), original.last_time,
            original.observations, {k: transform(t) for k, t in sorted(original.slots.items())})
        content = replace(r.content, value=transform(r.content.value))
        fields = dict(state=state) if kind == "read" else dict(comparison=state, control=transform(r.control))
        result.append(replace(r, content=content, **fields))
    return result


def _signature(memory, weights, request, kind, branch_override=None):
    tensors = []
    _rows([request], kind, lambda t: tensors.append((t.shape, t.dtype, t.device, t.requires_grad)))
    state = request.state if kind == "read" else request.comparison
    slots = None if state is None else tuple(sorted(state.slots))
    if branch_override is not None:
        # Only the KV cache axis is ragged. All other shape/dtype/device/grad
        # structure remains part of the boundary, including initial vs state.
        for i in range(1, 1+len(slots or ())):
            shape, *rest = tensors[i]
            tensors[i] = ((-1, *shape[1:]), *rest)
        return slots, branch_override, tuple(tensors)
    branch = ()
    c = memory.config
    if isinstance(c, KVConfig) and (kind == "full" or c.memory_read == "qry"):
        state = memory.initial(weights) if state is None else state
        mask, age = memory.visible(state, request.time)
        writes = state.slots["weights"][mask].detach()
        branch = (bool((writes > 0).any()),)
        if kind == "read":
            short = writes*passive_decay(age[mask], c.beta_short, writes)
            branch += (bool((short > 0).any()),)
    return slots, branch, tuple(tensors)


class _Query(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, *inputs):
        from .read_batch import evaluate as read
        from .full_batch import evaluate as full
        bundle, bias, requests, kind, slots, mode, zeta, nweights = plan
        ctx.set_materialize_grads(False)
        ctx.observation = observation.current()
        from .private_rows import PrivateRows
        packing = PrivateRows(inputs, nweights, len(requests))
        ctx.packing = packing
        with torch.enable_grad():
            it = iter(packing.views())
            bound = rebind(bundle, lambda _: next(it))
            rows = _rows(requests, kind, lambda _: next(it))
            weights = SimpleNamespace(kernel=bound["memory"], bias=bias)
            if kind == "read":
                results = read(bound["program"], weights, rows)
                refs = [r[name] for r in results for name in sorted(r)]
            else:
                refs = [r.value for r in full(bound["program"], weights, rows, slots, mode, zeta)]
        outputs = tuple(t.detach().clone() for t in refs)
        ctx.mark_non_differentiable(*(v for v, ref in zip(outputs, refs) if not ref.requires_grad))
        ctx.rows, ctx.weights = len(rows), nweights
        ctx.row_inputs, ctx.row_outputs = (len(inputs)-nweights)//len(rows), len(refs)//len(rows)
        ctx.save_for_backward(*inputs, *packing.leaves, *refs)
        packing.leaves = []  # SavedVariables own lifecycle/version checks.
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        with observation.resume(ctx.observation):
            return _Query._backward(ctx, *bars)

    @staticmethod
    def _backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("weighted queries support first-order VJP only")
        count = ctx.weights+ctx.rows*ctx.row_inputs
        saved = ctx.saved_tensors
        packed_count = ctx.weights+ctx.packing.fields
        leaves, refs = saved[count:count+packed_count], saved[count+packed_count:]
        groups, result = {}, [None]*count
        for row in range(ctx.rows):
            mask = tuple(j for j in range(ctx.row_outputs) if bars[row*ctx.row_outputs+j] is not None)
            if mask:
                groups.setdefault(mask, []).append(row)
        for mask, rows in groups.items():
            ids = [i for i in range(ctx.weights) if leaves[i].requires_grad]
            ids += [i for i in range(ctx.weights, packed_count) if leaves[i].requires_grad]
            if not ids:
                continue
            roots = [r*ctx.row_outputs+j for r in rows for j in mask]
            from .query_roots import coalesce
            outputs, cotangents = coalesce([refs[i] for i in roots], [bars[i] for i in roots])
            grads = torch.autograd.grad(outputs, [leaves[i] for i in ids],
                                       cotangents, retain_graph=True, allow_unused=True)
            for i, grad in zip(ids, grads):
                ctx.packing.scatter(i, rows, grad, result)
        return None, *result


def evaluate(program, weights, requests, *, kind="read", slots=0, mode="hard", zeta=1.):
    if kind not in {"read", "full"}:
        raise ValueError("unknown weighted query kind")
    groups, results = {}, [None]*len(requests)
    bundle = torch.nn.ModuleDict(dict(memory=weights.kernel, program=program))
    from .query_execution import dense, packed, branches
    if packed(program):
        from .query_packed import branches
    branch = branches(weights.kernel, weights, requests, kind) if dense(program) else [None]*len(requests)
    for i, r in enumerate(requests):
        groups.setdefault(_signature(weights.kernel, weights, r, kind, branch[i]), []).append(i)
    for ids in groups.values():
        if len(ids) == 1:
            from .read_batch import evaluate as read
            from .full_batch import evaluate as full
            from .singleton_vjp import boundary
            originals = list(bundle.parameters())
            def keep(value):
                originals.append(value)
                return value
            rows = _rows([requests[ids[0]]], kind, keep)
            if kind == "read":
                row = read(program, weights, rows)[0]
                names = sorted(row)
                results[ids[0]] = dict(zip(names, boundary(originals, [row[n] for n in names])))
            else:
                row = full(program, weights, rows, slots, mode, zeta)[0]
                value, = boundary(originals, [row.value])
                results[ids[0]] = FullResult(value, {slot: value for slot in range(slots)})
            continue
        inputs = []
        def capture(value):
            inputs.append(value)
            return value
        bound = rebind(bundle, capture)
        nweights = len(inputs)
        rows = _rows([requests[i] for i in ids], kind, capture)
        values = iter(_Query.apply((bound, weights.bias, rows, kind, slots, mode, zeta, nweights), *inputs))
        for i in ids:
            if kind == "read":
                results[i] = {name: next(values) for name in sorted(program.field_layout)}
            else:
                value = next(values)
                results[i] = FullResult(value, {slot: value for slot in range(slots)})
    return results
