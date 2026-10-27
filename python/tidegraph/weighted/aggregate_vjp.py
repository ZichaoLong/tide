"""Architecture §2: batched first-order Aggregate with independent fiber roots.

Groups share class order, zero-total branch and input gradient metadata. The
private graph is an actual batch, never scalar replay. Parameters keep aliases;
only fibers with defined cotangents contribute input gradients.
"""
from dataclasses import replace
import torch
from ..aggregate import AggregateResult
from .aggregate_batch import evaluate as numeric
from .parameter_tree import rebind


def _requests(requests, transform):
    return [replace(r, sources=tuple(replace(s, atom=replace(s.atom, value=transform(s.atom.value)))
                                     for s in r.sources)) for r in requests]


def _signature(program, request):
    values = [s.atom.value for s in request.sources]
    zero = not bool(torch.stack([v[0].detach().clamp_min(0) for v in values]).sum() > 0)
    labels = tuple(program.classes.classify(s.atom) for s in request.sources) if program.alpha else ()
    return zero, labels, tuple((v.shape, v.device, v.dtype, v.requires_grad) for v in values)


def _mass_only(requests):
    # Scalar zero-total branch does not use source transforms or content.
    payload = torch.stack([s.atom.value for r in requests for s in r.sources])
    total = payload[:, 0].clamp_min(0).reshape(len(requests), -1).sum(1)
    values = torch.cat((total[:, None], payload.new_zeros(len(requests), payload.shape[-1]-1)), -1)
    return [AggregateResult(row, {}) for row in values.unbind()]


class _Aggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, *inputs):
        ctx.set_materialize_grads(False)
        leaves = [t.detach().requires_grad_(t.requires_grad) for t in inputs]
        it = iter(leaves)
        program = rebind(plan[0], lambda _: next(it))
        requests = _requests(plan[1], lambda _: next(it))
        with torch.enable_grad():
            results = _mass_only(requests) if plan[2] else numeric(program, requests)
        refs = [r.value for r in results]
        outputs = tuple(t.detach().clone() for t in refs)
        ctx.mark_non_differentiable(*(v for v, ref in zip(outputs, refs) if not ref.requires_grad))
        ctx.rows, ctx.sources = len(requests), len(requests[0].sources)
        ctx.weights = len(inputs)-ctx.rows*ctx.sources
        ctx.save_for_backward(*inputs, *leaves, *refs)
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("weighted Aggregate supports first-order VJP only")
        count = ctx.weights+ctx.rows*ctx.sources
        saved = ctx.saved_tensors  # Checks versions of the original parameters/atoms.
        leaves, refs = saved[count:2*count], saved[2*count:]
        rows = [i for i, bar in enumerate(bars) if bar is not None]
        ids = [i for i in range(ctx.weights) if leaves[i].requires_grad]
        ids += [ctx.weights+i*ctx.sources+j for i in rows for j in range(ctx.sources)
                if leaves[ctx.weights+i*ctx.sources+j].requires_grad]
        result = [None]*count
        if ids and rows:
            grads = torch.autograd.grad([refs[i] for i in rows], [leaves[i] for i in ids],
                                       [bars[i] for i in rows], retain_graph=True, allow_unused=True)
            for i, grad in zip(ids, grads):
                result[i] = grad
        return None, *result


def evaluate(program, weights, requests):
    groups, results = {}, [None]*len(requests)
    for i, request in enumerate(requests):
        if not request.sources:
            raise ValueError("weighted Aggregate requires nonempty fibers")
        groups.setdefault(_signature(program, request), []).append(i)
    for signature, ids in groups.items():
        inputs = []
        def capture(value):
            inputs.append(value)
            return value
        plan = (rebind(program, capture), _requests([requests[i] for i in ids], capture), signature[0])
        for i, value in zip(ids, _Aggregate.apply(plan, *inputs)):
            results[i] = AggregateResult(value, {})
    return results
