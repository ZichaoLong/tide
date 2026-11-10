"""Allocation §2–6: independent frame/control roots over actual regional batches.

History is the weighted singleton; every candidate field stays an independent
input root. Parameter aliases are rebound once and receive all used contributions.
"""
from dataclasses import replace
import torch
from .parameter_tree import rebind
from ..autograd_census import register_private_graph


def fields(requests, transform):
    return [replace(r, candidates=tuple((v, {k: transform(t) for k, t in sorted(d.items())})
                                        for v, d in r.candidates)) for r in requests]


class _Region(torch.autograd.Function):
    @staticmethod
    def forward(ctx, program, requests, weights, *inputs):
        from .region_batch import evaluate
        ctx.set_materialize_grads(False)
        from .private_rows import PrivateRows
        ctx.packing = PrivateRows(inputs, weights, len(requests))
        ctx.dependencies = [None]*len(requests)
        with torch.enable_grad():
            iterator = iter(ctx.packing.views())
            bound = rebind(program, lambda _: next(iterator))
            rows = fields(requests, lambda _: next(iterator))
            results = evaluate(bound, rows, dependencies=ctx.dependencies)
        refs = [result.controls[v] for result, r in zip(results, rows) for v, _ in r.candidates]
        register_private_graph("region", refs)
        from .independent_copies import detached_copies
        outputs = detached_copies(refs)
        ctx.mark_non_differentiable(*(t for t, ref in zip(outputs, refs) if not ref.requires_grad))
        ctx.weights, ctx.rows = weights, len(rows)
        ctx.row_inputs = (len(inputs)-weights)//len(rows)
        ctx.row_outputs = len(refs)//len(rows)
        ctx.save_for_backward(*inputs, *ctx.packing.leaves, *refs)
        ctx.packing.leaves = []
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("batched regional execution supports first-order VJP only")
        count = ctx.weights+ctx.rows*ctx.row_inputs
        saved = ctx.saved_tensors
        packed_count = ctx.weights+ctx.packing.fields
        leaves, refs = saved[count:count+packed_count], saved[count+packed_count:]
        groups, result = {}, [None]*count
        for row in range(ctx.rows):
            mask = tuple(j for j in range(ctx.row_outputs) if bars[row*ctx.row_outputs+j] is not None)
            if mask:
                groups.setdefault((mask, ctx.dependencies[row]), []).append(row)
        for (mask, _), rows in groups.items():
            ids = [i for i in range(ctx.weights) if leaves[i].requires_grad]
            ids += [i for i in range(ctx.weights, packed_count) if leaves[i].requires_grad]
            if not ids:
                continue
            used = [r*ctx.row_outputs+j for r in rows for j in mask]
            gradients = torch.autograd.grad([refs[i] for i in used], [leaves[i] for i in ids],
                [bars[i] for i in used], retain_graph=True, allow_unused=True)
            for i, gradient in zip(ids, gradients):
                ctx.packing.scatter(i, rows, gradient, result)
        return None, None, None, *result


def evaluate(program, requests):
    from ..history import History
    from ..region import Selection
    groups, results = {}, [None]*len(requests)
    for i, r in enumerate(requests):
        metadata = []
        fields([r], lambda t: metadata.append((t.shape, t.dtype, t.device, t.requires_grad)))
        layout = tuple(tuple(sorted(d)) for _, d in r.candidates)
        groups.setdefault((layout, tuple(metadata)), []).append(i)
    for ids in groups.values():
        inputs = []
        def capture(t):
            inputs.append(t)
            return t
        bound = rebind(program, capture)
        weights = len(inputs)
        rows = fields([requests[i] for i in ids], capture)
        values = iter(_Region.apply(bound, rows, weights, *inputs))
        mappings = [{v: next(values) for v, _ in requests[i].candidates} for i in ids]
        flags = None
        if len(ids) > 1 and next(iter(mappings[0].values())).device.type != "cpu":
            from ..region_metadata import positive
            flags = iter(positive([value for controls in mappings for value in controls.values()]))
        for i, controls in zip(ids, mappings):
            active = {v for v, c in controls.items() if (next(flags) if flags is not None else c.detach() > 0)}
            results[i] = Selection(active, controls, History(last_time=requests[i].time))
    return results
