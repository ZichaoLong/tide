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
    def forward(ctx, program, requests, weights, active, *inputs):
        from .regional_columns import region_columns
        from .config import HARD
        from .allocation_failure import AllocationFailure
        ctx.set_materialize_grads(False)
        from .private_rows import PrivateRows
        ctx.packing = PrivateRows(inputs, weights, len(requests))
        with torch.enable_grad():
            iterator = iter(ctx.packing.leaves[:weights])
            bound = rebind(program, lambda _: next(iterator))
            # Each private leaf already owns an entire field column. Do not
            # recreate R row views just to stack them in the next module.
            names = sorted(requests[0].candidates[0][1])
            columns = {name: [] for name in names}
            for candidate, (_, descriptor) in enumerate(requests[0].candidates):
                for j, name in enumerate(names):
                    leaf = ctx.packing.leaves[weights+candidate*len(names)+j]
                    columns[name].append(leaf.reshape(len(requests), *descriptor[name].shape))
            columns = {name: torch.stack(values, 1) for name, values in columns.items()}
            nodes = None
            if program.config.selection_mode == "regional_total_zero" or program.config.selection_mode in HARD:
                nodes = torch.tensor([[v for v, _ in r.candidates] for r in requests],
                                     dtype=torch.int64, device=columns["w"].device)
            try:
                result = region_columns(bound, columns, nodes)
            except AllocationFailure as error:
                request = requests[error.row]
                error.time, error.nodes = request.time, [v for v, _ in request.candidates]
                error.members = list(request.layout.members)
                raise
        ctx.dependencies = result.dependencies()
        support = result.active.cpu().tolist()
        active.extend({v for (v, _), flag in zip(r.candidates, flags) if flag}
                      for r, flags in zip(requests, support))
        register_private_graph("region", [result.controls])
        differentiable = result.differentiable_rows()
        outputs = tuple(value.detach() for row in result.controls.unbind() for value in row.unbind())
        ctx.weights, ctx.rows = weights, len(requests)
        ctx.row_inputs = (len(inputs)-weights)//ctx.rows
        ctx.row_outputs = result.controls.shape[1]
        ctx.mark_non_differentiable(*(t for i, t in enumerate(outputs) if not differentiable[i//ctx.row_outputs]))
        ctx.mode, ctx.has_allocation = result.mode, result.allocation is not None
        extra = []
        if result.allocation is not None:
            a = result.allocation
            ctx.rule, ctx.policy = a.rule, a.policy
            extra = [*a.inputs, a.numeric.controls, a.numeric.multiplier, a.numeric.branch,
                     a.numeric.response_input, a.numeric.free, a.numeric.residual]
        ctx.save_for_backward(*inputs, *ctx.packing.leaves, result.controls, *extra)
        ctx.packing.leaves = []
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("batched regional execution supports first-order VJP only")
        count = ctx.weights+ctx.rows*ctx.row_inputs
        saved = ctx.saved_tensors
        packed_count = ctx.weights+ctx.packing.fields
        from .regional_columns import RegionColumns
        from .allocation_columns import AllocationColumns
        from .allocation_batch import BatchAllocation
        leaves, extra = saved[count:count+packed_count], saved[count+packed_count:]
        allocation = None
        if ctx.has_allocation:
            allocation = AllocationColumns(extra[0], BatchAllocation(*extra[7:]), tuple(extra[1:7]), ctx.rule, ctx.policy)
        column = RegionColumns(extra[0], allocation, ctx.mode)
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
            indices = torch.tensor(used, dtype=torch.int64, device=column.controls.device)
            grad = torch.zeros_like(column.controls).flatten().index_copy_(0, indices, torch.stack([bars[i] for i in used]))
            supported = torch.zeros(ctx.rows, dtype=torch.bool, device=grad.device)
            supported[rows] = True
            sources = column.vjp_sources(grad.reshape_as(column.controls), supported)
            if not sources:
                continue
            gradients = torch.autograd.grad([x for x, _ in sources], [leaves[i] for i in ids],
                [bar for _, bar in sources], retain_graph=True, allow_unused=True)
            for i, gradient in zip(ids, gradients):
                ctx.packing.scatter(i, rows, gradient, result)
        return None, None, None, None, *result


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
        active = []
        values = iter(_Region.apply(bound, rows, weights, active, *inputs))
        mappings = [{v: next(values) for v, _ in requests[i].candidates} for i in ids]
        for i, controls, support in zip(ids, mappings, active):
            results[i] = Selection(support, controls, History(last_time=requests[i].time))
    return results
