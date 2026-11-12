"""Complete candidate-count groups with independently rooted node columns.

Allocation §2–6: numeric kernels see R×N; publication and its transpose retain
actual node incidence. Equal N never creates a dependency on a different row's
node. Parameters, packed descriptor leaves and branch support are separate.
"""
import torch
from .parameter_tree import rebind
from .regional_packing import PackedRegion
from ..autograd_census import register_private_graph


def _save(column):
    if column.allocation is None:
        return (column.mode, None), [column.controls]
    a = column.allocation
    return (column.mode, (a.rule, a.policy)), [column.controls, *a.inputs,
        a.numeric.controls, a.numeric.multiplier, a.numeric.branch,
        a.numeric.response_input, a.numeric.free, a.numeric.residual]


def _restore(metadata, tensors):
    from .regional_columns import RegionColumns
    from .allocation_columns import AllocationColumns
    from .allocation_batch import BatchAllocation
    mode, allocation = metadata
    a = None
    if allocation is not None:
        rule, policy = allocation
        a = AllocationColumns(tensors[0], BatchAllocation(*tensors[7:]), tuple(tensors[1:7]), rule, policy)
    return RegionColumns(tensors[0], a, mode)


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, template, packing, names, weights, placement, *inputs):
        from .column_placement import region
        ctx.set_materialize_grads(False)
        ctx.packing, ctx.names, ctx.weights = packing, names, weights
        ctx.originals = len(inputs)
        parameters = [t.detach().requires_grad_(t.requires_grad) for t in inputs[:weights]]
        it = iter(parameters)
        program = rebind(template, lambda _: next(it))
        sources = {v: {} for v in packing.domains}
        for (v, name), tensor in zip(names, inputs[weights:]):
            sources[v][name] = tensor
        saved, groups, values, enabled = [*inputs, *parameters], [], [], set()
        for group in packing.groups:
            packed = group.pack(sources)
            fields = {n: t.detach().requires_grad_(any(sources[v][n].requires_grad for v in group.participants))
                      for n, t in packed.fields.items()}
            with torch.enable_grad():
                column = region(program, fields, packed.nodes, *placement)
            register_private_graph("regional-groups", [column.controls])
            metadata, tape = _save(column)
            groups.append((tuple(fields), len(saved), metadata, column.dependencies()))
            saved.extend((*fields.values(), *tape))
            values.append(column.controls)
            differentiable = column.differentiable_rows()
            enabled.update(v for i, v in enumerate(group.node_ids) if differentiable[i//group.candidates])
        outputs = tuple(value.detach() for value in packing.unpack(values).values())
        ctx.mark_non_differentiable(*(x for v, x in zip(packing.domains, outputs) if v not in enabled))
        ctx.groups = groups
        ctx.save_for_backward(*saved)
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("regional node groups support first-order VJP only")
        saved = ctx.saved_tensors  # Check every original version, including unused nodes.
        parameters = saved[ctx.originals:ctx.originals+ctx.weights]
        sources, indices = {v: {} for v in ctx.packing.domains}, {}
        for i, (v, name) in enumerate(ctx.names, ctx.weights):
            sources[v][name], indices[v, name] = saved[i], i
        first = ctx.groups[0]
        prototype = saved[first[1]+len(first[0])]
        group_bars, supports = ctx.packing.controls_pullback(dict(zip(ctx.packing.domains, bars)), prototype)
        result = [None]*ctx.originals

        def accumulate(index, value):
            if value is not None:
                result[index] = value if result[index] is None else result[index]+value

        for group, info, bar, support in zip(ctx.packing.groups, ctx.groups, group_bars, supports):
            if bar is None:
                continue
            names, start, metadata, dependencies = info
            fields = dict(zip(names, saved[start:start+len(names)]))
            count = 13 if metadata[1] is not None else 1
            column = _restore(metadata, saved[start+len(names):start+len(names)+count])
            leaves = [*parameters, *fields.values()]
            ids = [i for i, t in enumerate(leaves) if t.requires_grad]
            if not ids:
                continue
            # Finite mathematical branch classes, not candidate-member sets.
            # A caps row must not connect its score root merely because another
            # row in this physical batch uses the general KKT map.
            branches = {}
            for row, (used, dependency) in enumerate(zip(support.cpu().tolist(), dependencies)):
                if used:
                    branches.setdefault(dependency, []).append(row)
            packed = PackedRegion(fields, None, group, {v: sources[v] for v in group.participants})
            for rows in branches.values():
                active = torch.zeros_like(support)
                active[torch.tensor(rows, dtype=torch.int64, device=support.device)] = True
                incoming = torch.where(active[:, None], bar, 0)
                roots = column.vjp_sources(incoming, active)
                if not roots:
                    continue
                grads = torch.autograd.grad([x for x, _ in roots], [leaves[i] for i in ids],
                    [g for _, g in roots], retain_graph=True, allow_unused=True)
                field_bars = {}
                for i, grad in zip(ids, grads):
                    if i < ctx.weights:
                        accumulate(i, grad)
                    else:
                        field_bars[names[i-ctx.weights]] = grad
                for v, fields_by_name in packed.pullback(field_bars, active).items():
                    for name, grad in fields_by_name.items():
                        accumulate(indices[v, name], grad)
        return None, None, None, None, None, *result


def region_groups(program, packing, fields, *, control, selection):
    """Node descriptor columns → node control columns over actual incidence.

    Public roots remain whole node fields. Explicit event-root adapters use the
    separate record contract. No padded candidate or reference trace is used.
    Inference skips the retained private graph and custom backward boundary.
    """
    from .column_placement import region
    if not packing.groups:
        return {}
    if (not torch.is_grad_enabled()
            or all(len(g.participants) == g.candidates for g in packing.groups)
            or not any(t.requires_grad for t in [*program.parameters(), *program.buffers(),
                       *(t for v in packing.domains for t in fields[v].values())])):
        values = []
        for group in packing.groups:
            p = group.pack(fields)
            values.append(region(program, p.fields, p.nodes, control, selection).controls)
        return packing.unpack(values)
    parameters = []
    def capture(value):
        parameters.append(value)
        return value
    template = rebind(program, capture)
    names = tuple((v, name) for v in packing.domains for name in sorted(fields[v]))
    inputs = [*parameters, *(fields[v][name] for v, name in names)]
    outputs = _Grouped.apply(template, packing, names, len(parameters), (control, selection), *inputs)
    return dict(zip(packing.domains, outputs))
