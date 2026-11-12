"""Architecture §2: unique source bank → incidence mean → block transpose.

The numerical handoff keeps source owners, integer incidence and row maps.
Independent receiver fields are projections; no receiver-expanded payload is
saved. The mass transpose retains the centered z-u expression.
"""
from types import SimpleNamespace
import torch
from .aggregate_layout import SourceIncidence, SourceFootprint
from .column_sharing import ordered_uses
from .packing_observation import bucket


def _rows(sources):
    first = sources[0][0].rows
    return first if all(torch.equal(p.rows, first) for p, _ in sources) else torch.cat(
        [p.rows for p, _ in sources]).unique(sorted=True)


class _Aggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layout, record_grad, *originals):
        ctx.set_materialize_grads(False)
        index, inverse, counts, publish, singleton = layout.tensors(originals[0].device)
        raw = torch.cat((*originals[::2], originals[0].new_zeros(1))).index_select(
            0, index.flatten()).reshape(index.shape)
        value = torch.cat((*originals[1::2], originals[1].new_zeros(1, originals[1].shape[-1]))).index_select(
            0, index.flatten()).reshape(*index.shape, originals[1].shape[-1])
        mass, incidence = raw.clamp_min(0), counts.to(raw.dtype)
        total = incidence@mass
        positive = total > 0
        denominator = torch.where(positive, total, 1)
        mean = (incidence@(mass[..., None]*value).flatten(1)).reshape(*total.shape, value.shape[-1])
        mean = torch.where(positive[..., None], mean/denominator[..., None], 0)
        ctx.layout, ctx.count, ctx.publish = layout.metadata(), len(originals), publish
        ctx.save_for_backward(*originals, mass, value, total, mean, positive, raw >= 0, incidence, inverse, singleton)
        record_grad = record_grad and any(x.requires_grad for x in originals)
        live = positive.any(-1).cpu().tolist() if record_grad else [False]*len(layout.receivers)
        outputs, frozen = [], []
        for g, ids in enumerate(publish):
            m, u = total[g], mean[g]
            if len(ids) != len(layout.rows):
                m, u = m.index_select(0, ids), u.index_select(0, ids)
            slots = [s for s, count in enumerate(layout.incidence[g]) if count]
            if not any(originals[2*s].requires_grad for s in slots):
                frozen.append(m)
            if not live[g] or not any(originals[2*s+k].requires_grad for s in slots for k in (0, 1)):
                frozen.append(u)
            outputs.extend((m, u))
        ctx.mark_non_differentiable(*frozen)
        bucket("aggregate_tensor", len(layout.rows)*len(layout.sources), len(layout.receivers))
        bucket("aggregate_source_bank", len(layout.rows), len(layout.sources))
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("aggregate tensor supports first-order VJP only")
        saved = ctx.saved_tensors
        originals = saved[:ctx.count]
        mass, value, total, mean, positive, nonnegative, incidence, inverse, singleton = saved[ctx.count:]
        gm, gu = torch.zeros_like(total), torch.zeros_like(mean)
        used_mean, used_mass = [], []
        for g, ids in enumerate(ctx.publish):
            if bars[2*g] is not None:
                gm[g].index_copy_(0, ids, bars[2*g])
            if bars[2*g+1] is not None:
                gu[g].index_copy_(0, ids, bars[2*g+1])
            used_mass.append(bars[2*g] is not None)
            used_mean.append(bars[2*g+1] is not None)
        dm = incidence.t()@gm
        dv = None
        if any(used_mean):
            denominator = torch.where(positive, total, 1)
            gu = torch.where(positive[..., None], gu, 0)
            h = gu/denominator[..., None]
            dv = (incidence.t()@h.flatten(1)).reshape_as(value)*mass[..., None]
            # One receiver tile at a time: scratch is S x R x D, independent
            # of G. Center before dotting; two large quotient terms would lose
            # the established stable mass derivative. Never loop over B x L.
            for g, used in enumerate(used_mean):
                if not used or ctx.layout.physical_counts[g] == 1:
                    continue
                live = (incidence[g] > 0)[:, None] & (positive[g] & ~singleton[g])[None]
                delta = torch.where(live[..., None], value-mean[g], 0)
                dm = dm+incidence[g, :, None]*(delta*gu[g]).sum(-1)/denominator[g]
        dm = (dm*nonnegative).flatten().index_select(0, inverse).split(ctx.layout.lengths)
        if dv is not None:
            dv = dv.flatten(0, 1).index_select(0, inverse).split(ctx.layout.lengths)
        result = []
        for s in range(len(ctx.layout.sources)):
            owners = [g for g, row in enumerate(ctx.layout.incidence) if row[s]]
            m, v = originals[2*s:2*s+2]
            result.append(dm[s] if m.requires_grad and any(used_mass[g] or used_mean[g] for g in owners) else None)
            result.append(dv[s] if v.requires_grad and any(used_mean[g] for g in owners) else None)
        return None, None, *result


def assemble_many(entries, budget, fallback):
    """Physical uses → unique source incidence → distinct receiver roots.

    Half, singleton and over-budget groups use the existing scalar assembly.
    Budget splits create stable group use roots, never per-physical-use roots
    on the normal source-bank path. Tensor identities remain distinct across
    equal-valued independent inputs.
    """
    result, groups, tasks = [None]*len(entries), {}, []
    for i, (sources, values) in enumerate(entries):
        z = values[0]
        entry = sources, values, _rows(sources)
        if z.dtype not in (torch.float32, torch.float64):
            tasks.append(([(i, entry)], None))
        else:
            groups.setdefault((z.device, z.dtype, z.shape[-1]), []).append((i, entry))
    for group in groups.values():
        pending, footprint = [], SourceFootprint()

        def flush():
            if pending:
                plan = SourceIncidence.build([entry for _, entry in pending]) if len(pending) > 1 else None
                tasks.append((list(pending), plan))
                pending.clear()

        for item in group:
            footprint.add(item[1])
            if pending and footprint.bytes() > budget:
                flush()
                footprint = SourceFootprint()
                footprint.add(item[1])
            if footprint.bytes() > budget:
                tasks.append(([item], None))
                footprint = SourceFootprint()
            else:
                pending.append(item)
        flush()
    # A source shared across bounded numerical groups has one ordered reverse
    # boundary. Within a group it is passed exactly once, with all edge counts.
    roots, mappings = {}, [{} for _ in tasks]
    for t, (items, _) in enumerate(tasks):
        for _, (sources, values, _) in items:
            for (p, _), z in zip(sources, values):
                for value in (p.strength, z):
                    root, users = roots.setdefault(id(value), (value, []))
                    if not users or users[-1] != t:
                        users.append(t)
    for key, (value, users) in roots.items():
        for task, view in zip(users, ordered_uses(value, len(users))):
            mappings[task][key] = view
    for (items, plan), mapped in zip(tasks, mappings):
        if plan is not None:
            out = _Aggregate.apply(plan, torch.is_grad_enabled(), *(mapped[id(t)] for t in plan.originals))
            values = [(entry[2], out[2*g], out[2*g+1]) for g, (_, entry) in enumerate(items)]
        else:
            sources, transformed, _ = items[0][1]
            borrowed = [(SimpleNamespace(rows=p.rows, strength=mapped[id(p.strength)], content=p.content), c) for p, c in sources]
            values = [fallback(borrowed, [mapped[id(t)] for t in transformed])]
        for (i, _), value in zip(items, values):
            result[i] = value
    return result
