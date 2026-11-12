"""Physical-atom means as one regional tensor map and centered transpose.

The receiver/row/source axes remain tensors. Packing is outside autograd;
every original mass/value column retains its own root and version guard.
"""
import torch
from .packing_observation import bucket


def _rows(sources):
    first = sources[0][0].rows
    return first if all(torch.equal(p.rows, first) for p, _ in sources) else torch.cat(
        [p.rows for p, _ in sources]).unique(sorted=True)


def _cost(entries):
    g = len(entries)
    r = max(len(rows) for _, _, rows in entries)
    a = max(len(sources) for sources, _, _ in entries)
    d = entries[0][1][0].shape[-1]
    size = entries[0][1][0].element_size()
    # Packed fields, centered reverse temporaries and index maps; excludes the
    # already-owned source transforms. Recalculate after each proposed padding.
    return g*r*a*((18*d+18)*size+24)+g*r*(6*d+8)*size


class _Aggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, *originals):
        ctx.set_materialize_grads(False)
        index, inverse, sizes, groups, row_counts, source_counts, record_grad = metadata
        mass = torch.cat(originals[::2])
        value = torch.cat(originals[1::2])
        mass = torch.cat((mass, mass.new_zeros(1))).index_select(0, index.flatten()).reshape(index.shape).clamp_min(0)
        value = torch.cat((value, value.new_zeros(1, value.shape[-1]))).index_select(
            0, index.flatten()).reshape(*index.shape, value.shape[-1])
        total = mass.sum(-1)
        positive = total > 0
        denominator = torch.where(positive, total, 1)
        value = torch.where(positive[..., None, None], value, 0)
        mean = (mass[..., None]*value).sum(-2)/denominator[..., None]
        ctx.aggregate_layout, ctx.count = metadata, len(originals)
        ctx.save_for_backward(*originals, mass, value, total, mean, positive)
        record_grad = record_grad and any(x.requires_grad for x in originals)
        live = positive.any(-1).cpu().tolist() if record_grad else [False]*len(row_counts)
        outputs, frozen = [], []
        for g, rows in enumerate(row_counts):
            m, u = total[g, :rows], mean[g, :rows]
            slots = [j for j, owner in enumerate(groups) if owner == g]
            if not any(originals[2*j].requires_grad for j in slots):
                frozen.append(m)
            if not live[g] or not any(originals[2*j+k].requires_grad for j in slots for k in (0, 1)):
                frozen.append(u)
            outputs.extend((m, u))
        ctx.mark_non_differentiable(*frozen)
        bucket("aggregate_tensor", index.shape[1]*index.shape[2], index.shape[0])
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("aggregate tensor supports first-order VJP only")
        saved = ctx.saved_tensors
        originals = saved[:ctx.count]
        mass, value, total, mean, positive = saved[ctx.count:]
        index, inverse, sizes, groups, row_counts, source_counts, _ = ctx.aggregate_layout
        gm, gu = torch.zeros_like(total), torch.zeros_like(mean)
        used_mean = []
        for g, rows in enumerate(row_counts):
            if bars[2*g] is not None:
                gm[g, :rows].copy_(bars[2*g])
            if bars[2*g+1] is not None:
                gu[g, :rows].copy_(bars[2*g+1])
            used_mean.append(bars[2*g+1] is not None)
        if not any(used_mean):
            gradients = gm.unsqueeze(-1).expand(index.shape).flatten().index_select(0, inverse).split(sizes)
            result = []
            for j, g in enumerate(groups):
                m = originals[2*j]
                result.extend((gradients[j]*(m >= 0) if bars[2*g] is not None and m.requires_grad else None, None))
            return None, *result
        live = positive & torch.tensor(used_mean, device=total.device)[:, None]
        gu = torch.where(live[..., None], gu, 0)
        denominator = torch.where(positive, total, 1)
        delta = torch.where(live[..., None, None], value-mean.unsqueeze(-2), 0)
        dm = (delta*gu.unsqueeze(-2)).sum(-1)/denominator.unsqueeze(-1)
        singleton = torch.tensor(source_counts, device=total.device) == 1
        dm = torch.where(singleton[:, None, None], 0, dm)+gm.unsqueeze(-1)
        dv = (mass/denominator.unsqueeze(-1)).unsqueeze(-1)*gu.unsqueeze(-2)
        # Each actual atom-row has exactly one packed slot. Inverse gather,
        # rather than atomic scatter-add, preserves deterministic transpose.
        dm = dm.flatten().index_select(0, inverse).split(sizes)
        dv = dv.flatten(0, 2).index_select(0, inverse).split(sizes)
        result = []
        for j, g in enumerate(groups):
            m, v = originals[2*j:2*j+2]
            connected = bars[2*g] is not None or used_mean[g]
            result.append(dm[j]*(m >= 0) if connected and m.requires_grad else None)
            result.append(dv[j] if used_mean[g] and v.requires_grad else None)
        return None, *result


def _evaluate(entries):
    count = len(entries)
    rows_max = max(len(rows) for _, _, rows in entries)
    sources_max = max(len(sources) for sources, _, _ in entries)
    total = sum(len(p.rows) for sources, _, _ in entries for p, _ in sources)
    index = torch.full((count, rows_max, sources_max), total, dtype=torch.int64)
    originals, inverse, sizes, groups, row_counts, source_counts = [], [], [], [], [], []
    offset = 0
    for g, (sources, values, rows) in enumerate(entries):
        row_counts.append(len(rows)); source_counts.append(len(sources))
        for a, ((p, _), z) in enumerate(zip(sources, values)):
            positions = torch.searchsorted(rows, p.rows)
            index[g, positions, a] = torch.arange(len(p.rows))+offset
            inverse.append(g*rows_max*sources_max+positions*sources_max+a)
            originals.extend((p.strength, z)); sizes.append(len(p.rows)); groups.append(g)
            offset += len(p.rows)
    device = originals[0].device
    metadata = index.to(device), torch.cat(inverse).to(device), sizes, groups, row_counts, source_counts, torch.is_grad_enabled()
    out = _Aggregate.apply(metadata, *originals)
    return [(rows, out[2*g], out[2*g+1]) for g, (_, _, rows) in enumerate(entries)]


def assemble_many(entries, budget, fallback):
    """(physical sources, transformed columns) → distinct receiver columns.

    The original scalar boundary remains for half or oversized/single groups.
    Actual duplicate edges are slots, even when their parameter class matches.
    """
    result, groups = [None]*len(entries), {}
    for i, (sources, values) in enumerate(entries):
        z = values[0]
        if z.dtype not in (torch.float32, torch.float64):
            result[i] = fallback(sources, values)
            continue
        groups.setdefault((z.device, z.dtype, z.shape[-1]), []).append((i, (sources, values, _rows(sources))))
    for group in groups.values():
        pending = []
        def flush():
            nonlocal pending
            if not pending:
                return
            values = (_evaluate([entry for _, entry in pending]) if len(pending) > 1
                      else [fallback(*pending[0][1][:2])])
            for (i, _), value in zip(pending, values):
                result[i] = value
            pending = []
        for item in group:
            if pending and _cost([entry for _, entry in (*pending, item)]) > budget:
                flush()
            if _cost([item[1]]) > budget:
                result[item[0]] = fallback(*item[1][:2])
            else:
                pending.append(item)
        flush()
    return result
