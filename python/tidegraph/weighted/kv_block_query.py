"""Hidden-prefix query boundary with column leaves and owner-sized adjoints.

Public terminal snapshots keep their ordinary causal roots. The boundary owns
actual prepared-write inputs and returns None for causally future writes, even
when a matrix transpose computes a zero in that column. No scalar replay.
"""
from collections import defaultdict
from types import SimpleNamespace
import torch
from .kv_block_state import view
from .parameter_tree import rebind
from . import kv_block_math as math
from . import packing_observation as observation


def plan(program, weights, requests, kind):
    c = weights.kernel.config
    if (not getattr(program, 'kv_blocks', False) or c.read_frequencies
            or any(c.full_frequencies) or any(c.distance_slopes)):
        return None
    views, owners, owner_ids = [], [], {}
    for r in requests:
        state = r.state if kind == 'read' else r.comparison
        v = view(state)
        if v is None or (c.window and r.time-v.bank.times[0] >= c.window):
            return None
        if id(v.bank) not in owner_ids:
            owner_ids[id(v.bank)] = len(owners)
            owners.append(v.bank)
        views.append(v)
    groups = [[] for _ in owners]
    for i, v in enumerate(views):
        groups[owner_ids[id(v.bank)]].append(i)
    q, length = max(map(len, groups)), max(len(b.times) for b in owners)
    ref = requests[0].content.value
    if len(owners)*q*length*(4*c.heads+4)*ref.element_size() > 32*1024**2:
        return None
    indices, spans, times, reverse = [], [], [], [None]*len(requests)
    for b, ids in enumerate(groups):
        indices.append(ids+[len(requests)]*(q-len(ids)))
        spans.append([views[i].end for i in ids]+[0]*(q-len(ids)))
        times.append([requests[i].time for i in ids]+[0]*(q-len(ids)))
        for j, i in enumerate(ids):
            reverse[i] = b*q+j
    opt = dict(device=ref.device, dtype=torch.int64)
    ids, ends, ts = (torch.tensor(x, **opt) for x in (indices, spans, times))
    stamps = torch.tensor([list(b.times)+[0]*(length-len(b.times)) for b in owners], **opt)
    ages = (ts[..., None]-stamps[:, None]).clamp_min(0)
    visible = torch.arange(length, **opt)[None, None] < ends[..., None]
    # Hidden positive prefixes must be causal; no saturated int64 conversion.
    visible &= ts[..., None] >= stamps[:, None]
    return SimpleNamespace(owners=owners, views=views, groups=groups, indices=ids, visible=visible,
        ages=ages, reverse=torch.tensor(reverse, **opt), length=length, kind=kind,
        program=program, memory=weights.kernel, requests=requests)


def inputs_for(p):
    inputs, layout = [], []
    def add(t):
        inputs.append(t)
        return len(inputs)-1
    bundle = torch.nn.ModuleDict(dict(memory=p.memory, program=p.program))
    rebind(bundle, lambda t: (layout.append(('parameter', [add(t)], None)), t)[1])
    old_ids = []
    for b, bank in enumerate(p.owners):
        row = {}
        for name in ('keys', 'values', 'summaries', 'weights'):
            row[name] = add(bank.old.slots[name])
            layout.append(('old', [row[name]], b))
        old_ids.append(row)
    write_ids = {}
    for name in ('values', 'summaries', 'contents'):
        ids = [add(t) for bank in p.owners for t in getattr(bank, name)]
        write_ids[name] = ids
        layout.append(('writes', ids, name))
    for name in ('payload', 'control'):
        if name == 'control' and p.kind == 'read':
            continue
        ids = [add(r.content.value if name == 'payload' else r.control) for r in p.requests]
        layout.append(('query', ids, name))
    # These operands own version guards, not additional gradient contributions.
    for b in p.owners:
        for t in b.guards:
            add(t)
    for v in p.views:
        add(v.value)
        for t in v.slots.values():
            add(t)
    p.bundle, p.layout, p.old_ids, p.write_ids = bundle, layout, old_ids, write_ids
    return inputs


def compute(p, inputs, *, private):
    leaves, columns, old = [], {}, [{} for _ in p.owners]
    param, old_it = [], iter((b, name) for b in range(len(old))
                              for name in ('keys', 'values', 'summaries', 'weights'))
    for kind, ids, label in p.layout:
        raw = inputs[ids[0]] if kind in ('parameter', 'old') else torch.stack([inputs[i] for i in ids])
        leaf = raw.detach().requires_grad_(any(inputs[i].requires_grad for i in ids)) if private else raw
        leaves.append(leaf)
        if kind == 'parameter':
            param.append(leaf)
        elif kind == 'old':
            b, name = next(old_it); old[b][name] = leaf
        else:
            columns[label] = leaf
    it = iter(param)
    bundle = rebind(p.bundle, lambda _: next(it))
    raw = math.banks(old, columns['values'], columns['summaries'], columns['contents'],
                     [len(b.values) for b in p.owners], p.length)
    fields = math.evaluate(bundle['program'], bundle['memory'], columns['payload'],
                           columns.get('control'), raw, p.indices, p.visible, p.ages, p.kind)
    names = sorted(fields)
    refs = [fields[name].flatten(0, 1).index_select(0, p.reverse) for name in names]
    return leaves, refs


class _Query(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, *inputs):
        ctx.set_materialize_grads(False)
        with torch.enable_grad():
            leaves, refs = compute(p, inputs, private=True)
        from ..autograd_census import register_private_graph
        register_private_graph('query.block.'+p.kind, refs)
        outputs = tuple(row.detach() for values in zip(*(v.unbind() for v in refs)) for row in values)
        flags = [v.requires_grad for _ in p.requests for v in refs]
        ctx.mark_non_differentiable(*(x for x, flag in zip(outputs, flags) if not flag))
        ctx.plan, ctx.count, ctx.width = p, len(inputs), len(refs)
        ctx.save_for_backward(*inputs, *leaves, *refs)
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError('KV block query supports first-order VJP only')
        p = ctx.plan
        saved = ctx.saved_tensors
        inputs = saved[:ctx.count]
        leaves = saved[ctx.count:ctx.count+len(p.layout)]
        refs = saved[ctx.count+len(p.layout):]
        groups, result = defaultdict(list), [None]*len(inputs)
        for row in range(len(p.requests)):
            mask = tuple(f for f in range(ctx.width) if bars[row*ctx.width+f] is not None)
            if mask:
                groups[mask].append(row)
        for mask, rows in groups.items():
            row_set = set(rows)
            roots, gradients = [], []
            for f in mask:
                if not refs[f].requires_grad:
                    continue
                zero = torch.zeros_like(refs[f][0])
                gradients.append(torch.stack([bars[i*ctx.width+f] if i in row_set else zero for i in range(len(p.requests))]))
                roots.append(refs[f])
            active = [i for i, x in enumerate(leaves) if x.requires_grad]
            if not roots or not active:
                continue
            grads = torch.autograd.grad(roots, [leaves[i] for i in active], gradients, retain_graph=True, allow_unused=True)
            needed = {id(p.views[i].bank): max(p.views[j].end for j in rows if p.views[j].bank is p.views[i].bank) for i in rows}
            for i, grad in zip(active, grads):
                if grad is None:
                    continue
                kind, ids, label = p.layout[i]
                if kind == 'parameter':
                    chosen = [(ids[0], grad)]
                elif kind == 'old':
                    chosen = [(ids[0], grad)] if id(p.owners[label]) in needed else []
                else:
                    values = grad.unbind()
                    if kind == 'query':
                        chosen = [(ids[j], values[j]) for j in rows]
                    else:
                        chosen, offset = [], 0
                        for bank in p.owners:
                            last = needed.get(id(bank), 0)-len(bank.old.slots['times'])
                            chosen.extend((ids[offset+j], values[offset+j]) for j in range(min(len(bank.values), max(0, last))))
                            offset += len(bank.values)
                for ident, value in chosen:
                    if inputs[ident].requires_grad:
                        result[ident] = value if result[ident] is None else result[ident]+value
        return None, *result


def dispatch(program, weights, requests, *, kind='read', slots=0, mode='hard', zeta=1.):
    if not getattr(program, 'kv_blocks', False) or mode != 'hard':
        return {}
    chosen = [i for i, r in enumerate(requests) if view(r.state if kind == 'read' else r.comparison) is not None]
    if len(chosen) < 2:
        return {}
    rows = [requests[i] for i in chosen]
    p = plan(program, weights, rows, kind)
    if p is None:
        return {}
    # Keep underflowed short-channel support on the existing exact branch path.
    # No per-query scalar readback, and never repair support by clipping.
    if kind == 'read':
        from .numeric_time import passive_decay
        with torch.no_grad():
            raw = torch.stack([torch.nn.functional.pad(torch.cat((b.old.slots['weights'],
                torch.stack(b.contents)[:, 0].clamp(max=1))), (0, p.length-len(b.times))) for b in p.owners])
            w = torch.where(p.visible, raw[:, None], 0)
            short = w*passive_decay(p.ages, weights.kernel.config.beta_short, w)
            if bool(((w > 0) & (short == 0)).any()):
                return {}
    # Column leaves require homogeneous structural flags. Decline heterogeneous
    # frozen/trainable rows rather than turning a disconnected root into zero.
    if torch.is_grad_enabled():
        families = [[getattr(b, name)[j] for b in p.owners for j in range(len(b.values))]
                    for name in ('values', 'summaries', 'contents')]
        families += [[b.old.slots[name] for b in p.owners]
                     for name in ('keys', 'values', 'summaries', 'weights')]
        families += [[r.content.value for r in rows]]
        if kind == 'full':
            families += [[r.control for r in rows]]
        if any(len({t.requires_grad for t in tensors}) > 1 for tensors in families):
            return {}
    args = inputs_for(p)
    if torch.is_grad_enabled():
        values = iter(_Query.apply(p, *args))
    else:
        _, refs = compute(p, args, private=False)
        values = iter(row for fields in zip(*(v.unbind() for v in refs)) for row in fields)
    names = sorted(program.field_layout) if kind == 'read' else ['value']
    result = {}
    from ..full import FullResult
    for i in chosen:
        fields = {name: next(values) for name in names}
        result[i] = fields if kind == 'read' else FullResult(fields['value'], {j: fields['value'] for j in range(slots)})
    observation.projection('kv_block_'+kind, len(chosen))
    return result
