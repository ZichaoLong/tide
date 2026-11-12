"""KV §3–4: bounded Read/Full groups with private banks and node roots.

The node axis joins B only inside independent block operators. Query visibility
still comes from each bank's int64 logical times and the actual Read snapshot.
"""
import math
import torch
from torch.nn import functional as F
from .kv_columns import _queries, supported, read_kv_columns, full_kv_columns, FullColumns
from .kv_columns_state import KVWriteBank
from .column_query_ops import linear, contract
from .dense_describe_groups import columns as describe
from .numeric_time import passive_decay, time_values
from .tensors import norm, rotate_relative
from .packing_observation import bucket


def _shape(layout):
    return layout.samples, max((z-a for a, z in layout.spans), default=1, ) or 1


def query_group_bytes(node, query_layout, bank_layout, element_size):
    """Additional packing, saved operands and block adjoints; not total HBM.

Reserve both Read/Full/Describe matrices, padded queries/banks, and relative
position temporaries. Existing per-node admission remains the enclosing bound.
    """
    c = node.kernel.config
    b, q = _shape(query_layout); _, k = _shape(bank_layout)
    matrices = sum(p.numel() for _, p in node.named_parameters(remove_duplicate=False))
    per_pair = 32+8*c.heads
    if c.read_frequencies or any(c.full_frequencies):
        per_pair += 8*(c.width+c.summary_width)
    elements = (4*matrices+b*q*(32*c.width+8*c.ffn_width+16*c.summary_width+128)
                +b*k*(20*c.width+8*c.summary_width)+b*q*k*per_pair)
    return elements*max(element_size, 4)+32*b*(q+k+q*k)


def _groups(entries, budget):
    if type(budget) is not int or not 0 < budget < 2**63:
        raise ValueError("KV query group budget must be a positive int64 byte count")
    groups = {}
    for i, key, cost in entries:
        groups.setdefault(key, []).append((i, cost))
    for rows in groups.values():
        batch, used = [], 0
        for i, cost in rows:
            if batch and used+cost > budget:
                yield batch
                batch, used = [], 0
            batch.append(i); used += cost
        if batch:
            yield batch


def _read_bank(node, window, mode, initial):
    if not supported(node.kernel) or mode not in {"0", "-", "+"}:
        raise ValueError("unsupported KV query memory or Read snapshot")
    if not len(window.layout.times):
        raise ValueError("Read column group must contain a real query")
    if mode != "0":
        return window.bank
    if initial is None:
        raise ValueError("Read0 requires its explicitly bound initial state")
    initial.validate(node.kernel.config)
    if initial.layout.samples != window.layout.samples:
        raise ValueError("Read0 initial sample domain mismatch")
    return KVWriteBank.from_state(initial)


def read_kv_columns_many(requests, *, max_workspace_bytes):
    """Requests (node, window, read_mode, initial), in stable result order."""
    entries, banks, result = [], [], [None]*len(requests)
    for i, (node, window, mode, initial) in enumerate(requests):
        bank = _read_bank(node, window, mode, initial); banks.append(bank)
        key = (node.kernel.config, node.read_program.config, mode, window.content.dtype,
               window.content.device, _shape(window.layout), _shape(bank.layout))
        entries.append((i, key, query_group_bytes(node, window.layout, bank.layout, window.content.element_size())))
    for ids in _groups(entries, max_workspace_bytes):
        bucket("kv_read_nodes", _shape(requests[ids[0]][1].layout)[1], len(ids))
        if len(ids) == 1:
            i = ids[0]; n, w, m, s = requests[i]
            result[i] = read_kv_columns(n, w, read_mode=m, initial=s)
            continue
        nodes = [requests[i][0] for i in ids]; c = nodes[0].kernel.config
        names = ("summaries",) if c.memory_read == "sum" else ("keys", "values")
        queries = [_queries(requests[i][1], banks[i], torch.arange(len(requests[i][1].layout.times)),
                    before=requests[i][2] == "-", horizon=c.window, names=names) for i in ids]
        us, masses, raws, weights, ages, reverses = zip(*queries)
        short = [w*passive_decay(a.clamp_min(0), c.beta_short, u) for w, a, u in zip(weights, ages, us)]
        st, lt = [w.sum(-1) for w in short], [w.sum(-1) for w in weights]
        values = [(u.new_zeros((*u.shape[:2], c.summary_width)),)*2 for u in us]
        if c.memory_read == "sum":
            reads = [contract("matmul", [(w, raw["summaries"]) for w, raw in zip(ws, raws)])
                     for ws in (short, weights)]
            values = [(s/a.clamp_min(1)[..., None], l/b.clamp_min(1)[..., None])
                      for s, l, a, b in zip(*reads, st, lt)]
        else:
            active = torch.stack([(w.detach() > 0).any() for w in weights]).cpu().tolist()
            selected = [i for i, yes in enumerate(active) if yes]
            if selected:
                query = linear([norm(us[i], c.epsilon) for i in selected], [nodes[i].kernel.read_q for i in selected])
                keys = linear([raws[i]["keys"] for i in selected], [nodes[i].kernel.read_k for i in selected])
                vals = linear([raws[i]["values"] for i in selected], [nodes[i].kernel.read_v for i in selected])
                operands, kind = [], "attention"
                for i, q, k, v in zip(selected, query, keys, vals):
                    if c.read_frequencies:
                        displacement = -time_values(ages[i], us[i])/c.position_scale
                        rotated = rotate_relative(k[:, None].expand(-1, us[i].shape[1], -1, -1), displacement, c.read_frequencies)
                        q = q[:, :, None]
                        if q.dtype == torch.float16:
                            q, rotated = q.float(), rotated.float()
                        operands.append(((q*rotated).sum(-1)[:, None]/math.sqrt(c.summary_width), v[:, None]))
                        kind = "logits"
                    else:
                        operands.append((q[:, None], k[:, None], v[:, None]))
                sr = contract(kind, [(*x, short[i]) for i, x in zip(selected, operands)])
                lr = contract(kind, [(*x, weights[i]) for i, x in zip(selected, operands)])
                for i, s, l in zip(selected, sr, lr):
                    values[i] = (s[:, 0]*st[i].clamp_max(1)[..., None], l[:, 0]*lt[i].clamp_max(1)[..., None])
        fields = describe(nodes, masses, us, [(*v, a, b) for v, a, b in zip(values, st, lt)])
        for i, field, reverse in zip(ids, fields, reverses):
            result[i] = {name: value.flatten(0, 1).index_select(0, reverse) for name, value in field.items()}
    return result


def full_kv_columns_many(requests, *, max_workspace_bytes):
    """Requests (node, window, controls); no inactive Full or cG substitution."""
    entries, selected, result = [], {}, [None]*len(requests)
    for i, (node, window, controls) in enumerate(requests):
        if not supported(node.kernel):
            raise ValueError("unsupported KV Full memory")
        if (controls.shape != window.mass.shape or (controls.dtype, controls.device) !=
                (window.mass.dtype, window.mass.device) or not torch.isfinite(controls).all() or (controls < 0).any()):
            raise ValueError("invalid KV column controls")
        rows = (controls.detach() > 0).nonzero().flatten().cpu()
        result[i] = FullColumns(rows, controls.index_select(0, rows.to(controls.device)), window.content.new_empty((0, node.kernel.config.width)))
        if not len(rows):
            continue
        selected[i] = rows
        layout = window.layout.select(rows)
        key = (node.kernel.config, window.content.dtype, window.content.device, _shape(layout), _shape(window.bank.layout))
        entries.append((i, key, query_group_bytes(node, layout, window.bank.layout, window.content.element_size())))
    for ids in _groups(entries, max_workspace_bytes):
        bucket("kv_full_nodes", _shape(requests[ids[0]][1].layout.select(selected[ids[0]]))[1], len(ids))
        if len(ids) == 1:
            i = ids[0]; result[i] = full_kv_columns(*requests[i]); continue
        nodes = [requests[i][0] for i in ids]; c = nodes[0].kernel.config
        modules = [n.full_program.content for n in nodes]
        queries = [_queries(requests[i][1], requests[i][1].bank, selected[i], horizon=c.window) for i in ids]
        us, _, raws, weights, ages, reverses = zip(*queries)
        mixed = [torch.zeros_like(u) for u in us]
        active = torch.stack([(w.detach() > 0).any() for w in weights]).cpu().tolist()
        chosen = [i for i, yes in enumerate(active) if yes]
        if chosen:
            projected = linear([norm(us[i], c.epsilon) for i in chosen], [modules[i].query for i in chosen])
            operands, kind = [], "attention"
            for i, query in zip(chosen, projected):
                b, length, _ = us[i].shape
                q = query.reshape(b, length, c.heads, -1).transpose(1, 2)
                k, v = (raws[i][name].reshape(b, -1, c.heads, c.width//c.heads).transpose(1, 2) for name in ("keys", "values"))
                if any(c.full_frequencies) or any(c.distance_slopes):
                    displacement = -time_values(ages[i], us[i])/c.position_scale
                    logits = []
                    for h in range(c.heads):
                        key = rotate_relative(k[:, h, None].expand(-1, length, -1, -1), displacement, c.full_frequencies[h])
                        head = q[:, h, :, None]
                        if head.dtype == torch.float16:
                            head, key = head.float(), key.float()
                        logits.append((head*key).sum(-1)/math.sqrt(c.width//c.heads)+c.distance_slopes[h]*displacement)
                    operands.append((torch.stack(logits, 1), v, weights[i])); kind = "logits"
                else:
                    operands.append((q, k, v, weights[i]))
            for i, value in zip(chosen, contract(kind, operands)):
                mixed[i] = value.transpose(1, 2).reshape_as(us[i])
        out = linear([m*w.sum(-1).clamp_max(1)[..., None] for m, w in zip(mixed, weights)], [m.out for m in modules])
        xs = [u+c.alpha_att*y for u, y in zip(us, out)]
        ns = [norm(x, c.epsilon) for x in xs]
        gate, up = linear(ns, [m.gate for m in modules]), linear(ns, [m.up for m in modules])
        down = linear([F.silu(g)*u for g, u in zip(gate, up)], [m.down for m in modules])
        for i, x, y, reverse in zip(ids, xs, down, reverses):
            original = result[i]
            result[i] = FullColumns(original.rows, original.strength, (x+c.alpha_ff*y).flatten(0, 1).index_select(0, reverse))
    return result
