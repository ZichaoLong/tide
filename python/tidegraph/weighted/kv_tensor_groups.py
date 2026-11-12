"""KV node blocks chosen by capabilities and actual padded storage costs.

Only boundary input/output tensors are split by node. Query normalization,
history projections, scores, attention, heads and residuals run on whole blocks.
"""
import torch
from .kv_tensor_layout import Query, metadata, roles, shape
from .kv_tensor_formula import Formula
from .tensor_kernel import execute
from .packing_observation import bucket


def available(queries, kind):
    return all(q.window.content.dtype in {torch.float32, torch.float64}
               and q.positions(kind)[1] in {"plain", "rebased"} for q in queries)


def plan(queries, kind, budget, old_entries, old_groups):
    """One unavailable node must not disable independent compatible groups."""
    eligible = [i for i, q in enumerate(queries) if available([q], kind)]
    eligible_set = set(eligible)
    for ids in groups([queries[i] for i in eligible], kind, budget):
        yield [eligible[j] for j in ids], True
    for ids in old_groups([row for i, row in enumerate(old_entries) if i not in eligible_set], budget):
        yield ids, False


def cost(queries):
    """Bound packed operands, private first-order tape and attention workspaces.

    Recompute maxima after each proposed addition; summing unpadded per-node
    estimates would under-admit unequal sequence/history lengths.
    """
    c = queries[0].node.kernel.config
    b = queries[0].layout.samples
    q = max(shape(row.layout)[1] for row in queries)
    k = max(shape(row.bank.layout)[1] for row in queries)
    size = max(queries[0].window.content.element_size(), 4)
    matrices = sum(p.numel() for row in queries for _, p in row.node.named_parameters(remove_duplicate=False))
    pair = 32+8*c.heads
    if c.read_frequencies or any(c.full_frequencies):
        pair += 8*(c.width+c.summary_width)
    elements = (4*matrices+len(queries)*b*(q*(32*c.width+8*c.ffn_width+16*c.summary_width+128)
                +k*(20*c.width+8*c.summary_width)+q*k*pair))
    return elements*size+32*len(queries)*b*(q+k+q*k)


def groups(queries, kind, budget):
    compatible = {}
    for i, q in enumerate(queries):
        c = q.node.kernel.config
        key = (c, q.node.read_program.config if kind == "read" else None, q.before,
               q.window.content.dtype, q.window.content.device, q.layout.samples)
        compatible.setdefault(key, []).append(i)
    for ids in compatible.values():
        current = []
        for i in ids:
            if current and cost([queries[j] for j in (*current, i)]) > budget:
                yield current
                current = []
            current.append(i)
        if current:
            yield current


def evaluate(queries, kind, *, active=None):
    """A single compatible group; the caller already admitted padded bytes."""
    meta, qids, kids, reverse = metadata(queries, kind)
    inputs = roles(queries, qids, kids, kind)
    c = queries[0].node.kernel.config
    if active is None and (kind == "full" or c.memory_read == "qry"):
        with torch.no_grad():
            weight = inputs["weights"].pack()
            flags = ((weight[..., None, :] > 0) & meta["visible"]).flatten(1).any(-1).cpu().tolist()
        if any(flags) and not all(flags):
            result = [None]*len(queries)
            for flag in (False, True):
                ids = [i for i, value in enumerate(flags) if value == flag]
                for i, value in zip(ids, evaluate([queries[i] for i in ids], kind, active=flag)):
                    result[i] = value
            return result
        active = any(flags)
    formula = Formula(c, queries[0].node.read_program.config, meta, bool(active), kind)
    bucket("kv_tensor_"+kind, meta["age"].shape[2], len(queries))
    return execute(inputs, reverse, formula)


def read_queries(requests, banks):
    return [Query(n, w, bank, torch.arange(len(w.layout.times)), mode == "-")
            for (n, w, mode, _), bank in zip(requests, banks)]
