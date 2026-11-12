"""Ragged metadata and byte admission for node×B×Q KV tensor blocks."""
from dataclasses import dataclass
from functools import cached_property
import torch
from .column_position_ops import rebase_positions
from .tensor_kernel import mapped_role, Role


def shape(layout):
    return layout.samples, max((z-a for a, z in layout.spans), default=1) or 1


def bucket_width(width):
    return 1 << (width-1).bit_length()


def padded(layout, width):
    ids = torch.full((layout.samples, width), len(layout.times), dtype=torch.int64)
    reverse = []
    for b, (start, end) in enumerate(layout.spans):
        ids[b, :end-start] = torch.arange(start, end)
        reverse.extend(range(b*width, b*width+end-start))
    return ids, torch.tensor(reverse, dtype=torch.int64)


@dataclass
class Query:
    node: object
    window: object
    bank: object
    rows: torch.Tensor
    before: bool = False

    @cached_property
    def layout(self):
        return self.window.layout.select(self.rows)

    def positions(self, kind):
        c = self.node.kernel.config
        frequencies = (c.read_frequencies,) if kind == "read" else c.full_frequencies
        if not any(frequencies):
            return None, "plain"
        return rebase_positions(self.layout, self.bank.layout, frequencies, c.position_scale, self.window.content.dtype)


def metadata(queries, kind):
    qwidth = max(shape(q.layout)[1] for q in queries)
    kwidth = max(shape(q.bank.layout)[1] for q in queries)
    qids, kids, reverse, stamps, times, qt, kt = [], [], [], [], [], [], []
    for q in queries:
        ids, rev = padded(q.layout, qwidth)
        bank, _ = padded(q.bank.layout, kwidth)
        qids.append(torch.cat((q.rows, torch.tensor([len(q.window.layout.times)])))[ids])
        kids.append(bank); reverse.append(rev.to(q.window.content.device))
        times.append(torch.cat((q.layout.times, torch.zeros(1, dtype=torch.int64)))[ids])
        stamps.append(torch.cat((q.bank.layout.times, torch.zeros(1, dtype=torch.int64)))[bank])
        coordinates, reason = q.positions(kind)
        if reason not in {"plain", "rebased"}:
            raise ValueError("tensor KV position domain: "+reason)
        if coordinates is not None:
            a, b = coordinates
            qt.append(torch.nn.functional.pad(a, (0, qwidth-a.shape[1])))
            kt.append(torch.nn.functional.pad(b, (0, kwidth-b.shape[1])))
    age = torch.stack(times)[..., None]-torch.stack(stamps)[:, :, None]
    visible = age > 0 if queries[0].before else age >= 0
    horizon = queries[0].node.kernel.config.window
    if horizon:
        visible &= age < horizon
    for i, q in enumerate(queries):
        visible[i] &= (qids[i] < len(q.window.layout.times))[..., None]
        visible[i] &= (kids[i] < len(q.bank.layout.times))[:, None]
    device = queries[0].window.content.device
    result = dict(age=age.to(device), visible=visible.to(device))
    if qt:
        result.update(qt=torch.stack(qt).to(device), kt=torch.stack(kt).to(device))
    return result, qids, kids, tuple(reverse)


def roles(queries, qids, kids, kind):
    result = dict(u=mapped_role([q.window.content for q in queries], qids),
                  weights=mapped_role([q.bank.weights for q in queries], kids))
    if kind == "read":
        result["mass"] = mapped_role([q.window.mass for q in queries], qids)
    names = ("summaries",) if kind == "read" and queries[0].node.kernel.config.memory_read == "sum" else ("keys", "values")
    for name in names:
        result[name] = mapped_role([getattr(q.bank, name) for q in queries], kids)
    params = []
    for q in queries:
        if kind == "read":
            row = {"read."+name: value for name, value in q.node.read_program.named_parameters(remove_duplicate=False)}
            row["kernel.summary"] = q.node.kernel.summary
            if q.node.kernel.config.memory_read == "qry":
                row.update({"kernel."+name: getattr(q.node.kernel, name) for name in ("read_q", "read_k", "read_v")})
        else:
            row = {"full."+name: value for name, value in q.node.full_program.content.named_parameters(remove_duplicate=False)}
        params.append(row)
    for name in params[0]:
        values = tuple(row[name] for row in params)
        result[name] = Role(values, shared=all(x is values[0] for x in values))
    return result
