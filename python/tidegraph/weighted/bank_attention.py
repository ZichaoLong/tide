"""KV §4: matrix attention over shared numeric banks, independent snapshot VJP.

No query-by-history-by-width tensor is materialized in forward. Backward still
returns the required independent snapshot adjoints. Only contiguous overlapping
storage and position-free Full are eligible; generic packed attention is fallback.
"""
from collections import defaultdict
import math
import torch
from . import packing_observation as observation


class Plan:
    def __init__(self, keys, values):
        self.width = keys[0].shape[-1]
        self.lengths = [len(t) for t in keys]
        groups = defaultdict(list)
        for i, (k, v) in enumerate(zip(keys, values)):
            a, b = k.storage_offset()//self.width, v.storage_offset()//self.width
            groups[k.untyped_storage()._cdata, v.untyped_storage()._cdata, b-a].append((i, a, b))
        self.groups = []
        self.starts = [0]*len(keys)
        self.inverse = [0]*len(keys)
        self.queries = max(map(len, groups.values()))
        self.length = 1
        self.unique = 0
        for group, members in enumerate(groups.values()):
            start = min(a for _, a, _ in members)
            length = max(a+self.lengths[i] for i, a, _ in members)-start
            self.length = max(self.length, length)
            self.unique += length
            i, a, b = members[0]
            self.groups.append((i, start, start+b-a, length))
            for j, (i, a, _) in enumerate(members):
                self.starts[i] = a-start
                self.inverse[i] = group*self.queries+j
        self.valid = (self.unique < sum(self.lengths)
                      and len(self.groups)*self.queries*self.length <= 2*len(keys)*max(1, max(self.lengths)))
        if not self.valid:
            return
        row_ids = []
        for members in groups.values():
            row_ids.extend([i for i, _, _ in members]+[len(keys)]*(self.queries-len(members)))
        opts = dict(dtype=torch.int64, device=keys[0].device)
        self.rows = torch.tensor(row_ids, **opts).reshape(-1, self.queries)
        self.inverse_tensor = torch.tensor(self.inverse, **opts)

    def bank(self, tensors, value=False):
        parts = []
        for i, key_start, value_start, length in self.groups:
            start = value_start if value else key_start
            part = tensors[i].as_strided((length, self.width), (self.width, 1), start*self.width)
            parts.append(part if self.length == length else
                         torch.cat((part, part.new_zeros((self.length-length, self.width)))))
        return parts[0].unsqueeze(0) if len(parts) == 1 else torch.stack(parts)

    def group(self, tensor):
        return torch.cat((tensor, torch.zeros_like(tensor[:1]))).index_select(0, self.rows.flatten()).reshape(
            *self.rows.shape, *tensor.shape[1:])

    def ungroup(self, tensor):
        return tensor.flatten(0, 1).index_select(0, self.inverse_tensor)

    def weight_index(self, width):
        start = self.rows.new_tensor([*self.starts, 0])[self.rows]
        length = self.rows.new_tensor([*self.lengths, 0])[self.rows]
        col = torch.arange(self.length, device=self.rows.device)[None, None]-start[..., None]
        present = (col >= 0) & (col < length[..., None])
        return torch.where(present, self.rows[..., None]*width+col, len(self.lengths)*width)


def plan(keys, values):
    if len(keys) < 2 or len(keys) != len(values) or keys[0].ndim != 2:
        return None
    ref = keys[0]
    if not ref.shape[-1] or ref.dtype not in (torch.float32, torch.float64):
        return None
    for k, v in zip(keys, values):
        if k.shape != v.shape:
            return None
        for t in (k, v):
            if (t.ndim != 2 or t.shape[-1] != ref.shape[-1] or t.device != ref.device or t.dtype != ref.dtype
                    or not t.is_contiguous() or t.is_conj() or t.is_neg() or t.storage_offset()%ref.shape[-1]):
                return None
    result = Plan(keys, values)
    # Bound padding and work as well as checking reuse. This is capability/layout
    # dispatch, independent of graph names, values or an execution prepass.
    if not result.valid:
        return None
    return result


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, query, weights, *snapshots):
        ctx.set_materialize_grads(False)
        n, heads, d = query.shape
        keys, values = snapshots[:n], snapshots[n:]
        q = p.group(query).permute(0, 2, 1, 3)
        k, v = (p.bank(rows, value).reshape(-1, p.length, heads, d).transpose(1, 2)
                for rows, value in ((keys, False), (values, True)))
        index = p.weight_index(weights.shape[1])
        w = torch.cat((weights.flatten(), weights.new_zeros(1)))[index]
        positive = w > 0
        # Ignore unused bank values before matmul, including expired huge fields.
        used = positive.any(1)[:, None, :, None]
        k, v = (torch.where(used, t, 0) for t in (k, v))
        safe = torch.where(positive, w, 1)
        logits = (q@k.transpose(-1, -2))/math.sqrt(d)+safe.log()[:, None]
        logits = logits.masked_fill(~positive[:, None], -torch.inf)
        active = positive.any(-1)[:, None, :, None]
        probabilities = torch.where(active, logits, 0).softmax(-1)
        probabilities = torch.where(active, probabilities, 0)
        output = p.ungroup((probabilities@v).transpose(1, 2))
        ctx.plan, ctx.index = p, index
        ctx.save_for_backward(query, weights, *snapshots, q, k, v, w, probabilities)
        return output

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("KV bank attention supports first-order VJP only")
        saved = ctx.saved_tensors  # Also checks original row version counters.
        query, weights = saved[:2]
        n, heads, d = query.shape
        snapshots = saved[2:2+2*n]
        if bar is None:
            return (None, None, None, *([None]*(2*n)))
        q, k, v, w, prob = saved[2+2*n:]
        p = ctx.plan
        grad = p.group(bar).transpose(1, 2)
        positive = w > 0
        first = positive.to(torch.int64).argmax(-1)[:, None, :, None, None]
        expanded = v[:, :, None].expand(-1, -1, p.queries, -1, -1)
        anchor = expanded.gather(-2, first.expand(-1, heads, -1, 1, d))
        delta = torch.where(positive[:, None, :, :, None], expanded, anchor)-anchor
        centered = delta-(prob[..., None]*delta).sum(-2, keepdim=True)
        dl = prob*(centered*grad[..., None, :]).sum(-1)
        dq = p.ungroup(((dl@k)/math.sqrt(d)).transpose(1, 2))
        dw = dl.sum(1)/torch.where(positive, w, 1)
        # Indices are unique for each real row. No floating atomic accumulation.
        row_dw = p.ungroup(dw)
        row_dk = p.ungroup((dl[..., None]*q[..., None, :]/math.sqrt(d)).transpose(1, 2)).flatten(-2)
        row_dv = p.ungroup((prob[..., None]*grad[..., None, :]).transpose(1, 2)).flatten(-2)
        # [N,H,K,D] -> [N,K,H*D], before slicing original snapshots.
        row_dk = row_dk.reshape(n, heads, p.length, d).transpose(1, 2).reshape(n, p.length, heads*d)
        row_dv = row_dv.reshape(n, heads, p.length, d).transpose(1, 2).reshape(n, p.length, heads*d)
        weight_rows, key_rows, value_rows = [], [], []
        for i, (start, length) in enumerate(zip(p.starts, p.lengths)):
            weight_rows.append(torch.cat((row_dw[i, start:start+length], weights.new_zeros(weights.shape[1]-length))))
            key_rows.append(row_dk[i, start:start+length] if snapshots[i].requires_grad else None)
            value_rows.append(row_dv[i, start:start+length] if snapshots[n+i].requires_grad else None)
        return (None, dq if query.requires_grad else None, torch.stack(weight_rows) if weights.requires_grad else None,
                *key_rows, *value_rows)


def full(queries, keys, values, weights):
    p = plan(keys, values)
    if p is None:
        return None
    if not bool((weights > 0).any(-1).all()):
        return None  # Keep structurally empty attention disconnected.
    observation.projection("kv_bank_attention", len(keys))
    return _Attention.apply(p, queries, weights, *keys, *values)
