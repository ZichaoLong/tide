"""Architecture §2.2: source MLP columns, exact aliases and independent VJPs.

The cache lives for one regional batch. Equal values/classes are not cache keys;
every physical atom still participates in aggregation. Packing is outside the
autograd graph, with original versions and unused output support retained.
"""
import math
import torch
from .packing_observation import bucket
from .column_sharing import ordered_uses


def parameters(transform):
    names = ("a", "b", "embedding")
    if hasattr(transform, "a_left"):
        names += ("a_left", "a_right", "b_left", "b_right")
    # Module.parameters() removes aliases; these are ordered mathematical slots.
    return tuple(getattr(transform, name) for name in names)


def signature(value, weights, alpha, epsilon):
    return id(value), tuple(map(id, weights)), alpha, epsilon


def workspace_bytes(value, weights):
    rows, width = value.shape
    hidden = weights[0].shape[0]
    return (16*rows*(width+hidden+2)+12*sum(p.numel() for p in weights))*max(4, value.element_size())


class _Sources(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, epsilon, arity, *inputs):
        ctx.set_materialize_grads(False)
        count = len(inputs)//arity
        columns = [inputs[j::arity] for j in range(arity)]
        packed = [torch.stack(c) for c in columns]
        z, a, b, e = packed[:4]
        if arity == 8:
            a = a+torch.bmm(packed[4], packed[5])
            b = b+torch.bmm(packed[6], packed[7])
        # A/B aliases use shared GEMMs; embedding slots remain independent.
        ctx.shared_a = all(columns[1][i] is columns[1][0] and
            (arity == 4 or all(columns[j][i] is columns[j][0] for j in (4, 5))) for i in range(count))
        ctx.shared_b = all(columns[2][i] is columns[2][0] and
            (arity == 4 or all(columns[j][i] is columns[j][0] for j in (6, 7))) for i in range(count))
        operand = z.float() if z.dtype == torch.float16 else z
        denominator = torch.sqrt(epsilon+operand.square().mean(-1, keepdim=True))
        normalized = operand/denominator
        u = normalized.to(z.dtype)
        hidden = (u@a[0].t() if ctx.shared_a else torch.bmm(u, a.transpose(1, 2)))+e[:, None]
        activated = torch.nn.functional.silu(hidden)
        values = z+alpha*(activated@b[0].t() if ctx.shared_b else torch.bmm(activated, b.transpose(1, 2)))
        ctx.arity, ctx.count, ctx.alpha = arity, len(inputs), alpha
        ctx.save_for_backward(*inputs, a, b, u, normalized, denominator, hidden, activated, *packed[4:])
        outputs = tuple(values.unbind())
        ctx.mark_non_differentiable(*(y for i, y in enumerate(outputs)
                                     if not any(t.requires_grad for t in inputs[i*arity:(i+1)*arity])))
        bucket("source_columns", z.shape[1], count)
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("source columns support first-order VJP only")
        saved = ctx.saved_tensors  # All original versions, including aliases/frozen roots.
        result = [None]*ctx.count
        ids = [i for i, bar in enumerate(bars) if bar is not None]
        if not ids:
            return None, None, None, *result
        numeric = saved[ctx.count:]
        index = None if len(ids) == len(bars) else torch.tensor(ids, device=numeric[0].device)
        take = lambda t: t if index is None else t.index_select(0, index)
        a, b, u, normalized, denominator, hidden, activated, *factors = [take(t) for t in numeric]
        g = torch.stack([bars[i] for i in ids]); gv = g*ctx.alpha
        gh = gv@b[0] if ctx.shared_b else torch.bmm(gv, b)
        ht = torch.ops.aten.silu_backward(gh, hidden)
        db = torch.bmm(gv.transpose(1, 2), activated)
        da = torch.bmm(ht.transpose(1, 2), u)
        gu = ht@a[0] if ctx.shared_a else torch.bmm(ht, a)
        if g.dtype == torch.float16:
            gu = gu.float()
        dz = g+((gu-normalized*(gu*normalized).mean(-1, keepdim=True))/denominator).to(g.dtype)
        gradients = [dz, da, db, ht.sum(1)]
        if ctx.arity == 8:
            al, ar, bl, br = factors
            gradients += [torch.bmm(da, ar.transpose(1, 2)), torch.bmm(al.transpose(1, 2), da),
                          torch.bmm(db, br.transpose(1, 2)), torch.bmm(bl.transpose(1, 2), db)]
        for j, gradient in enumerate(gradients):
            for k, i in enumerate(ids):
                if ctx.needs_input_grad[3+i*ctx.arity+j]:
                    result[i*ctx.arity+j] = gradient[k]
        return None, None, None, *result


def source_columns(requests, *, max_workspace_bytes):
    """(content, ordered A/B/e/factors, alpha, epsilon) → one tensor per use.

    Group equal actual shapes/dtype/device/scalars; independent shapes use other
    groups. The byte ceiling bounds additional batch packing and adjoints. A
    single oversized column uses its ordinary formula, without extra packing.
    """
    if type(max_workspace_bytes) is not int or not 0 < max_workspace_bytes < 2**63:
        raise ValueError("source column workspace must be a positive int64 byte count")
    unique, uses, groups = {}, [], {}
    for request in requests:
        z, weights, alpha, epsilon = request
        if z.ndim != 2 or min(z.shape) < 1 or not math.isfinite(alpha) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("invalid source column content/scalars")
        key = signature(*request)
        if key not in unique:
            unique[key] = [request, []]
        unique[key][1].append(len(uses)); uses.append(None)
    result = {}
    for key, ((z, weights, alpha, epsilon), _) in unique.items():
        if not alpha:
            result[key] = z
            continue
        shape = (tuple(z.shape), tuple(tuple(p.shape) for p in weights), z.dtype, z.device, alpha, epsilon)
        if z.ndim != 2 or not z.shape[0] or len(weights) not in (3, 7):
            raise ValueError("invalid source column shape/parameter slots")
        d, h = z.shape[1], weights[0].shape[0]
        expected = [(h, d), (d, h), (h,)]
        if len(weights) == 7:
            rank = weights[3].shape[-1]; expected += [(h, rank), (rank, d), (d, rank), (rank, h)]
        if any(tuple(p.shape) != s or (p.device, p.dtype) != (z.device, z.dtype) for p, s in zip(weights, expected)):
            raise ValueError("source column parameter shape/dtype/device mismatch")
        groups.setdefault(shape, []).append(key)
    for keys in groups.values():
        pending, used = [], 0
        def flush():
            nonlocal pending, used
            if not pending:
                return
            rows = [unique[k][0] for k in pending]
            values = _Sources.apply(rows[0][2], rows[0][3], 1+len(rows[0][1]),
                                    *(t for z, weights, _, _ in rows for t in (z, *weights)))
            result.update(zip(pending, values)); pending, used = [], 0
        for key in keys:
            z, weights, alpha, epsilon = unique[key][0]
            size = workspace_bytes(z, weights)
            if pending and used+size > max_workspace_bytes:
                flush()
            if size > max_workspace_bytes:
                from .normalization import norm
                a, b, e = weights[:3]
                if len(weights) == 7:
                    a = a+weights[3]@weights[4]; b = b+weights[5]@weights[6]
                result[key] = z+alpha*(torch.nn.functional.silu(norm(z, epsilon)@a.t()+e)@b.t())
                bucket("source_columns_single_fallback", len(z), 1)
            else:
                pending.append(key); used += size
        flush()
    for key, (_, indices) in unique.items():
        values = ordered_uses(result[key], len(indices))
        if len(indices) > 1:
            bucket("source_exact_reuse", result[key].shape[0], len(indices))
        for i, value in zip(indices, values):
            uses[i] = value
    return uses
