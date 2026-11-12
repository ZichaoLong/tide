"""Independent raw-left-state FC formulas for 0770593; not a general executor.

Dense adjacent regions, dense input positions, KV qry without rotary/distance,
Read+, cat/ind/context/cat and regional_total/quad. Full executes selected rows.
Output/terminal-state numeric projection only; no complete graph trace/None-root claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from layer_tensor_baseline import normalize
from tensor_baseline_state import times as window_times, validate as validate_left, kv_join, kv_terminal


def head(module, x):
    return module.bias+(F.silu(x@module.matrix.t()+module.hidden_bias)*module.vector).sum(-1)


def quadratic(scores, caps, target, temperature):
    """Independent finite interval enumeration, O(n²), exact branch derivative.

    Midpoints identify affine intervals; solve their equations simultaneously.
    No foundation solver, iterations or reference trajectory are consumed.
    """
    with torch.no_grad():
        s = scores-scores.amax(-1, keepdim=True)
        points = torch.cat((s-temperature*caps, s), -1).sort(-1).values
        left, right = points[..., :-1], points[..., 1:]
        mid = (left+right)/2
        free = (mid[..., None] > (s-temperature*caps)[..., None, :]) & (mid[..., None] < s[..., None, :])
        upper = mid[..., None] <= (s-temperature*caps)[..., None, :]
        count = free.sum(-1)
        lam = ((free*s[..., None, :]).sum(-1)+temperature*((upper*caps[..., None, :]).sum(-1)-target[..., None]))/count.clamp_min(1)
        valid = (count > 0) & (left < right)
        error = torch.maximum((left-lam).clamp_min(0), (lam-right).clamp_min(0)).masked_fill(~valid, torch.inf)
        if not valid.any(-1).all():
            raise ValueError("FC tensor quadratic interval outside declared interior-target domain")
        chosen = error.argmin(-1)
        index = chosen[..., None, None].expand(*chosen.shape, 1, scores.shape[-1])
        free, upper = (x.gather(-2, index).squeeze(-2) for x in (free, upper))
    # Differentiate the selected exact solution, never midpoint selection.
    centered = scores-scores.detach().amax(-1, keepdim=True)
    lam = ((centered*free).sum(-1)+temperature*((caps*upper).sum(-1)-target))/free.sum(-1)
    controls = torch.where(free, (centered-lam[..., None])/temperature, torch.where(upper, caps, 0))
    controls = torch.minimum(controls.clamp_min(0), caps)
    residual = (controls.sum(-1)-target).abs()
    tolerance = 64*torch.finfo(scores.dtype).eps*scores.shape[-1]
    if (not torch.isfinite(controls).all() or (controls < 0).any() or (controls > caps).any()
            or (residual > tolerance).any()):
        raise ArithmeticError("FC tensor allocation residual/support failure")
    return controls


def validate(binding, spec, x):
    from tidegraph.weighted import KVConfig
    g = binding.graph
    depth = len(g.regions)
    sizes = [sum(n.region == r for n in g.nodes) for r in range(depth)]
    width = sizes[0]
    edges = [(r*width+s, (r+1)*width+t, 1) for r in range(depth-1) for s in range(width) for t in range(width)]
    if (spec.graph.identity != g.identity or len(set(sizes)) != 1 or not 2 <= width <= 8 or spec.binding != "ports"
            or [(e.source, e.target, e.delay) for e in g.edges] != edges
            or g.inputs != tuple(range(width)) or g.outputs != tuple(range((depth-1)*width, depth*width))
            or list(spec.ranks) != list(range(1, depth+1))):
        raise ValueError("FC tensor anchor requires adjacent equal-width dense regions and ports")
    binding.model.validate_binding(g)
    if (x.ndim != 3 or min(x.shape) < 1 or x.shape[1] > 256
            or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all()):
        raise ValueError("FC tensor anchor needs finite FP32/FP64 B,L,d and L<=256")
    for v, (node, c) in enumerate(zip(binding.model.nodes, binding.memories)):
        r = binding.regions[g.nodes[v].region]
        if (not isinstance(c, KVConfig) or c.memory_read != "qry" or any(c.full_frequencies)
                or c.read_frequencies or any(c.distance_slopes) or c.width != x.shape[-1]
                or (node.bias.dtype, node.bias.device) != (x.dtype, x.device)
                or (r.read_mode, r.local_score, r.region_score, r.selection_mode,
                    r.region_control_mode, r.region_evidence, r.regularizer, r.support_policy)
                   != ("+", "cat", "ind", "regional_total", "context", "cat", "quad", "strict")
                or not 1 <= r.k < width or r.gamma_max < 1):
            raise ValueError("FC tensor anchor requires KV qry/cat/ind/context/cat/quad without positions")
    return depth, width


def aggregate(node, masses, contents, labels):
    agg = node.aggregate_program
    transformed = []
    for z, label in zip(contents, labels):
        if agg.alpha:
            transform = agg.transforms[str(label)]
            a, b = transform.matrices()
            z = z+agg.alpha*(F.silu(normalize(z, agg.epsilon)@a.t()+transform.embedding)@b.t())
        transformed.append(z)
    total = masses.sum(-1)
    if not (total > 0).all():
        raise ValueError("FC tensor anchor needs positive receive at every position")
    u = (masses[..., None]*torch.stack(transformed, -2)).sum(-2)/total[..., None]
    return total, u


def prepare(node, u, mass, times, left=None):
    memory, c = node.kernel, node.kernel.config
    n = normalize(u, c.epsilon)
    key, value = (n@memory.base.t()+(n@memory.adapter_v.t())@memory.adapter_u.t()).chunk(2, -1)
    summary = n@memory.summary.t()
    raw, age, mask = kv_join(c, times, key, value, summary, mass.clamp_max(1), left)
    key, value = raw["keys"], raw["values"]
    weights = raw["weights"][:, None, :]*mask
    short_weights = weights*torch.exp(-age.clamp_min(0).to(u.dtype)*c.beta_short)
    q, k, v = n@memory.read_q.t(), key@memory.read_k.t(), value@memory.read_v.t()
    logits = q@k.transpose(-1, -2)/math.sqrt(c.summary_width)
    values, totals = [], []
    for w in (short_weights, weights):
        total = w.sum(-1)
        safe = torch.where(w > 0, w, 1)
        probabilities = (logits+safe.log()).masked_fill(w <= 0, -torch.inf).softmax(-1)
        values.append((probabilities@v)*total.clamp_max(1)[..., None])
        totals.append(total)
    stat = torch.stack((mass.log1p(), totals[0].log1p(), totals[1].log1p()), -1)
    features = torch.cat((n, stat[..., :1], *values, stat[..., 1:], summary*values[0], summary*values[1]), -1)
    read = node.read_program
    recv = mass.clamp_max(1)
    fields = dict(w=recv, mu=(read.config.gamma_max*recv).clamp_max(1), score=head(read.score, features),
                  evidence=normalize(features@read.evidence_weight.t()+read.evidence_bias, c.epsilon))
    return fields, (u, raw, mask)


def full(node, prepared, controls, *, attention="manual"):
    """KV textbook §4/§6: selected queries across B and time share B K/V banks.

    Only the Attention contractions pad to the largest selected query count.
    Query/output/FFN projections use actual selected rows; historical K/V is
    never gathered once per query. There is no host loop over samples or time.
    """
    u, raw, mask = prepared
    key, value, writes = raw["keys"], raw["values"], raw["weights"]
    c, module = node.kernel.config, node.full_program.content
    batch, length, dim = u.shape
    active = controls.detach() > 0
    selected = active.nonzero()
    if not len(selected):
        return torch.zeros_like(u), active
    sample, time = selected.unbind(-1)
    queries = int(active.sum(-1).max())
    ordinal = active.cumsum(-1, dtype=torch.int64)-1
    slots = sample*queries+ordinal[sample, time]
    positions = sample*length+time
    current = u.reshape(-1, dim).index_select(0, positions)
    projected = normalize(current, c.epsilon)@module.query.t()
    q = u.new_zeros((batch*queries, dim)).index_copy(0, slots, projected)
    q = q.reshape(batch, queries, c.heads, dim//c.heads).transpose(1, 2)
    k, v = (z.reshape(batch, -1, c.heads, dim//c.heads).transpose(1, 2) for z in (key, value))
    query_times = time.new_zeros(batch*queries).index_copy(0, slots, time)
    visible = mask.index_select(0, query_times).reshape(batch, queries, -1)
    weights = writes[:, None, :]*visible
    safe = torch.where(weights > 0, weights, 1)
    bias = safe.log()[:, None].masked_fill(weights[:, None] <= 0, -torch.inf)
    # The FC domain writes a positive entry at every position, including time 0.
    # Padding therefore has a finite softmax; its outputs are discarded below.
    if attention in {"sdpa", "sdpa4d"}:
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    elif attention == "manual":
        logits = q@k.transpose(-1, -2)/math.sqrt(dim//c.heads)
        mixed = (logits+bias).softmax(-1)@v
    else:
        raise ValueError("unknown FC formula Attention operator")
    read = mixed.transpose(1, 2).reshape(batch*queries, dim)
    read = (read*weights.sum(-1).clamp_max(1).reshape(-1, 1)).index_select(0, slots)
    x = current+c.alpha_att*(read@module.out.t())
    normal = normalize(x, c.epsilon)
    x = x+c.alpha_ff*((F.silu(normal@module.gate.t())*(normal@module.up.t()))@module.down.t())
    return u.new_zeros((batch*length, dim)).index_copy(0, positions, x).reshape_as(u), active


def run(binding, spec, x, *, start=0, left=None, full_attention="manual"):
    if full_attention not in {"manual", "sdpa", "sdpa4d"}:
        raise ValueError("unknown FC formula Attention operator")
    depth, width = validate(binding, spec, x)
    g, model = binding.graph, binding.model
    if left is not None and set(left) != set(range(len(g.nodes))):
        raise ValueError("FC tensor baseline left state node domain differs")
    values, controls, states, records = None, None, {}, []
    candidate_count, selected, messages = 0, 0, 0
    for region in range(depth):
        fields, prepared = [], []
        for local in range(width):
            node_id = region*width+local
            node = model.nodes[node_id]
            if region == 0:
                masses, incoming = x.new_ones((*x.shape[:2], 1)), [x]
                labels = [binding.source_classes.inputs[local]]
            else:
                masses, incoming = controls, values
                labels = [binding.source_classes.edges[e] for e, edge in enumerate(g.edges) if edge.target == node_id]
            mass, u = aggregate(node, masses, incoming, labels)
            times = window_times(spec, node_id, x, start)
            old = None if left is None else validate_left(left[node_id], node.kernel.config, x, times[0])
            descriptor, cache = prepare(node, u, mass, times, old)
            fields.append(descriptor); prepared.append(cache)
            states[node_id] = kv_terminal(node.kernel.config, cache[1], times[-1])
        descriptors = {name: torch.stack([row[name] for row in fields], 2) for name in fields[0]}
        r, program = binding.regions[region], model.regions[region]
        received, capacity = descriptors['w'].sum(-1), descriptors['mu'].sum(-1)
        z = (descriptors['w'][..., None]*descriptors['evidence']).sum(-2)/received.clamp_min(1)[..., None]
        context = head(program.head, torch.cat((z, received.log1p()[..., None], capacity.log1p()[..., None]), -1))
        low, high = received.clamp_max(1), capacity.clamp_max(r.k)
        target = low+context.sigmoid()*(high-low)
        controls = quadratic(descriptors['score'], descriptors['mu'], target, r.allocation_temperature)
        outputs = [full(model.nodes[region*width+j], prepared[j], controls[..., j], attention=full_attention) for j in range(width)]
        values = [output for output, _ in outputs]
        activity = torch.stack([active for _, active in outputs], -1)
        count = int(activity.sum())
        candidate_count += x.shape[0]*x.shape[1]*width
        selected += count
        if region != depth-1:
            messages += count*width
        records.append(dict(controls=controls, active=activity, descriptors=descriptors))
    mass = controls.sum(-1)
    hidden = (controls[..., None]*torch.stack(values, -2)).sum(-2)/mass.clamp_min(1)[..., None]
    return dict(hidden=hidden, states=states, regions=records,
                stats=dict(candidate_events=candidate_count, selected_events=selected, visited_edges=messages))


class TransformerLayers(nn.Module):
    """Declared conventional pre-RMS, causal SDPA, bias-free SwiGLU blocks."""
    def __init__(self, width=128, heads=2, ffn=512, depth=16, dtype=torch.float32):
        super().__init__()
        self.heads, self.epsilon = heads, 1e-5
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            qkv=nn.Linear(width, 3*width, bias=False, dtype=dtype),
            out=nn.Linear(width, width, bias=False, dtype=dtype),
            gate=nn.Linear(width, ffn, bias=False, dtype=dtype),
            up=nn.Linear(width, ffn, bias=False, dtype=dtype),
            down=nn.Linear(ffn, width, bias=False, dtype=dtype))) for _ in range(depth)])

    def forward(self, x):
        b, length, d = x.shape
        for block in self.blocks:
            n = normalize(x, self.epsilon)
            q, k, v = block['qkv'](n).reshape(b, length, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4).unbind()
            read = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape_as(x)
            x = x+block['out'](read)
            n = normalize(x, self.epsilon)
            x = x+block['down'](F.silu(block['gate'](n))*block['up'](n))
        return x
