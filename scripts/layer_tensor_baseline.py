"""Independent tensor formulas for a restricted core-5 layer-chain benchmark.

0770593 KV §2/4 and SSD §2/3. This is a projected-output cost anchor, not a
graph executor or a standard Transformer/Mamba implementation. It accepts one
unit-strength input per original-time position, a singleton proportional region
per layer, zero initial states and alpha_agg=0. No reference trajectory is used.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F

from tidegraph import External
from tidegraph.weighted import bind, fully_connected, KVConfig, RegionalConfig, RecurrentConfig


def numerics(device, policy="ieee"):
    """Explicitly match the public experiment's matmul/convolution policy."""
    if policy not in {"ieee", "vendor-default"}:
        raise ValueError("unknown layer numerical policy")
    from tidegraph.experiment.admission import precision
    if policy == "ieee":
        effective = precision(device)
    elif device.type == "npu":
        effective = dict(matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                         conv_allow_hf32=torch.npu.conv.allow_hf32)
    elif device.type == "cuda":
        effective = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    else:
        effective = dict(float32_matmul_precision=torch.get_float32_matmul_precision())
    return dict(requested=policy, effective=effective)


def fixture(family, *, width=128, depth=4, heads=4, dtype=torch.float32, device="cpu"):
    spec = fully_connected(depth, 1)
    common = dict(width=width, summary_width=max(2, width//4), ffn_width=2*width, heads=heads)
    if family == "kv":
        memory = KVConfig(**common, adapter_rank=max(1, width//8))
    elif family == "ssd":
        memory = RecurrentConfig(**common, memory_family="ssd", key_width=max(2, width//heads),
                                 value_width=width//heads, conv_size=3, conv_stride=spec.stride)
    else:
        raise ValueError("layer tensor anchor supports KV or SSD")
    binding = bind(spec.graph, memories=memory, regions=RegionalConfig(selection_mode="proportional", k=1),
                   alpha_agg=0., dtype=dtype, device=device)
    return binding, replace(spec, graph=binding.graph)


def validate(binding, spec, values):
    from tidegraph.weighted.kv import KVMemory, KVContent
    from tidegraph.weighted.ssd import SSDMemory, SSDContent
    graph = binding.graph
    count = len(graph.nodes)
    if (spec.graph.identity != graph.identity or spec.binding != "ports"
            or graph.inputs != (0,) or graph.outputs != (count-1,)
            or len(graph.regions) != count or [n.region for n in graph.nodes] != list(range(count))
            or [(e.source, e.target, e.delay) for e in graph.edges] != [(i, i+1, 1) for i in range(count-1)]):
        raise ValueError("tensor anchor requires an adjacent singleton-region chain")
    binding.model.validate_binding(graph)
    ref = binding.model.nodes[0].bias
    if (values.ndim != 3 or min(values.shape[:2]) < 1 or values.shape[-1] != binding.memories[0].width
            or values.dtype not in (torch.float32, torch.float64)
            or (values.device, values.dtype) != (ref.device, ref.dtype) or not torch.isfinite(values).all()):
        raise ValueError("tensor anchor requires finite B,L,d input on the model device/dtype")
    if values.shape[1] > 256:
        raise ValueError("tensor anchor dense causal workspace is bounded to L<=256")
    for node, region in zip(binding.model.nodes, binding.regions):
        memory, full = node.kernel, node.full_program.content
        c = memory.config
        if (region.selection_mode != "proportional" or region.k != 1 or region.send_gain != "fixed"
                or region.fixed_gain != 1 or region.gain_logit is not None or region.region_score != "ind"
                or node.aggregate_program.alpha != 0 or memory.support_policy != "strict"):
            raise ValueError("tensor anchor requires unit proportional sending and identity singleton Aggregate")
        if type(memory) is KVMemory and type(full) is KVContent:
            if any(c.full_frequencies) or any(c.distance_slopes):
                raise ValueError("tensor KV anchor has no positional rotation or distance bias")
        elif type(memory) is SSDMemory and type(full) is SSDContent:
            if c.conv_stride != spec.stride:
                raise ValueError("tensor SSD convolution stride must equal the original-time position stride")
        else:
            raise ValueError("tensor anchor requires the declared KV/SSD modules")


def external(spec, values):
    """No input compression: D*t+rank with constant nondifferentiable mass 1."""
    payload = torch.cat((torch.ones_like(values[..., :1]), values), -1)
    return [External(b, 0, t, t*spec.stride+spec.rank(0), payload[b, t])
            for b in range(values.shape[0]) for t in range(values.shape[1])]


def normalize(x, epsilon):
    return x/torch.sqrt(epsilon+x.square().mean(-1, keepdim=True))


def finish(full, x, epsilon, alpha):
    n = normalize(x, epsilon)
    return x+alpha*(F.silu(n@full.gate.t())*(n@full.up.t()))@full.down.t()


def kv(memory, full, x, times):
    c = memory.config
    n = normalize(x, c.epsilon)
    key, value = (n@memory.base.t()+(n@memory.adapter_v.t())@memory.adapter_u.t()).chunk(2, -1)
    batch, length, _ = x.shape
    shape = (batch, length, c.heads, c.width//c.heads)
    q = (n@full.query.t()).reshape(shape).transpose(1, 2)
    k, v = (t.reshape(shape).transpose(1, 2) for t in (key, value))
    age = times[:, None]-times[None, :]
    visible = age >= 0
    if c.window:
        visible = visible & (age < c.window)
    read = F.scaled_dot_product_attention(q, k, v, attn_mask=visible).transpose(1, 2).reshape_as(x)
    result = finish(full, x+c.alpha_att*(read@full.out.t()), c.epsilon, c.alpha_ff)
    keep = torch.ones_like(times, dtype=torch.bool) if not c.window else times[-1]-times < c.window
    raw = dict(times=times[keep], keys=key[:, keep], values=value[:, keep],
               summaries=(normalize(x, c.epsilon)@memory.summary.t())[:, keep],
               weights=x.new_ones((batch, int(keep.sum()))))
    return result, raw


def ssd(memory, full, x, times):
    c = memory.config
    gate, raw, dt = (normalize(x, c.epsilon)@memory.projection.t()).split(
        (c.heads*c.value_width, c.conv_width, c.heads), -1)
    # Correlation tap zero is the current input. All times are original int64
    # positions spaced by conv_stride, so this convolution has exact slot identity.
    convolved = F.conv1d(F.pad(raw.transpose(1, 2), (c.conv_size-1, 0)),
                         memory.conv.weight.flip(0).t().unsqueeze(1), memory.conv.bias,
                         groups=c.conv_width).transpose(1, 2)
    value, key, query = F.silu(convolved).split(
        (c.heads*c.value_width, c.groups*c.key_width, c.groups*c.key_width), -1)
    batch, length, _ = x.shape
    value = value.reshape(batch, length, c.heads, c.value_width)
    key, query = (a.reshape(batch, length, c.groups, c.key_width).repeat_interleave(
        c.heads//c.groups, dim=2) for a in (key, query))
    dt = F.softplus(dt+memory.delta_bias)
    gaps = torch.cat((times[:1], times[1:]-times[:-1])).to(x.dtype)
    log_a = -memory.log_rate.exp()*dt-gaps[None, :, None]*x.new_tensor(c.matrix_rates)
    prefix = log_a.cumsum(1).transpose(1, 2)
    visible = times[:, None] >= times[None, :]
    difference = prefix.unsqueeze(-1)-prefix.unsqueeze(-2)
    transition = torch.where(visible, torch.where(visible, difference, 0).exp(), 0)
    pairs = torch.einsum("bthk,bshk->bhts", query, key)
    written = dt[..., None]*value
    read = torch.einsum("bhts,bshv->bthv", transition*pairs, written)
    y = (read+full.direct[None, None, :, None]*value).flatten(2)*F.silu(gate)
    mix = (full.rho*normalize(y, c.output_epsilon))@full.out.t()
    result = finish(full, x+c.alpha_mem*mix, c.epsilon, c.alpha_ff)
    h = torch.einsum("bhs,bshv,bshk->bhvk", transition[:, :, -1], written, key)
    decay = torch.exp(-(times[-1]-times).to(x.dtype)[:, None]*x.new_tensor((c.lambda_short, 0.)))
    summary = normalize(x, c.epsilon)@memory.summary.t()
    raw_state = dict(time=times[-1], H=h, Z=torch.einsum("lc,bld->bcd", decay, summary),
                     W=decay.sum(0).expand(batch, -1), conv_times=times[-c.conv_size:],
                     conv_values=raw[:, -c.conv_size:])
    return result, raw_state


def run(binding, spec, values):
    """Return output content and terminal tensor fields for this zero-left-state domain.

    Skipped auxiliary Describe fields are outside this projection; no complete
    trace/checkpoint equivalence is claimed. The algebra eliminates dead
    dependencies, so its disconnected zero derivatives are compared numerically
    with graph derivatives; it does not certify the graph's None/zero contract.
    """
    validate(binding, spec, values)
    states, x = [], values
    for v, node in enumerate(binding.model.nodes):
        times = torch.arange(values.shape[1], device=x.device, dtype=torch.int64)*spec.stride+spec.rank(v)
        function = kv if isinstance(node.kernel.config, KVConfig) else ssd
        x, raw = function(node.kernel, node.full_program.content, x, times)
        states.append(raw)
    return x, states
