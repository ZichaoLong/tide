"""Independent tensor formulas for a restricted core-5 layer-chain benchmark.

0770593 KV §2/4 and SSD §2/3. This is a projected-output cost anchor, not a
graph executor or a standard Transformer/Mamba implementation. It accepts one
unit-strength input per original-time position, a singleton proportional region
per layer and alpha_agg=0. Raw nonzero left states and absolute positions are
explicit inputs. No reference trajectory is used.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F

from tensor_baseline_state import times as window_times, validate as validate_left, kv_join, kv_terminal, check_position

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


def external(spec, values, *, start=0):
    """No input compression: D*t+rank with constant nondifferentiable mass 1."""
    check_position(spec, 0, values, start)
    payload = torch.cat((torch.ones_like(values[..., :1]), values), -1)
    return [External(b, 0, start+t, (start+t)*spec.stride+spec.rank(0), payload[b, t])
            for b in range(values.shape[0]) for t in range(values.shape[1])]


def normalize(x, epsilon):
    return x/torch.sqrt(epsilon+x.square().mean(-1, keepdim=True))


def finish(full, x, epsilon, alpha):
    n = normalize(x, epsilon)
    return x+alpha*(F.silu(n@full.gate.t())*(n@full.up.t()))@full.down.t()


def kv(memory, full, x, times, left=None):
    c = memory.config
    n = normalize(x, c.epsilon)
    key, value = (n@memory.base.t()+(n@memory.adapter_v.t())@memory.adapter_u.t()).chunk(2, -1)
    raw, _, visible = kv_join(c, times, key, value, n@memory.summary.t(),
                              x.new_ones(x.shape[:2]), left)
    batch, length, _ = x.shape
    shape = (batch, length, c.heads, c.width//c.heads)
    q = (n@full.query.t()).reshape(shape).transpose(1, 2)
    k, v = (raw[name].reshape(batch, -1, c.heads, c.width//c.heads).transpose(1, 2)
            for name in ("keys", "values"))
    weights = raw["weights"][:, None, :]*visible
    positive = weights > 0
    mask = torch.where(positive, weights, 1).log().masked_fill(~positive, -torch.inf)
    read = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None]).transpose(1, 2).reshape_as(x)
    read = read*weights.sum(-1).clamp_max(1)[..., None]
    result = finish(full, x+c.alpha_att*(read@full.out.t()), c.epsilon, c.alpha_ff)
    return result, kv_terminal(c, raw, times[-1])


def ssd(memory, full, x, times, left=None):
    c = memory.config
    gate, raw, dt = (normalize(x, c.epsilon)@memory.projection.t()).split(
        (c.heads*c.value_width, c.conv_width, c.heads), -1)
    # Correlation tap zero is the current input. All times are original int64
    # positions spaced by conv_stride, so this convolution has exact slot identity.
    if left is None:
        prefix_raw = raw.new_zeros((x.shape[0], c.conv_size-1, c.conv_width))
        old_time = times.new_zeros(())
    else:
        taps = times[0]-c.conv_stride*torch.arange(c.conv_size-1, 0, -1, device=x.device, dtype=torch.int64)
        matches = (taps[:, None] == left["conv_times"][None, :]).to(x.dtype)
        prefix_raw = matches@left["conv_values"]
        old_time = left["time"]
    convolved = F.conv1d(torch.cat((prefix_raw, raw), 1).transpose(1, 2),
                         memory.conv.weight.flip(0).t().unsqueeze(1), memory.conv.bias,
                         groups=c.conv_width).transpose(1, 2)
    value, key, query = F.silu(convolved).split(
        (c.heads*c.value_width, c.groups*c.key_width, c.groups*c.key_width), -1)
    batch, length, _ = x.shape
    value = value.reshape(batch, length, c.heads, c.value_width)
    key, query = (a.reshape(batch, length, c.groups, c.key_width).repeat_interleave(
        c.heads//c.groups, dim=2) for a in (key, query))
    dt = F.softplus(dt+memory.delta_bias)
    gaps = torch.cat((times[:1]-old_time, times[1:]-times[:-1])).to(x.dtype)
    log_a = -memory.log_rate.exp()*dt-gaps[None, :, None]*x.new_tensor(c.matrix_rates)
    prefix = log_a.cumsum(1).transpose(1, 2)
    visible = times[:, None] >= times[None, :]
    difference = prefix.unsqueeze(-1)-prefix.unsqueeze(-2)
    transition = torch.where(visible, torch.where(visible, difference, 0).exp(), 0)
    pairs = torch.einsum("bthk,bshk->bhts", query, key)
    written = dt[..., None]*value
    read = torch.einsum("bhts,bshv->bthv", transition*pairs, written)
    if left is not None:
        read = read+torch.einsum("bht,bhvk,bthk->bthv", prefix.exp(), left["H"], query)
    y = (read+full.direct[None, None, :, None]*value).flatten(2)*F.silu(gate)
    mix = (full.rho*normalize(y, c.output_epsilon))@full.out.t()
    result = finish(full, x+c.alpha_mem*mix, c.epsilon, c.alpha_ff)
    h = torch.einsum("bhs,bshv,bshk->bhvk", transition[:, :, -1], written, key)
    decay = torch.exp(-(times[-1]-times).to(x.dtype)[:, None]*x.new_tensor((c.lambda_short, 0.)))
    summary = normalize(x, c.epsilon)@memory.summary.t()
    z, w = torch.einsum("lc,bld->bcd", decay, summary), decay.sum(0).expand(batch, -1)
    stamps, cache = times, raw
    if left is not None:
        h = h+prefix[:, :, -1].exp()[..., None, None]*left["H"]
        old_decay = torch.exp(-(times[-1]-old_time).to(x.dtype)*x.new_tensor((c.lambda_short, 0.)))
        z, w = z+old_decay[None, :, None]*left["Z"], w+old_decay*left["W"]
        stamps = torch.cat((left["conv_times"], times))
        cache = torch.cat((left["conv_values"], raw), 1)
    keep = stamps >= times[-1]-(c.conv_size-1)*c.conv_stride
    raw_state = dict(time=times[-1], H=h, Z=z, W=w, conv_times=stamps[keep], conv_values=cache[:, keep])
    return result, raw_state


def run(binding, spec, values, *, start=0, left=None):
    """Return output content and exact terminal tensor fields in the checked domain.

    Skipped auxiliary Describe fields are outside this projection; no complete
    trace/checkpoint equivalence is claimed. The algebra eliminates dead
    dependencies, so its disconnected zero derivatives are compared numerically
    with graph derivatives; it does not certify the graph's None/zero contract.
    """
    validate(binding, spec, values)
    if left is not None and len(left) != len(binding.model.nodes):
        raise ValueError("tensor baseline left state node count differs")
    states, x = [], values
    for v, node in enumerate(binding.model.nodes):
        times = window_times(spec, v, values, start)
        old = None if left is None else validate_left(left[v], node.kernel.config, values, times[0])
        function = kv if isinstance(node.kernel.config, KVConfig) else ssd
        x, raw = function(node.kernel, node.full_program.content, x, times, old)
        states.append(raw)
    return x, states
