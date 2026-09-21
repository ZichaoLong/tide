"""ema-ffn-v1 local formulas, independent of graph scheduling."""
import torch
from torch import nn
from .records import State
from .memory import kernel, reset


class _HST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, g, p, zeta):
        ctx.save_for_backward(g - h)
        ctx.zeta = zeta
        return g.clone()

    @staticmethod
    def backward(ctx, grad):
        (delta,) = ctx.saved_tensors
        return torch.zeros_like(grad), grad, (grad * delta).sum(-1) * ctx.zeta, None


def emit(h, g, p, mode, zeta=1.0):
    if mode == "hard":
        return g
    if mode == "softp":
        return h + p.unsqueeze(-1) * (g - h)
    if mode == "hst":
        return _HST.apply(h, g, p, zeta)
    raise ValueError(f"unsupported emit mode: {mode}")


class NodeWeights(nn.Module):
    def __init__(self, width, generator, dtype, spec=None):
        super().__init__()
        def parameter(shape, scale):
            return nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype) * scale)
        self.decay = parameter((width,), 0.1)
        self.weight = parameter((width, width), 0.15)
        self.bias = parameter((width,), 0.05)
        self.read = parameter((width,), 0.2)
        self.kernel = kernel("ema" if spec is None else spec.memory, spec)
        self.full_kind = "tanh" if spec is None else spec.full
        self.extra = nn.ParameterDict()
        if spec is not None and spec.memory == "attention":
            if width % spec.query_heads:
                raise ValueError("attention width must be divisible by query heads")
            kv_width = width // spec.query_heads * spec.kv_heads
            for name, size in (("attn_q", width), ("attn_k", kv_width), ("attn_v", kv_width), ("attn_out", width)):
                self.extra[name] = parameter((width, size), 0.15)
        if spec is not None and spec.memory == "ssm":
            for name in ("ssm_dt", "ssm_b", "ssm_c"):
                self.extra[name] = parameter((width, width), 0.15)
            self.extra["ssm_a"] = parameter((width,), 0.1)
            self.extra["ssm_skip"] = parameter((width,), 0.2)
        if spec is not None and spec.memory in {"linear", "delta"}:
            for name in ("mem_q", "mem_k", "mem_v", "mem_out"):
                self.extra[name] = parameter((width, width), 0.15)
            if spec.memory == "delta":
                for name in ("mem_beta", "mem_decay"):
                    self.extra[name] = parameter((width,), 0.15)
        if self.full_kind == "swiglu":
            for name, shape in (("ffn_gate", (width, width * 2)), ("ffn_up", (width, width * 2)),
                                ("ffn_down", (width * 2, width))):
                self.extra[name] = parameter(shape, 0.15)
        elif self.full_kind != "tanh":
            raise ValueError("unknown Full profile")

    def initial(self):
        return self.kernel.initial(self)

    def prepare(self, old, content, time):
        proposal = self.kernel.step(self, old, content, time)
        return proposal, (proposal.value * self.read).sum(-1)

    def full(self, comparison, content, probability, mode, zeta):
        if self.full_kind == "swiglu":
            g = content + (torch.nn.functional.silu(comparison @ self.extra["ffn_gate"])
                           * (comparison @ self.extra["ffn_up"])) @ self.extra["ffn_down"]
        else:
            g = content + (comparison @ self.weight + self.bias).tanh()
        return emit(content, g, probability, mode, zeta)

    def prepare_block(self, old, contents, times):
        states = self.kernel.sequence(self, old, contents, times)
        return states, (torch.stack([s.value for s in states]) * self.read).sum(-1)

    @property
    def can_prefill(self):
        return self.kernel.sequence_contract

    def next(self, comparison, clear):
        return getattr(self.kernel, "reset", reset)(comparison) if clear else comparison

    def validate(self, state):
        self.kernel.validate(self, state)


class Model(nn.Module):
    def __init__(self, graph, width=3, seed=7, dtype=torch.float64):
        super().__init__()
        if dtype not in (torch.float32, torch.float64) or width < 1:
            raise ValueError("CPU float32/float64 and positive width required")
        self.width = width
        generator = torch.Generator().manual_seed(seed)
        self.nodes = nn.ModuleList(BoundaryWeights(width, dtype) if n.identity else NodeWeights(width, generator, dtype, n)
                                   for n in graph.nodes)
        def scales(count):
            return nn.ParameterList(nn.Parameter(torch.tensor(0.8 + 0.03 * i, dtype=dtype))
                                    for i in range(count))
        self.input_scale = scales(len(graph.inputs))
        self.agg_scale = scales(len(graph.edges))
        self.edge_scale = scales(len(graph.edges))
        self.output_scale = scales(len(graph.outputs))

    def aggregate(self, atoms):
        values = [a.value * (self.input_scale[a.source] if a.kind == 0
                            else self.agg_scale[a.source]) for a in atoms]
        # Explicit canonical fold also fixes floating addition order.
        h = values[0]
        for x in values[1:]:
            h = h + x
        return h


def select(nodes, descriptors, history, region):
    if any(not torch.isfinite(d).all() for d in descriptors.values()):
        raise ValueError("nonfinite selector score")
    order = sorted(nodes, key=lambda v: (history.get(v, 0) if region.count_priority else 0,
                                        -float(descriptors[v].detach()), v))
    active = set(order[:region.budget])
    probs = torch.stack([descriptors[v] for v in nodes]).softmax(0)
    updated = dict(history)
    for v in active:
        updated[v] = updated.get(v, 0) + 1
    return active, dict(zip(nodes, probs.unbind(0))), updated


class BoundaryWeights(nn.Module):
    """Stateless identity adapter, with no trainable parameters."""
    def __init__(self, width, dtype):
        super().__init__()
        for name, shape in (("decay", (width,)), ("weight", (width, width)), ("bias", (width,)), ("read", (width,))):
            self.register_buffer(name, torch.zeros(shape, dtype=dtype))
        self.extra = nn.ParameterDict()

    def initial(self):
        return State(torch.zeros_like(self.bias))

    def prepare(self, old, content, time):
        return old, content.new_zeros(())

    def prepare_block(self, old, contents, times):
        return [old for _ in times], contents.new_zeros((len(times),))

    @property
    def can_prefill(self):
        return True

    def full(self, comparison, content, probability, mode, zeta):
        return content

    def next(self, comparison, clear):
        return reset(comparison) if clear else comparison

    def validate(self, state):
        if state.slots:
            raise ValueError("identity state has unexpected slots")
