"""ema-ffn-v1 local formulas, independent of graph scheduling."""
import torch
from torch import nn
from .records import State
from .memory import kernel, reset
from .full import ProjectionEmit
from .aggregate import SourceAggregate


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
    def __init__(self, width, generator, dtype, spec=None, output_slots=0, full_program=None,
                 input_slots=0, aggregate_program=None):
        super().__init__()
        def parameter(shape, scale):
            return nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype) * scale)
        self.decay = parameter((width,), 0.1)
        self.weight = parameter((width, width), 0.15)
        self.bias = parameter((width,), 0.05)
        self.read = parameter((width,), 0.2)
        self.kernel = kernel("ema" if spec is None else spec.memory, spec)
        self.full_kind = "tanh" if spec is None else spec.full
        self.full_program = ProjectionEmit(spec) if full_program is None else full_program
        self.aggregate_program = SourceAggregate("sum" if spec is None else spec.aggregation) if aggregate_program is None else aggregate_program
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
        if isinstance(self.full_program, ProjectionEmit) and self.full_program.kind == "slot_affine":
            for slot in range(output_slots):
                self.extra[f"emit_w_{slot}"] = parameter((width, width), 0.2)
                self.extra[f"emit_b_{slot}"] = parameter((width,), 0.05)
        if isinstance(self.aggregate_program, SourceAggregate):
            kind = self.aggregate_program.kind
            prefix = "agg_mass_" if kind == "weighted_mean" else "agg_logit_" if "softmax" in kind else None
            if prefix:
                for slot in range(input_slots):
                    self.extra[f"{prefix}{slot}"] = parameter((), 0.2)

    def initial(self):
        return self.kernel.initial(self)

    def prepare(self, old, content, time):
        proposal = self.kernel.step(self, old, content, time)
        return proposal, self.describe(old, proposal, content, time)

    def describe(self, old, proposal, content, time):
        return (proposal.value * self.read).sum(-1)

    def full(self, comparison, content, probability, mode, zeta):
        return emit(content, self.fresh(comparison, content), probability, mode, zeta)

    def fresh(self, comparison, content):
        if self.full_kind == "swiglu":
            g = content + (torch.nn.functional.silu(comparison @ self.extra["ffn_gate"])
                           * (comparison @ self.extra["ffn_up"])) @ self.extra["ffn_down"]
        else:
            g = content + (comparison @ self.weight + self.bias).tanh()
        return g

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
    def __init__(self, graph, width=3, seed=7, dtype=torch.float64, full_programs=None, aggregate_programs=None):
        super().__init__()
        if dtype not in (torch.float32, torch.float64) or width < 1:
            raise ValueError("CPU float32/float64 and positive width required")
        self.width = width
        generator = torch.Generator().manual_seed(seed)
        programs = {} if full_programs is None else full_programs
        if any(v < 0 or v >= len(graph.nodes) or graph.nodes[v].identity for v in programs):
            raise ValueError("invalid custom Full owner")
        aggregates = {} if aggregate_programs is None else aggregate_programs
        if any(v < 0 or v >= len(graph.nodes) or graph.nodes[v].identity for v in aggregates):
            raise ValueError("invalid custom Aggregate owner")
        offsets = graph.port_indexes[1].offsets
        incoming = graph.port_indexes[0].offsets
        self.nodes = nn.ModuleList(BoundaryWeights(width, dtype) if n.identity else NodeWeights(
            width, generator, dtype, n, offsets[v+1] - offsets[v], programs.get(v),
            incoming[v+1] - incoming[v], aggregates.get(v)) for v, n in enumerate(graph.nodes))
        def scales(count):
            return nn.ParameterList(nn.Parameter(torch.tensor(0.8 + 0.03 * i, dtype=dtype))
                                    for i in range(count))
        self.input_scale = scales(len(graph.inputs))
        self.agg_scale = scales(len(graph.edges))
        self.edge_scale = scales(len(graph.edges))
        self.output_scale = scales(len(graph.outputs))


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
        from .graph import Node
        self.full_program = ProjectionEmit(Node(0, identity=True))
        self.aggregate_program = SourceAggregate()

    def initial(self):
        return State(torch.zeros_like(self.bias))

    def prepare(self, old, content, time):
        return old, content.new_zeros(())

    def describe(self, old, proposal, content, time):
        return content.new_zeros(())

    def prepare_block(self, old, contents, times):
        return [old for _ in times], contents.new_zeros((len(times),))

    @property
    def can_prefill(self):
        return True

    def full(self, comparison, content, probability, mode, zeta):
        return content

    def fresh(self, comparison, content):
        return content

    def next(self, comparison, clear):
        return reset(comparison) if clear else comparison

    def validate(self, state):
        if state.slots:
            raise ValueError("identity state has unexpected slots")
