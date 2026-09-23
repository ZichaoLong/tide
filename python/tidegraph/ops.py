"""ema-ffn-v1 local formulas, independent of graph scheduling."""
import torch
from torch import nn
from .records import State
from .memory import kernel, reset
from .full import ProjectionEmit
from .aggregate import SourceAggregate
from .content import as_content, Content
from .readout import LinearRead, program as make_read, evaluate as evaluate_read, request as read_request
from .next import program as next_program, AdoptNext
from . import lh_full
from .fiber_pool import PROFILES as FIBER_PROFILES, LEARNED as FIBER_LEARNED
from .clocked_state import with_clock


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
                 input_slots=0, aggregate_program=None, state_program=None, read_program=None, transition=None,
                 projection_layout="input"):
        super().__init__()
        def parameter(shape, scale):
            return nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype) * scale)
        self.decay = parameter((width,), 0.1)
        self.weight = parameter((width, width), 0.15)
        self.bias = parameter((width,), 0.05)
        self.read = parameter((width,), 0.2)
        self.kernel = kernel("ema" if spec is None else spec.memory, spec, input_slots) if state_program is None else state_program
        if spec is not None:
            self.kernel = with_clock(self.kernel, spec.state_clock)
        self.read_program = make_read("linear-v1" if spec is None else spec.readout) if read_program is None else read_program
        self.next_program = next_program("adopt-v1" if spec is None else spec.next_state) if transition is None else transition
        self.full_kind = "tanh" if spec is None else spec.full
        self.full_program = ProjectionEmit(spec) if full_program is None else full_program
        self.aggregate_program = SourceAggregate("sum" if spec is None else spec.aggregation) if aggregate_program is None else aggregate_program
        self.extra = nn.ParameterDict()
        if spec is not None and spec.memory == "lh-add-repeat-v1":
            self.extra["add_retention"] = nn.Parameter(torch.tensor(1.0 - 0.01, dtype=dtype))
        if spec is not None and spec.memory in FIBER_PROFILES:
            if width % spec.query_heads:
                raise ValueError("fiber attention width must be divisible by heads")
            self.extra["fiber_qkv"] = parameter((width, 3*width), .15)
            self.extra["fiber_out"] = parameter((width, width), .15)
            if projection_layout == "linear":
                for name in ("fiber_qkv", "fiber_out"):
                    self.extra[name] = nn.Parameter(self.extra[name].detach().t().contiguous().t())
            self.extra["fiber_qkv_bias"] = nn.Parameter(torch.zeros(3*width, dtype=dtype))
            self.extra["fiber_out_bias"] = nn.Parameter(torch.zeros(width, dtype=dtype))
            self.extra["fiber_decay"] = nn.Parameter(torch.tensor(.01, dtype=dtype))
            if FIBER_PROFILES[spec.memory] in FIBER_LEARNED:
                self.extra["fiber_pool"] = nn.Parameter(torch.ones(input_slots, dtype=dtype))
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
        if spec is not None and spec.memory in {"linear", "delta", "delta-rule-v1"}:
            for name in ("mem_q", "mem_k", "mem_v", "mem_out"):
                self.extra[name] = parameter((width, width), 0.15)
            if spec.memory != "linear":
                for name in (("mem_beta", "mem_decay") if spec.memory == "delta" else ("mem_beta",)):
                    self.extra[name] = parameter((width,), 0.15)
        if self.full_kind == "swiglu":
            for name, shape in (("ffn_gate", (width, width * 2)), ("ffn_up", (width, width * 2)),
                                ("ffn_down", (width * 2, width))):
                self.extra[name] = parameter(shape, 0.15)
        elif self.full_kind in lh_full.PROFILES:
            lh_full.initialize(self)
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

    def prepare(self, old, content, time, read_mode="proposal"):
        content = as_content(content)
        proposal = self.propose(old, content, time)
        return proposal, self.describe(old, proposal, content, time, read_mode)

    def propose(self, old, content, time):
        return self.kernel.step(self, old, as_content(content), time)

    def describe(self, old, proposal, content, time, read_mode="proposal"):
        r = read_request(read_mode, old, proposal, as_content(content), time)
        return evaluate_read(self, [r])[0]

    def describe_batch(self, old, proposals, batch, read_mode="proposal"):
        requests = [read_request(read_mode, a, b, c, t)
                    for a, b, c, t in zip(old, proposals, batch.views, batch.times)]
        return evaluate_read(self, requests, packed=True)

    def full(self, comparison, content, probability, mode, zeta):
        return emit(content, self.fresh(comparison, content), probability, mode, zeta)

    def fresh(self, comparison, content):
        if self.full_kind in lh_full.PROFILES:
            return lh_full.fresh(self, comparison)
        if self.full_kind == "swiglu":
            g = content + (torch.nn.functional.silu(comparison @ self.extra["ffn_gate"])
                           * (comparison @ self.extra["ffn_up"])) @ self.extra["ffn_down"]
        else:
            g = content + (comparison @ self.weight + self.bias).tanh()
        return g

    def propose_block(self, old, contents, times, views=None):
        views = [Content(h) for h in contents] if views is None else views
        return self.kernel.sequence(self, old, contents, times, views)

    def prepare_block(self, old, contents, times, views=None, read_mode="proposal"):
        views = [Content(h) for h in contents] if views is None else views
        states = self.propose_block(old, contents, times, views)
        previous = [old, *states[:-1]]
        return states, torch.stack([self.describe(a, b, c, t, read_mode) for a, b, c, t in zip(previous, states, views, times)])

    @property
    def can_prefill(self):
        return self.kernel.sequence_contract and self.next_program.comparison_identity

    def reset(self, state):
        return getattr(self.kernel, "reset", reset)(state)

    def validate(self, state):
        self.kernel.validate(self, state)


class Model(nn.Module):
    def __init__(self, graph, width=3, seed=7, dtype=torch.float64, full_programs=None, aggregate_programs=None,
                 state_programs=None, read_programs=None, next_programs=None, region_programs=None,
                 projection_layout="input"):
        super().__init__()
        if dtype not in (torch.float32, torch.float64) or width < 1:
            raise ValueError("CPU float32/float64 and positive width required")
        if projection_layout not in {"input", "linear"}:
            raise ValueError("unknown projection layout")
        if projection_layout != "input" and not any(n.memory in FIBER_PROFILES for n in graph.nodes):
            raise ValueError("nondefault projection layout requires same-fiber attention")
        self.width = width
        generator = torch.Generator().manual_seed(seed)
        programs = {} if full_programs is None else full_programs
        if any(v < 0 or v >= len(graph.nodes) or graph.nodes[v].identity for v in programs):
            raise ValueError("invalid custom Full owner")
        aggregates = {} if aggregate_programs is None else aggregate_programs
        if any(v < 0 or v >= len(graph.nodes) or graph.nodes[v].identity for v in aggregates):
            raise ValueError("invalid custom Aggregate owner")
        states = {} if state_programs is None else state_programs
        if any(v < 0 or v >= len(graph.nodes) or graph.nodes[v].identity for v in states):
            raise ValueError("invalid custom state owner")
        readers = {} if read_programs is None else read_programs
        if any(v < 0 or v >= len(graph.nodes) or graph.nodes[v].identity for v in readers):
            raise ValueError("invalid custom Read owner")
        transitions = {} if next_programs is None else next_programs
        if any(v < 0 or v >= len(graph.nodes) or graph.nodes[v].identity for v in transitions):
            raise ValueError("invalid custom Next owner")
        offsets = graph.port_indexes[1].offsets
        self.nodes = nn.ModuleList(BoundaryWeights(width, dtype) if n.identity else NodeWeights(
            width, generator, dtype, n, offsets[v+1] - offsets[v], programs.get(v),
            graph.source_counts[v], aggregates.get(v), states.get(v), readers.get(v), transitions.get(v),
            projection_layout) for v, n in enumerate(graph.nodes))
        from .region import program as region_program
        selectors = {} if region_programs is None else region_programs
        if any(type(r) is not int or not 0 <= r < len(graph.regions) for r in selectors):
            raise ValueError("invalid custom region owner")
        self.regions = nn.ModuleList(selectors[r] if r in selectors else region_program(layout, dtype)
                                     for r, layout in enumerate(graph.region_layouts))
        def scales(count):
            return nn.ParameterList(nn.Parameter(torch.tensor(0.8 + 0.03 * i, dtype=dtype))
                                    for i in range(count))
        self.input_scale = scales(len(graph.inputs))
        self.agg_scale = scales(len(graph.edges))
        self.edge_scale = scales(len(graph.edges))
        self.output_scale = scales(len(graph.outputs))


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
        self.read_program = LinearRead(identity=True)
        self.next_program = AdoptNext()

    def initial(self):
        return State(torch.zeros_like(self.bias))

    def prepare(self, old, content, time, read_mode="proposal"):
        return old, as_content(content).value.new_zeros(())

    def propose(self, old, content, time):
        return old

    def describe(self, old, proposal, content, time, read_mode="proposal"):
        return as_content(content).value.new_zeros(())

    def describe_batch(self, old, proposals, batch, read_mode="proposal"):
        return [batch.contents.new_zeros(()) for _ in batch.times]

    def propose_block(self, old, contents, times, views=None):
        return [old for _ in times]

    def prepare_block(self, old, contents, times, views=None, read_mode="proposal"):
        return [old for _ in times], contents.new_zeros((len(times),))

    @property
    def can_prefill(self):
        return True

    def full(self, comparison, content, probability, mode, zeta):
        return content

    def fresh(self, comparison, content):
        return content

    def reset(self, state):
        return reset(state)

    def validate(self, state):
        if state.slots:
            raise ValueError("identity state has unexpected slots")
