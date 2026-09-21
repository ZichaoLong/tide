"""SettleGraph direct region-major execution and explicit TimedDAG embedding."""
from collections import defaultdict
from dataclasses import dataclass, replace
import torch
from .blocks import canonicalize, deliver, evaluate_block
from .graph import Edge, Graph, Node, Region
from .ops import Model
from .records import Atom, Continuation, External, Result
from .validation import validate_window


@dataclass(frozen=True)
class SettleGraph:
    graph: Graph
    ranks: tuple[int, ...]  # One strictly positive rank per region.

    def __post_init__(self):
        g = self.graph
        if (len(self.ranks) != len(g.regions) or len(set(self.ranks)) != len(self.ranks)
                or any(type(rank) is not int or rank <= 0 for rank in self.ranks)):
            raise ValueError("regions require distinct positive ranks")
        if not g.inputs or not g.outputs:
            raise ValueError("SettleGraph requires input and output boundaries")
        for e in g.edges:
            if e.delay != self.rank(e.target) - self.rank(e.source) or e.delay <= 0:
                raise ValueError("SettleGraph edges must follow region ranks with matching delays")

    def rank(self, node):
        return self.ranks[self.graph.nodes[node].region]

    @property
    def output_rank(self):
        return max(self.ranks) + 1

    @property
    def stride(self):
        return self.output_rank + 1

    def external(self, values, start_position=0, encoded=False):
        if values.ndim != 3:
            raise ValueError("values require [batch, sequence, width]")
        return [External(b, p, start_position + t,
                         self.stride * (start_position + t) + (0 if encoded else self.rank(v)), values[b, t])
                for b in range(len(values)) for p, v in enumerate((0,) if encoded else self.graph.inputs)
                for t in range(values.shape[1])]

    def embed(self, model):
        g = self.graph
        n, r = len(g.nodes), len(g.regions)
        edges = (g.edges + tuple(Edge(n, v, self.rank(v)) for v in g.inputs)
                 + tuple(Edge(v, n + 1, self.output_rank - self.rank(v)) for v in g.outputs))
        encoded = Graph(g.nodes + (Node(r, identity=True), Node(r + 1, identity=True)), edges,
                        g.regions + (Region(1), Region(1)), (n,), (n + 1,))
        em = Model(encoded, model.width, dtype=model.nodes[0].bias.dtype)
        em.nodes = torch.nn.ModuleList(list(model.nodes) + list(em.nodes[-2:]))
        def one():
            return torch.nn.Parameter(model.nodes[0].bias.new_ones(()), requires_grad=False)
        em.input_scale = torch.nn.ParameterList([one()])
        em.agg_scale = torch.nn.ParameterList(list(model.agg_scale) + list(model.input_scale) + list(model.output_scale))
        em.edge_scale = torch.nn.ParameterList(list(model.edge_scale) + [one() for _ in (*g.inputs, *g.outputs)])
        em.output_scale = torch.nn.ParameterList([one()])
        return encoded, em

    def embed_initial(self, q, encoded):
        if q.cut or q.pending or q.ledger or q.history:
            raise ValueError("embed_initial requires an initial cut; retain encoded continuation for subsequent windows")
        return replace(q.fork(), identity=encoded.identity)

    def project(self, result):
        if result.continuation.cut % self.stride or result.continuation.pending:
            raise ValueError("SettleGraph projection requires a complete position cut")
        g = self.graph
        n, r, e = len(g.nodes), len(g.regions), len(g.edges)
        def atom(a):
            if a.source < e:
                return a
            port = a.source - e
            if not 0 <= port < len(g.inputs):
                raise ValueError("invalid body input adapter edge")
            return Atom(a.batch, a.node, a.time, 0, port, a.position // self.stride, a.value)
        events = []
        for event in result.trace:
            if event["node"] < n:
                events.append(dict(event, fiber=[atom(a) for a in event["fiber"]]))
        q = result.continuation
        ledger = {(b, p): (position, time + self.rank(v)) for (b, _), (position, time) in q.ledger.items()
                  for p, v in enumerate(g.inputs)}
        projected = Continuation(g.identity, q.batch_size, q.cut, {k: s for k, s in q.states.items() if k[1] < n},
                                 {k: h for k, h in q.history.items() if k[1] < r},
                                 [a for a in q.pending if a.source < e], ledger)
        return Result(projected, events, result.outputs, [a for a in result.messages if a.source < e], result.stats)


def run(spec, model, q, values, *, mode="hard", zeta=1.0, prefill=True):
    """Independent SettleGraph schedule: each region settles the whole sequence."""
    if q.cut % spec.stride or q.pending or values.shape[0] != q.batch_size:
        raise ValueError("SettleGraph continuation must be a complete position boundary")
    start = q.cut // spec.stride
    stop = q.cut + values.shape[1] * spec.stride
    graph = spec.graph
    atoms, ledger = validate_window(graph, model, q, spec.external(values, start), stop, stop)
    q = q.fork(); q.ledger = ledger
    fibers = defaultdict(list)
    for a in atoms:
        fibers[a.batch, a.node, a.time].append(a)
    events, messages, raw_outputs = [], [], []
    stats = {"region_blocks": 0, "state_blocks": 0, "state_steps": 0, "full_blocks": 0}
    for region in sorted(range(len(graph.regions)), key=lambda r: spec.ranks[r]):
        members = {v for v, n in enumerate(graph.nodes) if n.region == region}
        frames = [(b, region, spec.stride * t + spec.ranks[region], members)
                  for t in range(start, start + values.shape[1]) for b in range(q.batch_size)]
        if not frames:
            continue
        block, counters = evaluate_block(graph, model, q, frames, fibers, mode=mode, zeta=zeta, prefill=prefill)
        deliver(graph, model, block, fibers, messages, raw_outputs)
        events.extend(block); stats["region_blocks"] += 1
        for key, value in counters.items():
            stats[key] = stats.get(key, 0) + value
    groups = defaultdict(list)
    for b, time, port, value in raw_outputs:
        position = (time - spec.rank(graph.outputs[port])) // spec.stride
        groups[b, position].append((port, value))
    outputs = []
    for (b, position), pairs in groups.items():
        values_out = [v for _, v in sorted(pairs)]
        value = values_out[0]
        for v in values_out[1:]:
            value = value + v
        outputs.append((b, position * spec.stride + spec.output_rank, 0, value))
    q.cut = stop
    result = canonicalize(graph, Result(q, events, outputs, messages, stats))
    result.outputs.sort(key=lambda o: (o[1], o[0]))
    return result
