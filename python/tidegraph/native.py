"""Tensor-preserving adapter for the separately compiled LibTorch core."""
from .records import Result
from .native_records import from_continuation, to_continuation, window_records


class Native:
    def __init__(self, graph, model, *, workers=1, packed=False, trace=True, mode="hard", zeta=1.0,
                 algorithm="streaming", prefill=True, max_events=1000000):
        import _tide_native as core
        self.core, self.graph, self.model = core, graph, model
        self.algorithm = algorithm
        g = core.Graph()
        g.nodes = [core.Node(n.region, n.clear, n.identity, n.memory, n.full, n.query_heads, n.kv_heads, n.window)
                   for n in graph.nodes]
        g.edges = [core.Edge(e.source, e.target, e.delay) for e in graph.edges]
        g.regions = [core.Region(r.budget, r.observe_all, r.count_priority) for r in graph.regions]
        g.inputs, g.outputs = graph.inputs, graph.outputs
        layout = core.PortLayout()
        for name in ("edge_source", "edge_target", "input", "output"):
            setattr(layout, name, getattr(graph.ports, name))
        g.layout = layout
        g.compile()
        self.compiled = g
        m = core.Model()
        weights = []
        for w in model.nodes:
            weight = core.NodeWeights(w.decay, w.weight, w.bias, w.read)
            weight.extra = dict(w.extra.items())
            weights.append(weight)
        m.nodes = weights
        for field in ("input_scale", "agg_scale", "edge_scale", "output_scale"):
            setattr(m, field, list(getattr(model, field)))
        options = core.Options()
        options.workers, options.packed, options.trace = workers, packed, trace
        options.mode, options.zeta = mode, zeta
        options.prefill, options.max_events = prefill, max_events
        if algorithm not in {"streaming", "frontier", "self_loop", "chain"}:
            raise ValueError("unknown native algorithm")
        if algorithm in {"self_loop", "chain"}:
            self.engine = core.Specialized(g, m, options, algorithm)
        else:
            self.engine = (core.Streaming if algorithm == "streaming" else core.Frontier)(g, m, options)

    def run(self, continuation, external, stop, *, sealed_until):
        q = to_continuation(self.core, self.graph, self.compiled, continuation)
        xs = [self.core.External(x.batch, x.port, x.position, x.time, x.value) for x in external]
        result = self.engine.run(q, xs, stop, sealed_until)
        return Result(from_continuation(self.graph, result.continuation), *window_records(result))

    def cursor(self, continuation):
        if self.algorithm != "streaming":
            raise ValueError("owned cursors currently require the streaming algorithm")
        from .cursor import NativeCursor
        return NativeCursor(self, continuation)
