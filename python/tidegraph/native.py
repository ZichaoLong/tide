"""Tensor-preserving adapter for the separately compiled LibTorch core."""
from .records import Atom, Continuation, Result, State


class Native:
    def __init__(self, graph, model, *, workers=1, packed=False, trace=True, mode="hard", zeta=1.0,
                 algorithm="streaming", prefill=True, max_events=1000000):
        import _tide_native as core
        self.core, self.graph, self.model = core, graph, model
        g = core.Graph()
        g.nodes = [core.Node(n.region, n.clear) for n in graph.nodes]
        g.edges = [core.Edge(e.source, e.target, e.delay) for e in graph.edges]
        g.regions = [core.Region(r.budget, r.observe_all, r.count_priority) for r in graph.regions]
        g.inputs, g.outputs = graph.inputs, graph.outputs
        g.compile()
        self.compiled = g
        m = core.Model()
        m.nodes = [core.NodeWeights(w.decay, w.weight, w.bias, w.read) for w in model.nodes]
        for field in ("input_scale", "agg_scale", "edge_scale", "output_scale"):
            setattr(m, field, list(getattr(model, field)))
        options = core.Options()
        options.workers, options.packed, options.trace = workers, packed, trace
        options.mode, options.zeta = mode, zeta
        options.prefill, options.max_events = prefill, max_events
        if algorithm not in {"streaming", "frontier"}:
            raise ValueError("unknown native algorithm")
        self.engine = (core.Streaming if algorithm == "streaming" else core.Frontier)(g, m, options)

    def run(self, continuation, external, stop, *, sealed_until):
        c = self.core
        if continuation.identity != self.graph.identity:
            raise ValueError("continuation graph identity mismatch")
        q = c.Continuation()
        q.identity, q.batch_size, q.cut = self.compiled.identity, continuation.batch_size, continuation.cut
        q.states = {k: c.State(s.value, s.last_time, s.observations) for k, s in continuation.states.items()}
        q.history, q.ledger = continuation.history, continuation.ledger
        q.pending = [c.Atom(a.batch, a.node, a.time, a.kind, a.source, a.position, a.value)
                     for a in continuation.pending]
        xs = [c.External(x.batch, x.port, x.position, x.time, x.value) for x in external]
        result = self.engine.run(q, xs, stop, sealed_until)
        def atom(a):
            return Atom(a.batch, a.node, a.time, a.kind, a.source, a.position, a.value)
        r = result.continuation
        out_q = Continuation(self.graph.identity, r.batch_size, r.cut,
                             {k: State(s.value, s.last_time, s.observations) for k, s in r.states.items()},
                             r.history, [atom(a) for a in r.pending], r.ledger)
        events = []
        for e in result.trace:
            event = {k: getattr(e, k) for k in ("batch", "node", "time", "content", "proposal", "descriptor",
                                              "control", "comparison", "next", "active", "history")}
            event["fiber"] = [atom(a) for a in e.fiber]
            if e.active:
                event["full"] = e.full
            events.append(event)
        return Result(out_q, events, [(o.batch, o.time, o.port, o.value) for o in result.outputs],
                      [atom(a) for a in result.messages], result.stats)
