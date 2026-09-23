"""Tensor-preserving adapter for the separately compiled LibTorch core."""
from .records import Result
from .native_records import from_continuation, to_continuation, window_records
from .coordinates import window_inputs


class Native:
    def __init__(self, graph, model, *, workers=1, packed=False, trace=True, mode="hard", zeta=1.0,
                 algorithm="streaming", prefill=True, max_events=1000000, parallel_regions=False, compact_events=False,
                 attention_packing="exact", fiber_pooling="event", fiber_cache="cloned", defer_state_release=False,
                 attention_layout="event", packed_sources=False, batch_next=False):
        if attention_packing not in {"exact", "single"}:
            raise ValueError("invalid fiber attention packing")
        if fiber_pooling not in {"event", "csr"}:
            raise ValueError("invalid fiber pooling execution")
        if fiber_cache not in {"cloned", "owned"}:
            raise ValueError("invalid fiber cache ownership")
        if attention_layout not in {"event", "head"}:
            raise ValueError("invalid fiber attention layout")
        if defer_state_release and not compact_events:
            raise ValueError("deferred state release requires compact events")
        if (packed_sources or batch_next) and not packed:
            raise ValueError("packed transport requires packed Streaming")
        import _tide_native as core
        self.core, self.graph, self.model = core, graph, model
        self.algorithm = algorithm
        from .full import ProjectionEmit, validate_program
        from .aggregate import SourceAggregate, validate_program as validate_aggregate
        from .state_program import validate_program as validate_state_program
        from .readout import validate_program as validate_read
        from .next import validate_program as validate_next
        for v, spec in enumerate(graph.nodes):
            if type(model.nodes[v].full_program) is not ProjectionEmit:
                raise ValueError("Python custom Full has no native implementation")
            offsets = graph.port_indexes[1].offsets
            validate_program(model.nodes[v], spec, offsets[v+1] - offsets[v])
            if type(model.nodes[v].aggregate_program) is not SourceAggregate:
                raise ValueError("Python custom Aggregate has no native implementation")
            validate_aggregate(model.nodes[v], spec, graph.source_counts[v])
            validate_state_program(model.nodes[v], spec, slots=graph.source_counts[v], native=True)
            validate_read(model.nodes[v], spec, native=True)
            validate_next(model.nodes[v], spec, native=True)
        from .region import validate_program as validate_region
        if len(model.regions) != len(graph.regions):
            raise ValueError("region program count mismatch")
        for p, layout in zip(model.regions, graph.region_layouts):
            validate_region(p, layout, model.nodes[0].bias, native=True)
        g = core.Graph()
        nodes = []
        for n in graph.nodes:
            node = core.Node(n.region, n.clear, n.identity, n.memory, n.full, n.query_heads, n.kv_heads, n.window,
                             n.emission, n.emit_period, n.emit_phases, n.aggregation, n.readout, n.next_state)
            node.state_clock = core.StateClock(n.state_clock.period, n.state_clock.first, n.state_clock.count)
            nodes.append(node)
        g.nodes = nodes
        g.edges = [core.Edge(e.source, e.target, e.delay) for e in graph.edges]
        g.regions = [core.Region(r.budget, r.observe_all, r.count_priority, r.read_mode, r.selector) for r in graph.regions]
        g.inputs, g.outputs = graph.inputs, graph.outputs
        layout = core.PortLayout()
        for name in ("edge_source", "edge_target", "input", "output"):
            setattr(layout, name, getattr(graph.ports, name))
        g.layout = layout
        g.origins = [core.InputOrigin(o.edge, o.port, o.stride) for o in graph.origins]
        domain = core.SourceDomain()
        domain.edge_target, domain.input = graph.domain.edge_target, graph.domain.input
        g.source_domain = domain
        g.compile()
        self.compiled = g
        m = core.Model()
        weights = []
        for w in model.nodes:
            weight = core.NodeWeights(w.decay, w.weight, w.bias, w.read)
            weight.extra = dict(w.extra.items())
            weights.append(weight)
        m.nodes = weights
        regions = []
        for program in model.regions:
            weight = core.RegionWeights()
            weight.extra = dict(program.named_parameters())
            regions.append(weight)
        m.regions = regions
        for field in ("input_scale", "agg_scale", "edge_scale", "output_scale"):
            setattr(m, field, list(getattr(model, field)))
        core.configure_fiber_attention(g, m, attention_packing, fiber_pooling, fiber_cache, attention_layout)
        self.weights = m  # Tensor-preserving model record for other native clients.
        options = core.Options()
        options.workers, options.packed, options.trace = workers, packed, trace
        options.mode, options.zeta = mode, zeta
        options.prefill, options.max_events = prefill, max_events
        options.parallel_regions, options.compact_events = parallel_regions, compact_events
        options.defer_state_release = defer_state_release
        options.packed_sources, options.batch_next = packed_sources, batch_next
        if algorithm != "streaming" and (parallel_regions or compact_events or defer_state_release or packed_sources or batch_next):
            raise ValueError("streaming optimizations require the streaming algorithm")
        if algorithm not in {"streaming", "frontier", "self_loop", "chain"}:
            raise ValueError("unknown native algorithm")
        if algorithm in {"self_loop", "chain"}:
            self.engine = core.Specialized(g, m, options, algorithm)
        else:
            self.engine = (core.Streaming if algorithm == "streaming" else core.Frontier)(g, m, options)

    def run(self, continuation, external, stop, *, sealed_until):
        external = window_inputs(external, stop, sealed_until)
        q = to_continuation(self.core, self.graph, self.compiled, continuation)
        xs = [self.core.External(x.batch, x.port, x.position, x.time, x.value) for x in external]
        result = self.engine.run(q, xs, stop, sealed_until)
        return Result(from_continuation(self.graph, result.continuation), *window_records(result))

    def cursor(self, continuation):
        if self.algorithm != "streaming":
            raise ValueError("owned cursors currently require the streaming algorithm")
        from .cursor import NativeCursor
        return NativeCursor(self, continuation)
