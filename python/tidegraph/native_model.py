"""Validate built-in programs and encode tensors; performs no graph execution."""


def encode_model(core, graph, model):
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
    from .ownership import region_reference
    for r, (p, layout) in enumerate(zip(model.regions, graph.region_layouts)):
        validate_region(p, layout, region_reference(graph, model, r), native=True)
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
    return g, m
