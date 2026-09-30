"""Tensor-preserving adapter for the separately compiled LibTorch core."""
from .records import Result
from .native_records import from_continuation, to_continuation, window_records
from .coordinates import window_inputs


class Native:
    def __init__(self, graph, model, *, workers=1, packed=False, trace=True, mode="hard", zeta=1.0,
                 algorithm="streaming", prefill=True, max_events=1000000, parallel_regions=False, compact_events=False,
                 attention_packing="exact", fiber_pooling="event", fiber_cache="cloned", defer_state_release=False,
                 attention_layout="event", packed_sources=False, batch_next=False, full_autograd="replay", aggregate_autograd="replay",
                 placement=None):
        if aggregate_autograd not in {"replay", "batched"} or (aggregate_autograd == "batched" and not packed):
            raise ValueError("invalid Aggregate autograd policy or unpacked execution")
        if full_autograd not in {"replay", "batched"} or (full_autograd == "batched" and not packed):
            raise ValueError("invalid Full autograd policy or unpacked execution")
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
            raise ValueError("packed transport requires packed execution")
        import _tide_native as core
        backend = model.nodes[0].bias.device.type
        if backend == "npu" and fiber_pooling == "csr":
            raise ValueError("NPU CSR pooling is unsupported; explicitly select fiber_pooling='event'")
        compiled = getattr(core, "execution_backend", lambda: "cpu")()
        if backend != "cpu" and backend != compiled:
            raise RuntimeError(f"native {backend} requested but the adapter was built for {compiled}")
        self.core, self.graph, self.model = core, graph, model
        self.algorithm = algorithm
        from .native_model import encode_model
        g, m = encode_model(core, graph, model)
        self.compiled = g
        core.configure_fiber_attention(g, m, attention_packing, fiber_pooling, fiber_cache, attention_layout)
        self.placement = None
        if placement is not None:
            from .placement import request
            requested = request(placement)
            config = core.ExecutionPlacement()
            for name, value in requested.to_dict().items():
                setattr(config, name, value)
            self.placement = dict(requested=requested.to_dict(),
                                  resolved=core.resolve_placement(config, model.nodes[0].bias.device))
            m = core.place_model(g, m, config)
        self.weights = m  # Tensor-preserving model record for other native clients.
        options = core.Options()
        options.workers, options.packed, options.trace = workers, packed, trace
        options.full_autograd = full_autograd
        options.aggregate_autograd = aggregate_autograd
        options.mode, options.zeta = mode, zeta
        options.prefill, options.max_events = prefill, max_events
        options.parallel_regions, options.compact_events = parallel_regions, compact_events
        options.defer_state_release = defer_state_release
        options.packed_sources, options.batch_next = packed_sources, batch_next
        self.options = options  # Scheduler-only options for native frontend clients.
        if algorithm not in {"streaming", "frontier", "greedy", "self_loop", "ring", "chain", "diamond"}:
            raise ValueError("unknown native algorithm")
        if algorithm in {"self_loop", "ring", "chain", "diamond"}:
            self.engine = core.Specialized(g, m, options, algorithm)
        else:
            engine = core.Greedy if algorithm == "greedy" else core.Streaming if algorithm == "streaming" else core.Frontier
            self.engine = engine(g, m, options)

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
