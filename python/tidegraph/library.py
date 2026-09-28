"""Public dependency API. Scheduling/formulas stay in the independent executors."""
from dataclasses import replace
import hashlib
from pathlib import Path
import torch
from .config import GraphConfig
from .execution_options import ExecutionOptions
from .ops import Model
from .records import Continuation, Result
from .runtime import resolve_device, manifest, synchronize


class GraphRuntime:
    """A model plus a compiled execution policy; external tasks own their loss.

    Device is explicit. Construct a new runtime to change device/dtype or
    parameter ownership; in-place optimizer updates and load_state_dict are
    supported. No inputs, task head, optimizer or output directories are hidden
    in this object. Several sessions may share its parameters.
    """
    def __init__(self, config, *, device, options=None, native_library=None, model=None):
        self.config = config if isinstance(config, GraphConfig) else GraphConfig.from_dict(config)
        c = self.config
        requested = c.execution if options is None else options
        if not isinstance(requested, ExecutionOptions):
            requested = ExecutionOptions(**requested)
        self.requested_options = requested
        self.device, self.resolution_reason = resolve_device(device)
        if requested.implementation == "native" and self.device.type != "cpu":
            raise ValueError("the native public runtime currently supports CPU only")
        if self.device.type not in {"cpu", "npu"}:
            raise ValueError("this library release supports CPU and experimental Python NPU only")
        if self.device.type == "npu" and c.dtype != "float32":
            raise ValueError("the NPU graph runtime requires float32")
        self.graph = c.graph
        self.spec = None
        if c.family == "settle":
            from .settle import SettleGraph
            self.spec = SettleGraph(c.graph, c.ranks)
        self.model = model if model is not None else Model(c.graph, c.width, c.seed, getattr(torch, c.dtype),
                                                         projection_layout=c.projection_layout, device=self.device)
        if (self.model.width != c.width or self.model.graph_identity != c.graph.identity
                or any(p.device != self.device or p.dtype != getattr(torch, c.dtype)
                       for p in self.model.state_dict().values())):
            raise ValueError("supplied model does not match configured width/nodes/device/dtype")
        if self.spec:
            self.execution_graph, self.execution_model = self.spec.embed(self.model)
            # The identity boundary buffers belong on the selected device too.
            self.execution_model.to(self.device)
        else:
            self.execution_graph, self.execution_model = c.graph, self.model
        self.options = requested.resolve(c.family, self.execution_graph)
        self.engine = None
        if self.options.implementation == "native":
            from .native_loader import load_native
            load_native(native_library)
            from .native import Native
            arguments = self.options.to_dict()
            arguments.pop("implementation")
            arguments["algorithm"] = arguments.pop("schedule")
            self.engine = Native(self.execution_graph, self.execution_model, **arguments)
        elif native_library is not None:
            raise ValueError("native_library requires implementation=native")

    def session(self, batch_size, *, continuation=None):
        from .session import Session
        return Session(self, batch_size, continuation=continuation)

    def _run(self, q, external, stop, sealed_until):
        o = self.options
        if self.engine:
            return self.engine.run(q, external, stop, sealed_until=sealed_until)
        base = dict(sealed_until=sealed_until, mode=o.mode, zeta=o.zeta)
        if o.schedule == "reference":
            from .reference import run
            return run(self.execution_graph, self.execution_model, q, external, stop, trace=o.trace, **base)
        policy = dict(packed=o.packed, full_autograd=o.full_autograd, aggregate_autograd=o.aggregate_autograd)
        if o.schedule == "streaming":
            from .streaming import run
            return run(self.execution_graph, self.execution_model, q, external, stop, trace=o.trace, **base, **policy)
        if o.schedule == "frontier":
            from .frontier import run
            return run(self.execution_graph, self.execution_model, q, external, stop,
                       trace=o.trace, prefill=o.prefill, max_events=o.max_events, **base, **policy)
        if o.schedule in {"chain", "diamond"}:
            from .specialized_blocks import run
            result = run(self.execution_graph, self.execution_model, q, external, stop,
                         topology=o.schedule, prefill=o.prefill, **base, **policy)
        else:
            from .specialized import run
            result = run(self.execution_graph, self.execution_model, q, external, stop, topology=o.schedule, **base)
        return result if o.trace else replace(result, trace=[], messages=[])

    def load_weights(self, path):
        """Weight-only initialization; never restores optimizer or session progress."""
        from .checkpoint_values import validate_weights
        record = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(record, dict) or record.get("schema") != "tide-continuation-v5" or record.get("identity") != self.execution_graph.identity:
            raise ValueError("checkpoint graph/schema mismatch")
        validate_weights(self.execution_model, record.get("weights"), record.get("aliases"))
        self.execution_model.load_state_dict(record["weights"])

    def synchronize(self):
        synchronize(self.device)

    def manifest(self):
        from .version import __version__
        root = Path(__file__).parent
        digest = hashlib.sha256()
        for path in sorted(root.glob("*.py")):
            digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
        record = dict(schema="tide-runtime-v1", package_version=__version__, package_sha256=digest.hexdigest(),
                      configuration=self.config.to_dict(), config_sha256=self.config.identity,
                      runtime=manifest(self.device, self.resolution_reason, getattr(torch, self.config.dtype)),
                      requested_options=self.requested_options.to_dict(), resolved_options=self.options.to_dict(),
                      graph_identity=self.graph.identity, execution_graph_identity=self.execution_graph.identity)
        if self.engine:
            path = Path(self.engine.core.__file__)
            record["native_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return record
