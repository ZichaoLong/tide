"""Public execution policy, separate from graph/model and checkpoint identity."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class ExecutionOptions:
    implementation: str = "python"
    schedule: str = "auto"
    packed: bool = True
    prefill: bool | None = None
    trace: bool = False
    mode: str = "hard"
    zeta: float = 1.0
    workers: int = 1
    max_events: int = 1000000
    full_autograd: str = "replay"
    aggregate_autograd: str = "replay"
    parallel_regions: bool = False
    compact_events: bool = False
    defer_state_release: bool = False
    packed_sources: bool = False
    batch_next: bool = False
    attention_packing: str = "exact"
    fiber_pooling: str = "event"
    fiber_cache: str = "cloned"
    attention_layout: str = "event"

    def __post_init__(self):
        if self.implementation not in {"python", "native"}:
            raise ValueError("implementation must be python or native")
        if self.schedule not in {"auto", "reference", "streaming", "frontier", "greedy", "chain", "diamond", "ring", "self_loop"}:
            raise ValueError("unknown execution schedule")
        if self.mode not in {"hard", "hst", "softp"} or type(self.zeta) not in (int, float) or not math.isfinite(self.zeta):
            raise ValueError("invalid mode/zeta")
        for key in ("packed", "trace", "parallel_regions", "compact_events", "defer_state_release", "packed_sources", "batch_next"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.prefill is not None and type(self.prefill) is not bool:
            raise ValueError("prefill must be boolean or null")
        for key in ("workers", "max_events"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("full_autograd", "aggregate_autograd"):
            if getattr(self, key) not in {"replay", "batched"} or (getattr(self, key) == "batched" and not self.packed):
                raise ValueError(f"invalid {key} or unpacked batched request")
        for key, choices in {"attention_packing": {"exact", "single"}, "fiber_pooling": {"event", "csr"},
                             "fiber_cache": {"cloned", "owned"}, "attention_layout": {"event", "head"}}.items():
            if getattr(self, key) not in choices:
                raise ValueError(f"invalid {key}")
        if self.defer_state_release and not self.compact_events:
            raise ValueError("deferred release requires compact_events")
        if (self.packed_sources or self.batch_next) and not self.packed:
            raise ValueError("transport policies require packed execution")

    def resolve(self, family, graph):
        schedule = self.schedule
        if schedule == "auto":
            schedule = "streaming" if family == "pdg" else "frontier"
        prefill = schedule in {"frontier", "greedy", "chain", "diamond"} if self.prefill is None else self.prefill
        if schedule in {"reference", "streaming", "ring", "self_loop"} and prefill:
            raise ValueError("this schedule has no time-prefill contract")
        if schedule in {"frontier", "chain", "diamond"}:
            graph.topological_order()
        if schedule == "reference" and (self.implementation != "python" or self.packed):
            raise ValueError("reference requires python and packed=false")
        if schedule not in {"frontier", "greedy"} and self.max_events != 1000000:
            raise ValueError("max_events is only configurable for frontier or greedy")
        if self.implementation == "python":
            baseline = ExecutionOptions()
            for key in ("workers", "parallel_regions", "compact_events", "defer_state_release", "packed_sources",
                        "batch_next", "attention_packing", "fiber_pooling", "fiber_cache", "attention_layout"):
                if getattr(self, key) != getattr(baseline, key):
                    raise ValueError(f"{key} is not implemented by the Python public runtime")
            if schedule in {"ring", "self_loop"} and self.packed:
                raise ValueError("Python cyclic specializations require packed=false")
        if schedule in {"chain", "diamond", "ring", "self_loop"}:
            from .specialized import validate_topology
            validate_topology(graph, schedule)
        from .fiber_pool import PROFILES
        if (self.attention_packing, self.fiber_pooling, self.fiber_cache, self.attention_layout) != ("exact", "event", "cloned", "event"):
            if not any(node.memory in PROFILES for node in graph.nodes):
                raise ValueError("fiber kernel policies require a same-fiber module")
        return replace(self, schedule=schedule, prefill=prefill)

    def to_dict(self):
        return asdict(self)
