"""Versioned, JSON-compatible graph/model configuration for dependency users."""
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from .graph import Graph, Node, Region, Edge
from .clocks import StateClock
from .ports import PortLayout
from .origins import InputOrigin
from .source_domain import SourceDomain
from .execution_options import ExecutionOptions


def _record(kind, values):
    if not isinstance(values, dict):
        raise ValueError(f"{kind.__name__} must be an object")
    values = dict(values)
    for key in ("clear", "identity", "observe_all", "count_priority"):
        if key in values and type(values[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if kind is Node and "state_clock" in values:
        if not isinstance(values["state_clock"], StateClock):
            values["state_clock"] = _record(StateClock, values["state_clock"])
    try:
        return kind(**values)
    except TypeError as error:
        raise ValueError(f"invalid {kind.__name__}: {error}") from error


def graph_from_dict(data):
    if not isinstance(data, dict):
        raise ValueError("graph must be an object")
    allowed = {"nodes", "edges", "regions", "inputs", "outputs", "layout", "origins", "source_domain"}
    if set(data) - allowed or not {"nodes", "edges", "regions", "inputs", "outputs"} <= data.keys():
        raise ValueError("invalid graph fields")
    return Graph(tuple(_record(Node, n) for n in data["nodes"]),
                 tuple(_record(Edge, e) for e in data["edges"]),
                 tuple(_record(Region, r) for r in data["regions"]),
                 tuple(data["inputs"]), tuple(data["outputs"]),
                 _record(PortLayout, data["layout"]) if data.get("layout") is not None else None,
                 tuple(_record(InputOrigin, o) for o in data.get("origins", [])),
                 _record(SourceDomain, data["source_domain"]) if data.get("source_domain") is not None else None)


@dataclass(frozen=True)
class GraphConfig:
    family: str
    graph: Graph
    width: int = 8
    dtype: str = "float32"
    seed: int = 7
    ranks: tuple[int, ...] = ()
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    projection_layout: str = "input"
    scale_init: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "ranks", tuple(self.ranks))
        if self.family not in {"pdg", "timed-dag", "settle"} or not isinstance(self.graph, Graph):
            raise ValueError("family must be pdg, timed-dag or settle with a Graph")
        if type(self.width) is not int or self.width < 1 or self.dtype not in {"float16", "float32", "float64"}:
            raise ValueError("positive width and float16/float32/float64 required")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative int64")
        if self.projection_layout not in {"input", "linear"} or not isinstance(self.execution, ExecutionOptions):
            raise ValueError("invalid model layout/execution options")
        if self.scale_init is not None and (type(self.scale_init) not in (int,float) or not math.isfinite(self.scale_init)):
            raise ValueError("scale_init must be finite or null")
        if self.family != "pdg":
            self.graph.topological_order()
        if self.family == "settle":
            from .settle import SettleGraph
            SettleGraph(self.graph, self.ranks)
        elif self.ranks:
            raise ValueError("ranks are only valid for SettleGraph")
        from .fiber_pool import PROFILES
        for node in self.graph.nodes:
            if (node.memory == "attention" or node.memory in PROFILES) and self.width % node.query_heads:
                raise ValueError("model width must be divisible by attention query heads")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("expected graph config schema_version=1")
        if set(data) - {"schema_version", "family", "graph", "topology", "ranks", "model", "execution"}:
            raise ValueError("unknown graph configuration field")
        if ("graph" in data) == ("topology" in data):
            raise ValueError("specify exactly one of graph or topology")
        try:
            if "topology" in data:
                from .topology import topology
                graph, ranks = topology(**data["topology"])
            else:
                graph, ranks = graph_from_dict(data["graph"]), ()
            family = data["family"]
            ranks = data.get("ranks", ranks if family == "settle" else ())
            model = data.get("model", {})
            if set(model) - {"width", "dtype", "seed", "projection_layout", "scale_init"}:
                raise ValueError("unknown model configuration field")
            return cls(family, graph, ranks=ranks, execution=ExecutionOptions(**data.get("execution", {})), **model)
        except (TypeError, KeyError) as error:
            raise ValueError(f"invalid graph configuration: {error}") from error

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return {"schema_version": 1, "family": self.family, "graph": self.graph.wire(),
                "ranks": list(self.ranks), "model": {"width": self.width, "dtype": self.dtype,
                "seed": self.seed, "projection_layout": self.projection_layout,
                "scale_init": self.scale_init}, "execution": self.execution.to_dict()}

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
