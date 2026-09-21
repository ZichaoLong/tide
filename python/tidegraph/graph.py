"""Immutable graph identity and independently checked adjacency compilation."""
from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
from .ports import PortLayout
from .origins import InputOrigin, validate_origins


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    delay: int


@dataclass(frozen=True)
class Node:
    region: int
    clear: bool = False
    identity: bool = False
    memory: str = "ema"
    full: str = "tanh"
    query_heads: int = 1
    kv_heads: int = 1
    window: int = 0  # 0 means unbounded; otherwise accepted observations, including current.
    emission: str = "broadcast"
    emit_period: int = 1
    emit_phases: tuple[int, ...] = ()  # -1 always, -2 never; empty means all.
    aggregation: str = "sum"
    readout: str = "linear-v1"

    def __post_init__(self):
        object.__setattr__(self, "emit_phases", tuple(self.emit_phases))


@dataclass(frozen=True)
class Region:
    budget: int
    observe_all: bool = True
    count_priority: bool = True
    read_mode: str = "proposal"


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    regions: tuple[Region, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    layout: PortLayout | None = None
    origins: tuple[InputOrigin, ...] = ()

    def __post_init__(self):
        n = len(self.nodes)
        if not n or not self.regions:
            raise ValueError("nodes and regions must be nonempty")
        if any(not 0 <= x.region < len(self.regions) for x in self.nodes):
            raise ValueError("invalid region owner")
        for x in self.nodes:
            if (any(type(v) is not int for v in (x.query_heads, x.kv_heads, x.window))
                    or not 1 <= x.kv_heads <= x.query_heads < 2**63
                    or x.query_heads % x.kv_heads or not 0 <= x.window < 2**63):
                raise ValueError("invalid attention heads/window")
        for r, spec in enumerate(self.regions):
            if spec.read_mode not in {"content", "old", "proposal"}:
                raise ValueError("invalid region Read mode")
            count = sum(x.region == r for x in self.nodes)
            if not 1 <= spec.budget <= count:
                raise ValueError("invalid region budget or empty region")
        for e in self.edges:
            if not 0 <= e.source < n or not 0 <= e.target < n:
                raise ValueError("invalid edge endpoint")
            if type(e.delay) is not int or not 0 < e.delay < 2**63:
                raise ValueError("edge delay must be a positive int64")
        validate_origins(self.origins, len(self.edges))
        object.__setattr__(self, "origins", tuple(sorted(self.origins, key=lambda x: x.edge)))
        if any(not 0 <= v < n for v in (*self.inputs, *self.outputs)):
            raise ValueError("invalid port owner")
        if self.layout is not None and not isinstance(self.layout, PortLayout):
            raise ValueError("invalid local port layout")
        self.port_indexes  # Validate before any execution or checkpoint operation.
        outgoing = self.port_indexes[1]
        for v, node in enumerate(self.nodes):
            degree = outgoing.offsets[v + 1] - outgoing.offsets[v]
            if (type(node.emit_period) is not int or not 0 < node.emit_period < 2**63
                    or (node.emit_phases and len(node.emit_phases) != degree)
                    or any(type(p) is not int or not -2 <= p < node.emit_period for p in node.emit_phases)):
                raise ValueError("invalid emission phase policy")
            if node.identity and (node.emission != "broadcast" or node.emit_phases or node.emit_period != 1):
                raise ValueError("identity boundaries require unconditional broadcast")
            if node.identity and node.readout != "linear-v1":
                raise ValueError("identity boundaries require the default Read profile")
            if node.identity and node.aggregation != "sum":
                raise ValueError("identity boundaries require sum Aggregate")

    @cached_property
    def origin_index(self):
        return {origin.edge: origin for origin in self.origins}

    @cached_property
    def ports(self):
        return PortLayout.automatic(self) if self.layout is None else self.layout

    @cached_property
    def port_indexes(self):
        return self.ports.indexes(self)

    def wire(self):
        record = asdict(self)
        record["layout"] = asdict(self.ports)
        return record

    @property
    def identity(self):
        raw = json.dumps(self.wire(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def adjacency(self, outgoing=True):
        """CSR/CSC returns edge IDs, preserving parallel-edge identities."""
        rows = [[] for _ in self.nodes]
        for i, e in enumerate(self.edges):
            rows[e.source if outgoing else e.target].append(i)
        offsets, ids = [0], []
        for row in rows:
            ids.extend(row)
            offsets.append(len(ids))
        return offsets, ids

    def topological_order(self):
        degree = [0] * len(self.nodes)
        rows = [[] for _ in self.nodes]
        for e in self.edges:
            degree[e.target] += 1
            rows[e.source].append(e.target)
        ready = [v for v, d in enumerate(degree) if d == 0]
        order = []
        while ready:
            v = ready.pop()
            order.append(v)
            for w in rows[v]:
                degree[w] -= 1
                if degree[w] == 0:
                    ready.append(w)
        if len(order) != len(self.nodes):
            raise ValueError("TimedDAG requires an acyclic node graph")
        return order
