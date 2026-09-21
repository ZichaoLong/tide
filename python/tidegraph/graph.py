"""Immutable graph identity and independently checked adjacency compilation."""
from dataclasses import asdict, dataclass
import hashlib
import json


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


@dataclass(frozen=True)
class Region:
    budget: int
    observe_all: bool = True
    count_priority: bool = True


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    regions: tuple[Region, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]

    def __post_init__(self):
        n = len(self.nodes)
        if not n or not self.regions:
            raise ValueError("nodes and regions must be nonempty")
        if any(not 0 <= x.region < len(self.regions) for x in self.nodes):
            raise ValueError("invalid region owner")
        for r, spec in enumerate(self.regions):
            count = sum(x.region == r for x in self.nodes)
            if not 1 <= spec.budget <= count:
                raise ValueError("invalid region budget or empty region")
        for e in self.edges:
            if not 0 <= e.source < n or not 0 <= e.target < n:
                raise ValueError("invalid edge endpoint")
            if type(e.delay) is not int or not 0 < e.delay < 2**63:
                raise ValueError("edge delay must be a positive int64")
        if any(not 0 <= v < n for v in (*self.inputs, *self.outputs)):
            raise ValueError("invalid port owner")

    def wire(self):
        return asdict(self)

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
