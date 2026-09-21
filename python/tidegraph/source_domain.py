"""Logical incoming source slots, independent of physical wire/port slots."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDomain:
    edge_target: tuple[int, ...]
    input: tuple[int, ...]

    def __post_init__(self):
        for name in ("edge_target", "input"):
            values = tuple(getattr(self, name))
            if any(type(x) is not int or not 0 <= x < 2**63 for x in values):
                raise ValueError("source domain requires nonnegative int64 slots")
            object.__setattr__(self, name, values)

    def counts(self, graph):
        if len(self.edge_target) != len(graph.edges) or len(self.input) != len(graph.inputs):
            raise ValueError("source domain size mismatch")
        rows = [set() for _ in graph.nodes]
        for edge, slot in zip(graph.edges, self.edge_target):
            rows[edge.target].add(slot)
        for node, slot in zip(graph.inputs, self.input):
            rows[node].add(slot)
        for row in rows:
            if row and (min(row) != 0 or max(row) != len(row) - 1):
                raise ValueError("source domain must be dense for each node")
        return tuple(map(len, rows))
