"""Stable local program slots, independent of physical edge/port identities."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PortIndex:
    offsets: tuple[int, ...]
    bindings: tuple[tuple[int, int], ...]  # kind 0: boundary port; kind 1: edge

    def row(self, node):
        return self.bindings[self.offsets[node]:self.offsets[node + 1]]


@dataclass(frozen=True)
class PortLayout:
    edge_source: tuple[int, ...]
    edge_target: tuple[int, ...]
    input: tuple[int, ...]
    output: tuple[int, ...]

    def __post_init__(self):
        for name in ("edge_source", "edge_target", "input", "output"):
            values = tuple(getattr(self, name))
            if any(type(x) is not int or not 0 <= x < 2**63 for x in values):
                raise ValueError("local slots require nonnegative int64 values")
            object.__setattr__(self, name, values)

    @classmethod
    def automatic(cls, graph):
        incoming, outgoing = [0] * len(graph.nodes), [0] * len(graph.nodes)
        def allocate(owners, counts):
            slots = []
            for node in owners:
                slots.append(counts[node]); counts[node] += 1
            return tuple(slots)
        # Incoming boundaries precede edges, matching the default atom order.
        inputs = allocate(graph.inputs, incoming)
        targets = allocate((e.target for e in graph.edges), incoming)
        sources = allocate((e.source for e in graph.edges), outgoing)
        outputs = allocate(graph.outputs, outgoing)
        return cls(sources, targets, inputs, outputs)

    def indexes(self, graph):
        """Validate a bijection per direction/node; compile flat slot->wire rows."""
        def index(edge_owners, ports, edge_slots, port_slots):
            if len(edge_slots) != len(edge_owners) or len(port_slots) != len(ports):
                raise ValueError("local slot layout size mismatch")
            rows = [[] for _ in graph.nodes]
            for node in (*edge_owners, *ports):
                rows[node].append(None)
            for kind, owners, slots in ((1, edge_owners, edge_slots), (0, ports, port_slots)):
                for source, (node, slot) in enumerate(zip(owners, slots)):
                    if slot >= len(rows[node]) or rows[node][slot] is not None:
                        raise ValueError("local slots must be a bijection for each node/direction")
                    rows[node][slot] = kind, source
            offsets, bindings = [0], []
            for row in rows:
                bindings.extend(row); offsets.append(len(bindings))
            return PortIndex(tuple(offsets), tuple(bindings))
        incoming = index(tuple(e.target for e in graph.edges), graph.inputs, self.edge_target, self.input)
        outgoing = index(tuple(e.source for e in graph.edges), graph.outputs, self.edge_source, self.output)
        return incoming, outgoing

    def incoming_slot(self, kind, source):
        if kind not in (0, 1):
            raise ValueError("invalid source kind")
        return (self.input if kind == 0 else self.edge_target)[source]
