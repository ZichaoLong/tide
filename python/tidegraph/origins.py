"""Graph-owned views of boundary messages after an explicit time embedding."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class InputOrigin:
    edge: int
    port: int
    stride: int


def validate_origins(origins, edge_count):
    seen = set()
    for origin in origins:
        if (not isinstance(origin, InputOrigin)
                or any(type(v) is not int for v in (origin.edge, origin.port, origin.stride))
                or not 0 <= origin.edge < edge_count or origin.edge in seen
                or not 0 <= origin.port < 2**63 or not 0 < origin.stride < 2**63):
            raise ValueError("invalid input origin view")
        seen.add(origin.edge)


def view(graph, atom):
    origin = graph.origin_index.get(atom.source) if atom.kind == 1 and graph.origins else None
    if origin is None:
        return atom
    if atom.position % origin.stride:
        raise ValueError("message does not lie on input origin clock")
    return replace(atom, kind=0, source=origin.port, position=atom.position // origin.stride)
