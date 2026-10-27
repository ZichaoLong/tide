"""Weighted-message architecture §2.1: SrcClass is parameter classification.

Unlike SourceDomain, equal class labels never identify/deduplicate atoms.
Bindings use physical incoming ports/edges; an atom retains its complete tag.
"""
from dataclasses import dataclass
from .coordinates import integers


@dataclass(frozen=True)
class SourceClasses:
    edges: tuple[int, ...]
    inputs: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        integers("source classes", *self.edges, *self.inputs)
        if any(label < 0 for label in (*self.edges, *self.inputs)):
            raise ValueError("source classes must be nonnegative")

    def validate(self, graph):
        if len(self.edges) != len(graph.edges) or len(self.inputs) != len(graph.inputs):
            raise ValueError("source class physical domain mismatch")

    def classify(self, atom):
        integers("source atom", atom.kind, atom.source)
        domain = self.inputs if atom.kind == 0 else self.edges if atom.kind == 1 else ()
        if not 0 <= atom.source < len(domain):
            raise ValueError("source atom outside class domain")
        return domain[atom.source]
