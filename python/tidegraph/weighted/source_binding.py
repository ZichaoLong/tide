"""Architecture §2.2: four explicit source-transform parameter bindings.

Sharing binds parameters, not physical atoms, computations or private state.
Low-rank adapters are private to (node, class), including their embeddings.
"""
from collections.abc import Mapping
import torch
from .config import identity, positive_int
from .source_config import SourceBinding
from .tensors import norm, parameter




def aggregate_profile(classes, alpha, hidden, epsilon, binding):
    common = (classes, alpha, hidden, epsilon)
    # Keep the already published default graph/checkpoint identities unchanged.
    return identity("weighted-aggregate-v1", *common, *(() if binding == SourceBinding() else (binding,)))


def normalize_widths(widths, classes):
    """Freeze p_a; mappings must cover exactly the physical source classes."""
    if isinstance(widths, Mapping):
        labels = set((*classes.edges, *classes.inputs))
        if any(type(k) is not int for k in widths) or set(widths) != labels:
            raise ValueError("source_hidden_width class domain mismatch")
        for value in widths.values():
            positive_int("source_hidden_width", value)
        return tuple(sorted(widths.items()))
    positive_int("source_hidden_width", widths)
    return widths


def source_width(widths, label):
    return widths if type(widths) is int else dict(widths)[label]


class SourceTransform(torch.nn.Module):
    def __init__(self, a, b, embedding, factors=()):
        super().__init__()
        self.a, self.b, self.embedding = a, b, embedding
        if factors:
            self.a_left, self.a_right, self.b_left, self.b_right = factors

    def matrices(self):
        if not hasattr(self, "a_left"):
            return self.a, self.b
        return self.a+self.a_left@self.a_right, self.b+self.b_left@self.b_right

    def forward(self, z, epsilon):
        a, b = self.matrices()
        return b@torch.nn.functional.silu(a@norm(z, epsilon)+self.embedding)

    def validate(self, width, hidden, rank):
        expected = dict(a=(hidden, width), b=(width, hidden), embedding=(hidden,))
        if rank is not None:
            expected.update(a_left=(hidden, rank), a_right=(rank, width),
                            b_left=(width, rank), b_right=(rank, hidden))
        if self._parameters.keys() != expected.keys() or any(getattr(self, k).shape != v for k, v in expected.items()):
            raise ValueError("source transform parameter/adaptation shape mismatch")


class SourceFactory:
    def __init__(self, topology, classes, binding, width, hidden, generator, dtype):
        self.binding, self.width, self.hidden = binding, width, hidden
        self.generator, self.dtype, self.embeddings = generator, dtype, {}
        self.labels = []
        for v in range(len(topology.nodes)):
            labels = {classes.inputs[p] for p, n in enumerate(topology.inputs) if n == v}
            labels |= {classes.edges[e] for e, edge in enumerate(topology.edges) if edge.target == v}
            self.labels.append(sorted(labels))
        if binding.mode == "node":
            keys = list(range(len(topology.nodes)))
        elif binding.mode == "source":
            keys = sorted({(spec.region, a) for spec, labels in zip(topology.nodes, self.labels) for a in labels})
        else:
            keys = list(range(len(topology.regions)))
        widths = {key: set() for key in keys}
        for v, (spec, labels) in enumerate(zip(topology.nodes, self.labels)):
            for label in labels:
                widths[self.key(v, spec.region, label)].add(source_width(hidden, label))
        if any(len(hs) > 1 for hs in widths.values()):
            raise ValueError("source classes sharing A/B must have equal source_hidden_width")
        # Preserve the original scalar-width RNG order, even for empty groups.
        widths = {key: next(iter(hs)) if hs else hidden for key, hs in widths.items()
                  if hs or type(hidden) is int}
        self.a = {key: parameter((h, width), generator, dtype) for key, h in widths.items()}
        self.b = {key: parameter((width, h), generator, dtype) for key, h in widths.items()}

    def key(self, node, region, label):
        mode = self.binding.mode
        return node if mode == "node" else (region, label) if mode == "source" else region

    def transforms(self, node, region):
        result = {}
        for label in self.labels[node]:
            mode = self.binding.mode
            key, h = self.key(node, region, label), source_width(self.hidden, label)
            embedding_key = (node if mode in {"node", "adapted"} else region, label)
            if embedding_key not in self.embeddings:
                self.embeddings[embedding_key] = parameter((h,), self.generator, self.dtype)
            factors = ()
            if mode == "adapted":
                r, d = self.binding.rank, self.width
                factors = tuple(parameter(s, self.generator, self.dtype) for s in ((h,r),(r,d),(d,r),(r,h)))
            result[label] = SourceTransform(self.a[key], self.b[key], self.embeddings[embedding_key], factors)
        return result
