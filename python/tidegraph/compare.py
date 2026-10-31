"""Strict typed comparisons and complete-observable objective helpers."""
import torch
from .comparison import (equivalent, collect_comparison, NumericMismatch,
                         DiscreteMismatch, StructureMismatch, NonFiniteMismatch)


def objective(result, root="all"):
    def square(value):
        return (value.float() if value.dtype == torch.float16 else value).square().sum()
    terms = []
    if root in {"all", "output"}:
        terms += [square(x) * 0.7 for _, _, _, x in result.outputs]
    if root in {"all", "state"}:
        # Dictionaries represent unordered mathematical fields. Fix the floating
        # reduction order before comparing Python insertion order with std::map.
        states = [s for _,s in sorted(result.continuation.states.items())]
        terms += [square(s.value) * 0.3 for s in states]
        terms += [square(v) * 0.11 for s in states for _,v in sorted(s.slots.items())
                  if v.is_floating_point() or v.is_complex()]
    if root in {"all", "history"}:
        terms += [square(v) * 0.13 for _,h in sorted(result.continuation.history.items())
                  for _,v in sorted(h.tensors.items())]
    if root in {"all", "pending"}:
        terms += [square(m.value) * 0.2 for m in result.continuation.pending]
    if not terms:
        raise ValueError("objective has no tensor roots")
    return sum(term.to(terms[0].device) for term in terms)
