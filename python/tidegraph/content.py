"""Read-only local content; raw physical routing records stay in the event."""
from dataclasses import dataclass, field, replace
import torch
from .records import Atom


@dataclass(frozen=True)
class SourceInput:
    slot: int
    atom: Atom
    scale: torch.Tensor


@dataclass(frozen=True)
class Content:
    value: torch.Tensor
    sources: tuple[SourceInput, ...] = ()
    contributions: dict[int, torch.Tensor] = field(default_factory=dict)

    def with_value(self, value):
        return replace(self, value=value)


def as_content(value):
    return value if isinstance(value, Content) else Content(value)
