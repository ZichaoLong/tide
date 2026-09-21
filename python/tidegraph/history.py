"""Region-owned value records; dictionaries copy, tensor graphs remain connected."""
from dataclasses import dataclass, field, replace
import torch


def int64(value):
    return type(value) is int and -2**63 <= value < 2**63


def increment(value, amount=1):
    if not int64(value) or not int64(amount) or value < 0 or amount < 0 or value > 2**63-1-amount:
        raise ValueError("int64 counter overflow or invalid counter")
    return value + amount


@dataclass
class History:
    last_time: int = -1
    scalars: dict[str, int] = field(default_factory=dict)
    node_maps: dict[str, dict[int, int]] = field(default_factory=dict)
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def fork(self):
        return replace(self, scalars=dict(self.scalars), node_maps={k: dict(v) for k, v in self.node_maps.items()},
                       tensors=dict(self.tensors))

    def detach(self):
        return replace(self.fork(), tensors={k: v.detach() for k, v in self.tensors.items()})


def validate(history, layout, reference, time):
    if not isinstance(history, History) or not int64(history.last_time) or not -1 <= history.last_time <= time:
        raise ValueError("invalid region history clock")
    for fields in (history.scalars, history.node_maps, history.tensors):
        if not isinstance(fields, dict) or any(not isinstance(k, str) or not k for k in fields):
            raise ValueError("invalid region history field names")
    if any(not int64(v) for v in history.scalars.values()):
        raise ValueError("invalid region history int64 scalar")
    for counts in history.node_maps.values():
        if not isinstance(counts, dict) or any(type(v) is not int or v not in layout.slots or not int64(c)
                                              for v, c in counts.items()):
            raise ValueError("invalid region history node map")
    for value in history.tensors.values():
        if not isinstance(value, torch.Tensor) or (value.dtype, value.device) != (reference.dtype, reference.device):
            raise ValueError("incompatible region history tensor dtype/device")
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite region history tensor")
