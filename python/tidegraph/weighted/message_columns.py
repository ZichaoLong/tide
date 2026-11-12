"""Architecture §1–2: actual message columns and physical-atom aggregation.

Rows identify existing messages, including zero-strength input messages. Source
classes select parameters; repeated edges/ports remain repeated atoms. The host
uses only int64 support metadata. No differentiable row views cross a module.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MessageColumns:
    rows: torch.Tensor
    strength: torch.Tensor
    content: torch.Tensor

    def validate(self, domain, width, like, *, allow_remote=False):
        domain.select(self.rows)
        if self.content.shape != (len(self.rows), width) or self.strength.shape != (len(self.rows),):
            raise ValueError("message column dimensions differ from their binding")
        for x in (self.strength, self.content):
            device_ok = x.device == self.content.device and (x.device == like.device or
                allow_remote and x.device.type in {"cpu", "cuda", "npu"})
            if x.dtype != like.dtype or not device_ok or not torch.isfinite(x).all():
                raise ValueError("message columns require finite matching payload tensors")


def aggregate_columns(program, sources):
    """Union of actual fibers → (global row IDs, received mass, mean content).

    A source is (MessageColumns, SrcClass). No numerical-zero test removes an
    atom. An all-zero fiber has a constant zero mean, with its mass roots kept.
    """
    from .column_aggregate import aggregate_columns_many
    return aggregate_columns_many([(program, sources)])[0]


def readout_columns(outputs, domain, like):
    """§4.4 numerical projection; output existence remains in MessageColumns."""
    mass = like.new_zeros(len(domain.times))
    value = like.new_zeros((len(domain.times), like.shape[-1]))
    from .column_transport import message
    cache = {}
    for payload in outputs.values():
        payload = message(payload, like, cache)
        rows = payload.rows.to(like.device)
        strength = payload.strength.clamp_min(0)
        mass = mass.index_add(0, rows, strength)
        value = value.index_add(0, rows, strength[:, None]*payload.content)
    return value/mass.clamp_min(1)[:, None]
