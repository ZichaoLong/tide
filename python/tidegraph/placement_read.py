"""Read placement with one transfer per legal numeric node-time batch."""
import torch
from .readout import ReadProgram, LinearRead, NormRead, NormFloat32Read


class PlacedRead(ReadProgram):
    joint_batch = True

    def __init__(self, spec, original, dtype, device):
        super().__init__()
        if type(original) not in {LinearRead, NormRead, NormFloat32Read, PlacedRead} or original.profile != spec.readout:
            raise ValueError("placement cannot replace a custom or mismatched Read program")
        if isinstance(original, (LinearRead, PlacedRead)) and original.identity != spec.identity:
            raise ValueError("placement cannot replace a mismatched identity Read program")
        self.profile, self.identity = spec.readout, spec.identity
        self.precision = str(dtype).split(".")[-1] if dtype in {torch.float32, torch.float64} else "payload"
        self.dtype, self.device = dtype, device

    def descriptor_device(self, payload):
        return self.device

    def _calculate(self, weights, values):
        values = values.to(device=self.device, dtype=self.dtype)
        if self.profile == "linear-v1":
            return (values * weights.read.to(device=self.device, dtype=self.dtype)).sum(-1)
        return torch.linalg.vector_norm(values, ord=2, dim=-1)

    def step(self, weights, r):
        if self.identity:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return self._calculate(weights, r.content.value if r.state is None else r.state.value)

    def batch(self, weights, requests):
        if self.identity:
            return [self.step(weights, r) for r in requests]
        if not requests:
            return []
        values = torch.stack([r.content.value if r.state is None else r.state.value for r in requests])
        return list(self._calculate(weights, values).unbind())
