"""Architecture §1–2: separate typed views of payload (m,z) and content (r,u)."""
from dataclasses import dataclass
import torch
from .normalization import norm


@dataclass(frozen=True)
class Payload:
    strength: torch.Tensor
    content: torch.Tensor

    def encode(self):
        if (self.strength.shape != self.content.shape[:-1]
                or (self.strength.dtype, self.strength.device) != (self.content.dtype, self.content.device)):
            raise ValueError("payload strength/content shape or dtype/device mismatch")
        return torch.cat((self.strength.unsqueeze(-1), self.content), -1)

    @classmethod
    def decode(cls, tensor):
        if tensor.ndim < 1 or tensor.shape[-1] < 2 or not tensor.is_floating_point():
            raise ValueError("invalid weighted payload encoding")
        return cls(tensor[..., 0], tensor[..., 1:])

    def numerical(self):
        mass = self.strength.clamp_min(0)
        return mass, mass.unsqueeze(-1)*self.content


@dataclass(frozen=True)
class Received:
    mass: torch.Tensor
    content: torch.Tensor

    def encode(self):
        return Payload(self.mass, self.content).encode()

    @classmethod
    def decode(cls, tensor):
        payload = Payload.decode(tensor)
        if (payload.strength < 0).any():
            raise ValueError("negative aggregate received mass")
        return cls(payload.strength, payload.content)


def parameter(shape, generator, dtype, scale=.15):
    return torch.nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype)*scale)


def rotate_relative(keys, displacement, frequencies):
    """qᵀ Rot(Pos(xi)-Pos(theta)) k; int64 subtraction precedes casting."""
    if not frequencies:
        return keys
    payload_dtype = keys.dtype
    if payload_dtype == torch.float16:
        keys, displacement = keys.float(), displacement.float()
    count = len(frequencies)
    phase = displacement.unsqueeze(-1)*keys.new_tensor(frequencies)
    pairs = keys[..., -2*count:].reshape(*keys.shape[:-1], count, 2)
    a, b = pairs.unbind(-1)
    rotated = torch.stack((a*phase.cos()-b*phase.sin(), a*phase.sin()+b*phase.cos()), -1)
    return torch.cat((keys[..., :-2*count], rotated.flatten(-2)), -1).to(payload_dtype)
