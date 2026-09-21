"""Exact forward values with an independently constructed semantic autograd graph."""
import torch
from .records import State


class _SemanticValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def value(packed, reference):
    if (packed.shape, packed.dtype, packed.device) != (reference.shape, reference.dtype, reference.device):
        raise ValueError("batching contract changed tensor metadata")
    if not reference.requires_grad:
        return packed.detach()
    return _SemanticValue.apply(reference, packed.detach())


def state(packed, reference):
    if (packed.last_time, packed.observations, set(packed.slots)) != (
            reference.last_time, reference.observations, set(reference.slots)):
        raise ValueError("state batching contract changed state metadata")
    return State(value(packed.value, reference.value), packed.last_time, packed.observations,
                 {name: value(t, reference.slots[name]) for name, t in packed.slots.items()})
