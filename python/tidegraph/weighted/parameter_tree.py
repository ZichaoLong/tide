"""Private module copies for first-order VJP leaves, preserving parameter ties.

No live module is mutated. Configuration/state layout stays immutable; only
registered tensors are rebound. Every distinct registered tensor is captured.
"""
from copy import copy


def rebind(module, transform):
    modules, tensors = {}, {}

    def tensor(value):
        if value is None:
            return None
        if id(value) not in tensors:
            tensors[id(value)] = transform(value)
        return tensors[id(value)]

    def visit(original):
        if original is None:
            return None
        if id(original) in modules:
            return modules[id(original)]
        result = copy(original)
        modules[id(original)] = result
        result._parameters = {name: tensor(value) for name, value in sorted(original._parameters.items())}
        result._buffers = {name: tensor(value) for name, value in sorted(original._buffers.items())}
        result._modules = {name: visit(value) for name, value in sorted(original._modules.items())}
        return result

    return visit(module)


def supported(kernel):
    from ..clocked_state import ClockedState
    from .kv import KVMemory
    from .recurrent import BasicMemory
    from .gated_delta import GatedDeltaMemory
    from .ssd import SSDMemory
    if type(kernel) is ClockedState:
        return supported(kernel.program)
    return type(kernel) in {KVMemory, BasicMemory, GatedDeltaMemory, SSDMemory}
