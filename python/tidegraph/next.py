"""Persistent transitions precede Full; graph-owned clear policy survives sharing."""
from dataclasses import dataclass, replace
import torch
from .content import Content
from .records import State


@dataclass(frozen=True)
class NextInput:
    old: State
    comparison: State
    time: int
    content: Content
    active: bool
    control: torch.Tensor


class NextProgram(torch.nn.Module):
    profile = "custom"
    comparison_identity = False

    def step(self, weights, request):
        raise NotImplementedError


class AdoptNext(NextProgram):
    profile = "adopt-v1"
    comparison_identity = True

    def step(self, weights, request):
        return request.comparison


class ControlBlendNext(NextProgram):
    profile = "control-blend-v1"

    def step(self, weights, r):
        a, b, c = r.old, r.comparison, r.control
        if a.slots.keys() != b.slots.keys() or any(a.slots[k].shape != b.slots[k].shape for k in a.slots):
            raise ValueError("control-blend Next requires matching slot shapes")
        return replace(b, value=(1-c)*a.value+c*b.value,
                       slots={k: (1-c)*a.slots[k]+c*b.slots[k] for k in a.slots})


def program(profile):
    if profile == AdoptNext.profile:
        return AdoptNext()
    if profile == ControlBlendNext.profile:
        return ControlBlendNext()
    raise ValueError("unknown Next profile")


def validate_program(weights, spec, *, native=False):
    p = weights.next_program
    if native and type(p) not in {AdoptNext, ControlBlendNext}:
        raise ValueError("Python custom Next has no native implementation")
    if not isinstance(p, NextProgram) or p.profile != spec.next_state:
        raise ValueError("Next program does not match graph profile")


def validate(state, weights, time):
    if (not isinstance(state, State) or type(state.last_time) is not int or type(state.observations) is not int
            or not -1 <= state.last_time <= time or not 0 <= state.observations < 2**63):
        raise ValueError("Next returned invalid state clocks")
    ref = weights.bias
    if not isinstance(state.value, torch.Tensor) or state.value.shape != ref.shape:
        raise ValueError("Next returned incompatible state shape")
    if not isinstance(state.slots, dict) or any(not isinstance(k, str) for k in state.slots):
        raise ValueError("Next returned invalid state slots")
    for value in [state.value, *state.slots.values()]:
        if not isinstance(value, torch.Tensor) or (value.dtype, value.device) != (ref.dtype, ref.device):
            raise ValueError("Next returned incompatible state dtype/device")
        if not torch.isfinite(value).all():
            raise ValueError("Next returned nonfinite state")
    weights.validate(state)


def evaluate(weights, spec, request):
    result = weights.next_program.step(weights, request)
    # The identity contract is also a validated-input fast path. Custom programs
    # opting in promise to return comparison, including its slots and clocks.
    if not weights.next_program.comparison_identity:
        validate(result, weights, request.time)
    if spec.clear and request.active:
        result = weights.reset(result)
    return result
