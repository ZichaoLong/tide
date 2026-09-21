"""Mode-restricted Read programs, independent of the memory update formula."""
from dataclasses import dataclass
import torch
from . import autograd
from .content import Content
from .records import State


@dataclass(frozen=True)
class ReadInput:
    state: State | None
    time: int
    content: Content


def request(mode, old, proposal, content, time):
    if mode not in {"content", "old", "proposal"}:
        raise ValueError("invalid region Read mode")
    state = None if mode == "content" else old if mode == "old" else proposal
    return ReadInput(state, time, content)


class ReadProgram(torch.nn.Module):
    profile = "custom"
    joint_batch = False

    def step(self, weights, request):
        raise NotImplementedError

    def batch(self, weights, requests):
        return [self.step(weights, r) for r in requests]


class LinearRead(ReadProgram):
    profile = "linear-v1"
    joint_batch = True

    def __init__(self, identity=False):
        super().__init__()
        self.identity = identity

    def step(self, w, r):
        if self.identity:
            return r.content.value.new_zeros(())
        value = r.content.value if r.state is None else r.state.value
        return (value * w.read).sum(-1)

    def batch(self, w, requests):
        if self.identity:
            return [r.content.value.new_zeros(()) for r in requests]
        values = torch.stack([r.content.value if r.state is None else r.state.value for r in requests])
        return list((values * w.read).sum(-1).unbind())


def validate_program(weights, spec, *, native=False):
    program = weights.read_program
    if native and type(program) is not LinearRead:
        raise ValueError("Python custom Read has no native implementation")
    if not isinstance(program, ReadProgram) or program.profile != spec.readout:
        raise ValueError("Read program does not match graph profile")
    if isinstance(program, LinearRead) and program.identity != spec.identity:
        raise ValueError("shared Read program does not match identity policy")


def validate(value, r):
    ref = r.content.value
    if not isinstance(value, torch.Tensor) or (value.shape, value.dtype, value.device) != (
            torch.Size([]), ref.dtype, ref.device):
        raise ValueError("Read returned incompatible scalar metadata")
    if not torch.isfinite(value):
        raise ValueError("nonfinite selector score from Read")


def evaluate(weights, requests, *, packed=False):
    program = weights.read_program
    if packed:
        with torch.no_grad():
            values = program.batch(weights, requests)
    else:
        values = [program.step(weights, r) for r in requests]
    if len(values) != len(requests):
        raise ValueError("Read batch changed event count")
    for value, r in zip(values, requests):
        validate(value, r)
    if packed and torch.is_grad_enabled():
        semantic = [program.step(weights, r) for r in requests]
        for value, r in zip(semantic, requests):
            validate(value, r)
        values = [autograd.value(a, b) for a, b in zip(values, semantic)]
    return values
