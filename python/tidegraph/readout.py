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
    precision = "payload"

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


class NormRead(ReadProgram):
    profile = "norm-fp64-v1"
    precision = "float64"
    joint_batch = True

    def step(self, w, r):
        value = r.content.value if r.state is None else r.state.value
        return torch.linalg.vector_norm(value, ord=2, dim=-1, dtype=torch.float64)

    def batch(self, w, requests):
        values = torch.stack([r.content.value if r.state is None else r.state.value for r in requests])
        return list(torch.linalg.vector_norm(values, ord=2, dim=-1, dtype=torch.float64).unbind())


class NormFloat32Read(ReadProgram):
    """Explicit FP32 norm, including conversion before reduction and its VJP."""
    profile = "norm-fp32-v1"
    precision = "float32"
    joint_batch = True

    def step(self, w, r):
        value = r.content.value if r.state is None else r.state.value
        return torch.linalg.vector_norm(value.to(torch.float32), ord=2, dim=-1)

    def batch(self, w, requests):
        values = torch.stack([r.content.value if r.state is None else r.state.value for r in requests])
        return list(torch.linalg.vector_norm(values.to(torch.float32), ord=2, dim=-1).unbind())


def program(profile, identity=False):
    if profile == LinearRead.profile:
        return LinearRead(identity)
    if profile == NormRead.profile and not identity:
        return NormRead()
    if profile == NormFloat32Read.profile and not identity:
        return NormFloat32Read()
    raise ValueError("unknown Read profile")


def validate_program(weights, spec, *, native=False):
    program = weights.read_program
    if native and type(program) not in {LinearRead, NormRead, NormFloat32Read}:
        raise ValueError("Python custom Read has no native implementation")
    if not isinstance(program, ReadProgram) or program.profile != spec.readout:
        raise ValueError("Read program does not match graph profile")
    if isinstance(program, LinearRead) and program.identity != spec.identity:
        raise ValueError("shared Read program does not match identity policy")
    if (program.precision not in {"payload", "float32", "float64"}
            or isinstance(program, LinearRead) and program.precision != "payload"
            or isinstance(program, NormRead) and program.precision != "float64"
            or isinstance(program, NormFloat32Read) and program.precision != "float32"):
        raise ValueError("invalid Read precision policy")


def validate(value, r, precision="payload"):
    ref = r.content.value
    choices = {"payload": ref.dtype, "float32": torch.float32, "float64": torch.float64}
    if precision not in choices:
        raise ValueError("invalid Read precision policy")
    dtype = choices[precision]
    if not isinstance(value, torch.Tensor) or (value.shape, value.dtype, value.device) != (
            torch.Size([]), dtype, ref.device):
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
        validate(value, r, program.precision)
    if packed and torch.is_grad_enabled():
        semantic = [program.step(weights, r) for r in requests]
        for value, r in zip(semantic, requests):
            validate(value, r, program.precision)
        values = [autograd.value(a, b) for a, b in zip(values, semantic)]
    return values
