"""Registered custom state programs; exact batching is a declared capability."""
import torch
from .content import Content
from .records import State


class StateProgram(torch.nn.Module):
    profile = "custom"
    sequence_contract = False
    joint_sequence = False

    def initial(self, weights):
        return State(torch.zeros_like(weights.bias))

    def step(self, weights, old, content, time):
        raise NotImplementedError

    def sequence(self, weights, old, values, times, views=None):
        views = [Content(h) for h in values] if views is None else views
        states = []
        for h, time, content in zip(values, times, views):
            old = self.step(weights, old, content.with_value(h), time)
            states.append(old)
        return states

    def packed_sequence(self, weights, old, batch):
        batch.validate()
        if len(old) != len(batch.owners):
            raise ValueError("packed initial-state count mismatch")
        states = []
        for i, state in enumerate(old):
            a, b = batch.offsets[i:i+2]
            states.extend(self.sequence(weights, state, batch.contents[a:b], batch.times[a:b], batch.views[a:b]))
        return states, len(old)

    def validate(self, weights, state):
        raise NotImplementedError


def validate_program(weights, spec, *, native=False):
    if spec.identity:
        return
    from .memory import EMA, DiagonalSSM
    from .attention import Attention
    from .matrix_memory import MatrixMemory
    from .lazy_add import LazyAdd
    builtins = {"ema": EMA, "ssm": DiagonalSSM, "linear": MatrixMemory, "delta": MatrixMemory,
                "attention": Attention, LazyAdd.profile: LazyAdd}
    program = weights.kernel
    if type(program) is not builtins.get(spec.memory):
        if native:
            raise ValueError("Python custom state program has no native implementation")
        if not isinstance(program, StateProgram) or program.profile != spec.memory:
            raise ValueError("custom state program does not match graph profile")
        return
    if isinstance(program, MatrixMemory) and program.kind != spec.memory:
        raise ValueError("shared state program does not match graph profile")
    if isinstance(program, LazyAdd):
        program.validate_weights(weights)
    if isinstance(program, Attention) and (program.query_heads, program.kv_heads, program.window) != (
            spec.query_heads, spec.kv_heads, spec.window):
        raise ValueError("shared state program does not match attention policy")
