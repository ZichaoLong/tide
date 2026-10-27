"""Independent scalar CPU oracle for resident FP16 storage / FP32 arithmetic.

Only textbook State/Full boundaries gain an explicit half store with identity
FP32 VJP. The Python time-major reference owns its schedule. No resident math,
packing, saved journal or decoder is used to compute expected results.
"""
import copy
from dataclasses import fields, is_dataclass, replace
from types import MethodType
import torch
from tidegraph.compare import equivalent


class Store(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        value = x.half().float()
        if not torch.isfinite(value).all():
            raise ValueError("reference half store overflow")
        return value

    @staticmethod
    def backward(ctx, bar):
        return bar


def store_state(state):
    return replace(state, value=Store.apply(state.value), slots={
        name: Store.apply(x) if x.is_floating_point() else x for name, x in state.slots.items()})


def reference_model(source):
    model = copy.deepcopy(source).float()
    for node in model.nodes:
        memory, full = node.kernel, node.full_program
        initial, step, emit = memory.initial, memory.step, full.step

        def initial_store(self, weights, initial=initial):
            return store_state(initial(weights))

        def step_store(self, weights, old, content, time, step=step):
            return store_state(step(weights, old, content, time))

        def payload_store(self, weights, request, slots, mode, zeta, emit=emit):
            result = emit(weights, request, slots, mode, zeta)
            memo = {}

            def cast(x):
                if id(x) not in memo:
                    value = Store.apply(x)
                    if x[0] > 0 and value[0] <= 0:
                        raise ValueError("reference half support underflow")
                    memo[id(x)] = value
                return memo[id(x)]

            return replace(result, value=cast(result.value),
                           emitted={slot: cast(x) for slot, x in result.emitted.items()})

        # Preserve installed types, configuration and parameter names/aliases.
        memory.initial = MethodType(initial_store, memory)
        memory.step = MethodType(step_store, memory)
        full.step = MethodType(payload_store, full)
    return model


def widen(value):
    """Lossless tensor encoding of complete records; does not evaluate a formula."""
    if isinstance(value, torch.Tensor):
        return value.float() if value.dtype == torch.float16 else value
    if is_dataclass(value):
        return replace(value, **{f.name: widen(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {key: widen(x) for key, x in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(widen(x) for x in value)
    return value


def assert_records(actual, expected, dtype):
    """Check all declared storage types, then all values/metadata losslessly."""
    def typed(x, expected_type):
        assert x.dtype == expected_type

    def slots(xs):
        for x in xs.values():
            typed(x, dtype if x.is_floating_point() else torch.int64)

    for state in actual.continuation.states.values():
        typed(state.value, dtype)
        slots(state.slots)
    for x in (*actual.messages, *actual.continuation.pending):
        typed(x.value, dtype)
    for *_, x in actual.outputs:
        typed(x, dtype)
    for event in actual.trace:
        for atom in event["fiber"]:
            typed(atom.value, dtype)
        for name in ("proposal", "comparison", "next"):
            typed(event[name], dtype)
            slots(event[name+"_slots"])
        typed(event["content"], torch.float32)
        typed(event["control"], torch.float32)
        for x in event["descriptor"].values():
            typed(x, torch.float32)
        if event["active"]:
            typed(event["full"], dtype)
            for x in event["emitted"].values():
                typed(x, dtype)
    tolerance = dict(atol=1e-3, rtol=2e-2) if dtype == torch.float16 else {}
    equivalent(widen(actual), expected, **tolerance)
