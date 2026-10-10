"""LH-compatible tick decay with explicit repeated-multiplication order."""
import torch
from .content import as_content
from .history import increment, int64
from .records import State
from .state_program import StateProgram


def repeat(value, retention, ticks):
    # No pow, affine scan or zero-value shortcut: rounding and zero VJPs matter.
    for _ in range(ticks):
        value = value * retention
    return value


class LazyAdd(StateProgram):
    profile = "lh-add-repeat-v1"
    sequence_contract = True
    joint_sequence = True  # Ordered time recurrence, independent sample batches.

    def step(self, weights, old, content, time):
        if not int64(time) or not int64(old.last_time) or not -1 <= old.last_time < time:
            raise ValueError("Add requires strictly increasing nonnegative tick times")
        observations = increment(old.observations)
        value = repeat(old.value, weights.extra["add_retention"], time - old.last_time)
        return State(as_content(content).value + value, time, observations)

    def validate_weights(self, weights):
        rho = weights.extra.get("add_retention")
        if (rho is None or rho.ndim != 0 or rho.device.type not in {"cpu", "cuda", "npu"}
                or rho.dtype not in (torch.float16, torch.float32, torch.float64)
                or (rho.dtype, rho.device) != (weights.bias.dtype, weights.bias.device)
                or not torch.isfinite(rho)):
            raise ValueError("Add requires a finite payload-dtype scalar retention")

    def packed_sequence(self, weights, old, batch):
        batch.validate()
        if len(old) != len(batch.owners):
            raise ValueError("packed initial-state count mismatch")
        current, states, depth, calls = list(old), [None]*len(batch.times), 0, 0
        while True:
            groups = {}
            for i, start in enumerate(batch.offsets[:-1]):
                j = start+depth
                if j < batch.offsets[i+1]:
                    time, last = batch.times[j], current[i].last_time
                    if not int64(last) or not -1 <= last < time:
                        raise ValueError("Add requires strictly increasing nonnegative tick times")
                    groups.setdefault(time-last, []).append((i, j))
            if not groups:
                return states, calls
            for ticks, pairs in groups.items():
                previous = torch.stack([current[i].value for i, j in pairs])
                h = torch.stack([batch.contents[j] for i, j in pairs])
                values = h + repeat(previous, weights.extra["add_retention"], ticks)
                for (i, j), value in zip(pairs, values.unbind()):
                    current[i] = states[j] = State(value.clone(), batch.times[j], increment(current[i].observations))
                calls += 1
            depth += 1

    def validate(self, weights, state):
        if state.slots:
            raise ValueError("Add state has unexpected slots")


def decode(weights, state, cut):
    """Physical hidden after ticks [0,cut); no mutation or autonomous event.

    Only this explicit read incurs work for idle ticks. The stored last_time and
    observation count remain unchanged. Parameters must stay fixed for an eager
    LH interpretation across composed windows.
    """
    from .clocked_state import ClockedState
    if isinstance(weights.kernel, ClockedState):
        state = weights.kernel.clock.local_state(state)
        cut = weights.kernel.clock.cut(cut)
    LazyAdd().validate_weights(weights)
    LazyAdd().validate(weights, state)
    if (not int64(cut) or not int64(state.last_time) or not int64(state.observations)
            or not -1 <= state.last_time < cut or state.observations < 0):
        raise ValueError("invalid Add cut/state clock")
    if (state.value.shape != weights.bias.shape or state.value.ndim != 1
            or (state.value.dtype, state.value.device) != (weights.bias.dtype, weights.bias.device)
            or not torch.isfinite(state.value).all()):
        raise ValueError("incompatible Add state value")
    return repeat(state.value, weights.extra["add_retention"], cut - 1 - state.last_time)
