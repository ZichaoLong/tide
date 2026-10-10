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


def _contents(values, ids):
    # Static row positions only, never input values or an oracle trajectory.
    if len(ids) == len(values) and ids == list(range(len(values))):
        return values
    return torch.stack([values[i] for i in ids])


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
            pairs = [(i, start+depth) for i, start in enumerate(batch.offsets[:-1])
                     if start+depth < batch.offsets[i+1]]
            if not pairs:
                return states, calls
            values, count = self._batch_rows(weights, [current[i] for i, j in pairs],
                _contents(batch.contents, [j for i, j in pairs]), [batch.times[j] for i, j in pairs], independent=True)
            for (i, j), value in zip(pairs, values):
                current[i] = states[j] = value
            calls += count
            depth += 1

    def _batch_rows(self, weights, old, contents, times, *, independent):
        """Internal structural VJP graphs may use views; published states own storage."""
        groups, states = {}, [None]*len(old)
        for i, (state, time) in enumerate(zip(old, times)):
            if not int64(time) or not int64(state.last_time) or not -1 <= state.last_time < time:
                raise ValueError("Add requires strictly increasing nonnegative tick times")
            groups.setdefault(time-state.last_time, []).append(i)
        for ticks, ids in groups.items():
            previous = torch.stack([old[i].value for i in ids])
            values = _contents(contents, ids) + repeat(previous, weights.extra["add_retention"], ticks)
            for i, value in zip(ids, values.unbind()):
                states[i] = State(value.clone() if independent else value, times[i], increment(old[i].observations))
        return states, len(groups)

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
