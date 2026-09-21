"""Change a state program's time argument while storing global timestamps."""
from dataclasses import replace
from .clocks import StateClock
from .state_program import StateProgram


class ClockedState(StateProgram):
    def __init__(self, program, clock):
        super().__init__()
        if type(clock) is not StateClock:
            raise ValueError("state clock requires a declarative StateClock policy")
        self.program, self.clock = program, clock
        self.profile = getattr(program, "profile", "custom")

    @property
    def sequence_contract(self):
        return self.program.sequence_contract

    @property
    def joint_sequence(self):
        return self.program.joint_sequence

    def initial(self, weights):
        return self.clock.global_state(self.program.initial(weights))

    def event(self, time):
        self.clock._index(time, 0)
        return self.clock.to_local(time)

    def step(self, weights, old, content, time):
        result = self.program.step(weights, self.clock.local_state(old), content, self.event(time))
        return self.clock.global_state(result)

    def sequence(self, weights, old, values, times, views=None):
        results = self.program.sequence(weights, self.clock.local_state(old), values,
                                        [self.event(t) for t in times], views)
        return [self.clock.global_state(state) for state in results]

    def packed_sequence(self, weights, old, batch):
        if not hasattr(self.program, "packed_sequence"):
            # Preserve the existing per-segment sequence fallback. The inherited
            # method calls our sequence with global clocks and reports each call.
            return super().packed_sequence(weights, old, batch)
        batch.validate()
        local = replace(batch, times=[self.event(t) for t in batch.times])
        states, calls = self.program.packed_sequence(weights, [self.clock.local_state(s) for s in old], local)
        return [self.clock.global_state(state) for state in states], calls

    def reset(self, state):
        from .memory import reset
        result = getattr(self.program, "reset", reset)(self.clock.local_state(state))
        return self.clock.global_state(result)

    def validate(self, weights, state):
        self.program.validate(weights, self.clock.local_state(state))


def with_clock(program, clock):
    if type(clock) is not StateClock:
        raise ValueError("state clock requires a declarative StateClock policy")
    if isinstance(program, ClockedState):
        if program.clock != clock:
            raise ValueError("shared state program does not match state clock")
        return program
    return program if clock == StateClock() else ClockedState(program, clock)
