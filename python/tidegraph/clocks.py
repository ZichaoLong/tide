"""Periodic local state clocks with a contiguous valid phase interval."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class StateClock:
    period: int = 1
    first: int = 0
    count: int = 1

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.period, self.first, self.count))
                or not 0 < self.period < 2**63 or not 0 <= self.first < self.period
                or not 0 < self.count <= self.period-self.first):
            raise ValueError("invalid periodic state clock")

    @staticmethod
    def _index(value, minimum=-1):
        if type(value) is not int or not minimum <= value < 2**63:
            raise ValueError("invalid state clock coordinate")

    def to_local(self, time):
        self._index(time)
        if time == -1:
            return -1
        cycle, phase = divmod(time, self.period)
        if not self.first <= phase < self.first+self.count:
            raise ValueError("event is outside the state clock phases")
        return cycle*self.count + phase-self.first

    def to_global(self, time):
        self._index(time)
        if time == -1:
            return -1
        cycle, phase = divmod(time, self.count)
        result = cycle*self.period + self.first+phase
        if result >= 2**63:
            raise ValueError("state clock inverse overflow")
        return result

    def cut(self, cut):
        self._index(cut, 0)
        cycle, phase = divmod(cut, self.period)
        return cycle*self.count + min(max(phase-self.first, 0), self.count)

    def local_state(self, state):
        return replace(state, last_time=self.to_local(state.last_time))

    def global_state(self, state):
        return replace(state, last_time=self.to_global(state.last_time))
