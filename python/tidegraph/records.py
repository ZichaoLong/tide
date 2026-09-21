"""Tensor-bearing event records. Copies preserve autograd unless detached."""
from dataclasses import dataclass, field, replace
import torch


@dataclass
class State:
    value: torch.Tensor
    last_time: int = -1
    observations: int = 0
    slots: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class External:
    batch: int
    port: int
    position: int
    time: int
    value: torch.Tensor


@dataclass
class Atom:
    batch: int
    node: int
    time: int
    kind: int  # 0 external port, 1 internal edge
    source: int
    position: int  # external position or message send time
    value: torch.Tensor

    def key(self):
        return self.batch, self.node, self.time, self.kind, self.source, self.position


@dataclass
class Continuation:
    identity: str
    batch_size: int
    cut: int = 0
    states: dict[tuple[int, int], State] = field(default_factory=dict)
    history: dict[tuple[int, int], dict[int, int]] = field(default_factory=dict)
    pending: list[Atom] = field(default_factory=list)
    ledger: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)

    def fork(self):
        return Continuation(self.identity, self.batch_size, self.cut, dict(self.states),
                            {k: dict(v) for k, v in self.history.items()},
                            list(self.pending), dict(self.ledger))

    def detach(self):
        """Explicit TBPTT boundary: detach state and every in-flight message."""
        q = self.fork()
        q.states = {owner: replace(state, value=state.value.detach(),
                                  slots={k: v.detach() for k, v in state.slots.items()})
                    for owner, state in self.states.items()}
        q.pending = [replace(atom, value=atom.value.detach()) for atom in self.pending]
        return q


@dataclass
class Result:
    continuation: Continuation
    trace: list[dict]
    outputs: list[tuple[int, int, int, torch.Tensor]]
    messages: list[Atom]
    stats: dict[str, int]


@dataclass
class AdvanceResult:
    """Only this window's records; request cursor.snapshot() for complete state."""
    cut: int
    trace: list
    outputs: list
    messages: list
    stats: dict
