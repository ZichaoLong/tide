"""Nonempty event segments. Padding is absent, so it cannot create candidates."""
from dataclasses import dataclass
from . import autograd
from .content import Content
import torch


@dataclass
class PackedSequence:
    contents: torch.Tensor  # [sum(lengths), width]
    offsets: list[int]      # segment boundaries, starting at zero
    owners: list[tuple[int, int]]  # (sample,node), one common node/program
    times: list[int]
    views: list[Content]   # program-visible sources and Aggregate contributions

    def validate(self):
        if (self.contents.ndim != 2 or not self.owners or not self.offsets or self.offsets[0] != 0
                or len(self.offsets) != len(self.owners) + 1
                or self.offsets[-1] != len(self.contents)
                or len(self.times) != len(self.contents) or len(self.views) != len(self.contents)
                or any(not isinstance(c, Content) for c in self.views)
                or any((c.value.shape, c.value.dtype, c.value.device) != (
                    self.contents.shape[1:], self.contents.dtype, self.contents.device) for c in self.views)
                or len(set(self.owners)) != len(self.owners)
                or any(b < 0 or n != self.owners[0][1] or n < 0 for b, n in self.owners)):
            raise ValueError("invalid packed sequence metadata")
        for a, b in zip(self.offsets, self.offsets[1:]):
            if not 0 <= a < b <= len(self.contents) or any(t < 0 for t in self.times[a:b]) or any(
                    x >= y for x, y in zip(self.times[a:b-1], self.times[a+1:b])):
                raise ValueError("packed segments require nonempty, ordered events")


def prepare_sequences(weights, sequences, q):
    """One node's independent sample sequences; scalar kernels remain an explicit fallback."""
    owners, old, flat, offsets = [], [], [], [0]
    for owner, events in sequences:
        owners.append(owner); old.append(q.states.get(owner, weights.initial()))
        flat.extend(events); offsets.append(len(flat))
    replay = torch.is_grad_enabled()
    with torch.no_grad():
        batch = PackedSequence(torch.stack([e["content"] for e in flat]), offsets, owners,
                               [e["time"] for e in flat], [e["_content"] for e in flat])
        batch.validate()
        if hasattr(getattr(weights, "kernel", None), "packed_sequence"):
            states, calls = weights.kernel.packed_sequence(weights, old, batch)
        else:
            states = []
            for i, state in enumerate(old):
                a, b = offsets[i:i+2]
                ss, _ = weights.prepare_block(state, batch.contents[a:b], batch.times[a:b], batch.views[a:b])
                states.extend(ss)
            calls = len(old)
        if len(states) != len(flat):
            raise ValueError("packed state program changed event count")
        previous = []
        for i, state in enumerate(old):
            a, b = offsets[i:i+2]
            previous.extend([state, *states[a:b-1]])
        descriptors = weights.describe_batch(previous, states, batch)
    if replay:
        for i, previous in enumerate(old):
            a, b = offsets[i:i+2]
            for j in range(a, b):
                event = flat[j]
                proposed = weights.propose(previous, event["_content"], event["time"])
                states[j] = autograd.state(states[j], proposed)
                descriptor = weights.describe(previous, states[j], event["_content"], event["time"])
                flat[j]["semantic_descriptor"] = autograd.value(descriptors[j], descriptor)
                previous = states[j]
    for e, state, descriptor in zip(flat, states, descriptors):
        e.update(proposal_state=state, proposal=state.value, descriptor=e.pop("semantic_descriptor", descriptor))
    return calls
