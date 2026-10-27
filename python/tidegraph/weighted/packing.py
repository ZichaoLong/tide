"""Complete local drivers → positive-write segments, with exact int64 clocks.

Batch §2 and recurrent §4. This is eager packing after the scheduler certifies
drivers; it neither predicts routes nor constructs a reference trajectory.
"""
import torch
from ..history import increment
from ..records import State
from .tensors import Received


def project(x, weight):
    """Batch the textbook column-vector projections without reorienting them."""
    return (weight@x.t()).t()


def sequence(memory, weights, old, values, times, views):
    from ..content import Content
    from ..packing import PackedSequence
    if not times:
        return []
    views = [Content(h) for h in values] if views is None else views
    batch = PackedSequence(values, [0, len(times)], [(0, 0)], list(times), views)
    return memory.packed_sequence(weights, [old], batch)[0]


class Writes:
    def __init__(self, old, batch):
        batch.validate()
        if len(old) != len(batch.owners):
            raise ValueError("packed initial-state count mismatch")
        h = Received.decode(batch.contents)
        self.positive = (h.mass > 0).tolist()
        self.ids = [j for j, yes in enumerate(self.positive) if yes]
        self.index = torch.tensor(self.ids, device=h.mass.device, dtype=torch.int64)
        self.times = torch.tensor([batch.times[j] for j in self.ids], device=h.mass.device, dtype=torch.int64)
        self.weight = h.mass[self.index].clamp(max=1)
        self.content = h.content[self.index]
        self.offsets, self.groups = [0], {}
        for i, state in enumerate(old):
            a, b = batch.offsets[i:i+2]
            n = sum(self.positive[a:b])
            if n:
                first = next(j for j in range(a, b) if self.positive[j])
                if batch.times[first] <= state.last_time:
                    raise ValueError("packed write times must strictly increase")
                increment(state.observations, n)
                self.groups.setdefault(n, []).append(i)
            self.offsets.append(self.offsets[-1]+n)

    def decode(self, old, batch, slots):
        """Publish computed positive states; zero receives retain the raw tuple."""
        states = []
        for i, initial in enumerate(old):
            previous, count = initial, 0
            for j in range(*batch.offsets[i:i+2]):
                if self.positive[j]:
                    k = self.offsets[i]+count
                    count += 1
                    previous = State(initial.value, batch.times[j], increment(initial.observations, count), slots[k])
                states.append(previous)
        return states

    def group_index(self, length, owners):
        return torch.tensor([[self.offsets[i]+j for i in owners] for j in range(length)],
                            device=self.times.device, dtype=torch.int64)

    def gaps(self, old):
        gaps = []
        for i, state in enumerate(old):
            a, b = self.offsets[i:i+2]
            if a < b:
                times = self.times[a:b]
                previous = torch.cat((state.slots["time"].reshape(1), times[:-1]))
                gaps.append((times-previous).clamp_min(0))
        return torch.cat(gaps)  # Subtract in int64 before any floating conversion.


def matrix_scan(a, b, initial):
    """Recurrent §4.1: ordered composition of H → H A + B, no step replay."""
    stride = 1
    while stride < len(b):
        b = torch.cat((b[:stride], b[:-stride]@a[stride:]+b[stride:]))
        a = torch.cat((a[:stride], a[:-stride]@a[stride:]))
        stride *= 2
    return initial@a+b
