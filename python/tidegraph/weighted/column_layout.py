"""Host window metadata: ragged samples and original int64 logical times.

This is a host-scheduled batch contract, not a resident-ready queue. Float
fields live on their declared payload device; metadata is never packed as float.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SequenceLayout:
    offsets: torch.Tensor
    times: torch.Tensor

    def __post_init__(self):
        for t in (self.offsets, self.times):
            if t.ndim != 1 or t.dtype != torch.int64 or t.device.type != "cpu":
                raise ValueError("sequence layout requires host int64 vectors")
        if len(self.offsets) < 2 or self.offsets[0] != 0 or self.offsets[-1] != len(self.times):
            raise ValueError("sequence offsets do not cover rows")
        if (self.offsets[1:] < self.offsets[:-1]).any() or (self.times < 0).any():
            raise ValueError("invalid sequence offsets/times")
        for a, b in self.spans:
            if b-a > 1 and (self.times[a+1:b] <= self.times[a:b-1]).any():
                raise ValueError("logical times must strictly increase per sample")

    @property
    def spans(self):
        xs = self.offsets.tolist()
        return tuple(zip(xs, xs[1:]))

    @property
    def samples(self):
        return self.offsets.numel()-1

    def owners(self):
        return torch.repeat_interleave(torch.arange(self.samples), self.offsets.diff())

    def select(self, rows):
        if rows.ndim != 1 or rows.dtype != torch.int64 or rows.device.type != "cpu":
            raise ValueError("column row IDs must be host int64")
        if len(rows) and (rows[0] < 0 or rows[-1] >= len(self.times) or (rows[1:] <= rows[:-1]).any()):
            raise ValueError("column row IDs must be sorted, unique and in range")
        counts = torch.bincount(self.owners()[rows], minlength=self.samples)
        return SequenceLayout(torch.cat((counts.new_zeros(1), counts.cumsum(0))), self.times[rows])

    def padded(self, device, *, minimum=1):
        """Actual row indices and one sentinel; no floating event views."""
        width = max(minimum, int(self.offsets.diff().max()))
        rows = torch.full((self.samples, width), len(self.times), dtype=torch.int64)
        reverse = []
        for owner, (a, b) in enumerate(self.spans):
            rows[owner, :b-a] = torch.arange(a, b)
            reverse.extend(range(owner*width, owner*width+b-a))
        return rows.to(device), torch.tensor(reverse, dtype=torch.int64, device=device)


def gather_column(value, indices):
    """A single gather/transpose for the complete column, with a zero sentinel."""
    padding = value.new_zeros((1, *value.shape[1:]))
    return torch.cat((value, padding)).index_select(0, indices.flatten()).reshape(*indices.shape, *value.shape[1:])
