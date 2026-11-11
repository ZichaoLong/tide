"""Recurrent §1/4: typed matrix/summary state and sequence scans over B and L.

Private state differs from KV. Clocks and exact convolution slots remain int64;
only real positive writes enter scans. Query snapshots index actual scan outputs.
"""
from dataclasses import dataclass, replace
import torch
from ..records import State
from ..history import increment
from ..scan import affine_scan
from .packing import matrix_scan
from .column_layout import SequenceLayout, gather_column
from .numeric_time import passive_decay
from .tensors import norm


@dataclass(frozen=True)
class RecurrentStateColumns:
    H: torch.Tensor
    Z: torch.Tensor
    W: torch.Tensor
    inert: torch.Tensor
    last_times: torch.Tensor
    observations: torch.Tensor
    cache_layout: SequenceLayout | None = None
    cache_values: torch.Tensor | None = None

    @property
    def samples(self):
        return len(self.last_times)

    @classmethod
    def from_states(cls, states):
        if not states:
            raise ValueError("recurrent columns require samples")
        cache = None
        if "conv_times" in states[0].slots:
            offsets = [0]
            for s in states:
                offsets.append(offsets[-1]+len(s.slots["conv_times"]))
            cache = SequenceLayout(torch.tensor(offsets), torch.cat([s.slots["conv_times"].cpu() for s in states]))
        for s in states:
            if s.slots["time"].dtype != torch.int64 or s.slots["time"].shape != () or s.slots["time"] != max(0, s.last_time):
                raise ValueError("recurrent columns require the exact raw state clock")
        return cls(*(torch.stack([s.slots[n] for s in states]) for n in ("H", "Z", "W")),
            torch.stack([s.value for s in states]), torch.tensor([s.last_time for s in states]),
            torch.tensor([s.observations for s in states]), cache,
            None if cache is None else torch.cat([s.slots["conv_values"] for s in states]))

    def states(self):
        result = []
        for b in range(self.samples):
            slots = {n: getattr(self, n)[b] for n in ("H", "Z", "W")}
            slots["time"] = self.last_times[b].clamp_min(0).to(self.H.device)
            if self.cache_layout is not None:
                a, z = self.cache_layout.spans[b]
                slots.update(conv_times=self.cache_layout.times[a:z].to(self.H.device), conv_values=self.cache_values[a:z])
            result.append(State(self.inert[b], int(self.last_times[b]), int(self.observations[b]), slots))
        return result

    def detach(self):
        return replace(self, **{n: getattr(self, n).detach() for n in ("H", "Z", "W", "inert")},
                       cache_values=None if self.cache_values is None else self.cache_values.detach())

    def validate(self, config):
        b, c = self.samples, config
        for n, shape in dict(H=(b, *c.matrix_shape), Z=(b, 2, c.summary_width), W=(b, 2), inert=(b, c.width+1)).items():
            x = getattr(self, n)
            if x.shape != shape or (x.device, x.dtype) != (self.H.device, self.H.dtype) or not torch.isfinite(x).all():
                raise ValueError("invalid recurrent state column: "+n)
        for t in (self.last_times, self.observations):
            if t.shape != (b,) or t.dtype != torch.int64 or t.device.type != "cpu":
                raise ValueError("recurrent clocks/counts must be host int64")
        if ((self.last_times < -1).any() or (self.observations < 0).any()
                or ((self.last_times == -1) != (self.observations == 0)).any() or self.inert.count_nonzero() or (self.W < 0).any()):
            raise ValueError("invalid recurrent state clock/count/value")
        if c.basic:
            if self.cache_layout is not None or self.cache_values is not None:
                raise ValueError("basic recurrent state has no convolution cache")
        else:
            if self.cache_layout is None or self.cache_layout.samples != b:
                raise ValueError("recurrent convolution sample domain differs")
            n = len(self.cache_layout.times)
            if (self.cache_values is None or self.cache_values.shape != (n, c.conv_width)
                    or (self.cache_values.dtype, self.cache_values.device) != (self.H.dtype, self.H.device)
                    or not torch.isfinite(self.cache_values).all()):
                raise ValueError("invalid recurrent convolution columns")
            for owner, (a, z) in enumerate(self.cache_layout.spans):
                if z-a > min(int(self.observations[owner]), (c.conv_size-1)*c.conv_stride+1) or z>a and self.cache_layout.times[z-1]>self.last_times[owner]:
                    raise ValueError("invalid raw convolution times/counts")


@dataclass(frozen=True)
class RecurrentWindowColumns:
    layout: SequenceLayout
    mass: torch.Tensor
    content: torch.Tensor
    write_layout: SequenceLayout
    bank: dict
    times: torch.Tensor
    cache_layout: SequenceLayout | None
    cache_values: torch.Tensor | None
    terminal: RecurrentStateColumns


def scan(layout, a, b, initial, *, matrix=False):
    ids, reverse = layout.padded(b.device)
    pa, pb = gather_column(a, ids), gather_column(b, ids)
    mask = (ids < len(layout.times)).reshape(*ids.shape, *((1,)*(a.ndim-1)))
    identity = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device) if matrix else 1
    pa = torch.where(mask, pa, identity)
    result = (matrix_scan if matrix else affine_scan)(pa.transpose(0, 1), pb.transpose(0, 1), initial)
    return result.transpose(0, 1).flatten(0, 1).index_select(0, reverse)


def prepare_recurrent_columns(memory, layout, mass, content, left):
    from .recurrent_column_math import coefficients, cache_bank
    c = memory.config
    left.validate(c)
    if (layout.samples != left.samples or content.shape != (len(layout.times), c.width) or mass.shape != content.shape[:1]
            or (content.dtype, content.device) != (left.H.dtype, left.H.device)
            or (mass.dtype, mass.device) != (content.dtype, content.device)
            or (mass < 0).any() or not torch.isfinite(mass).all() or not torch.isfinite(content).all()):
        raise ValueError("invalid recurrent window columns")
    for b, (a, z) in enumerate(layout.spans):
        if z>a and layout.times[a] <= left.last_times[b]:
            raise ValueError("recurrent column window must advance logical time")
    rows = (mass.detach() > 0).nonzero().flatten().cpu()
    writes = layout.select(rows)
    bank = {n: getattr(left, n) for n in ("H", "Z", "W")}
    time = left.last_times.clamp_min(0)
    result = RecurrentWindowColumns(layout, mass, content, writes, bank, time, left.cache_layout, left.cache_values, left)
    if not len(rows):
        return result
    ids = rows.to(content.device)
    u, weight = content.index_select(0, ids), mass.index_select(0, ids).clamp_max(1)
    x = norm(u, c.epsilon)
    gaps, terminal_ids = [], []
    lasts, counts = left.last_times.clone(), left.observations.clone()
    for owner, (a, z) in enumerate(writes.spans):
        old_time = left.last_times[owner:owner+1].clamp_min(0)
        gaps.append(writes.times[a:z]-torch.cat((old_time, writes.times[a:z-1])) if z>a else writes.times[:0])
        terminal_ids.append(left.samples+z-1 if z>a else owner)
        if z>a:
            lasts[owner], counts[owner] = writes.times[z-1], increment(int(counts[owner]), z-a)
    delta = torch.cat(gaps).to(content.device)
    cache_layout, cache_values = cache_bank(memory, writes, x, weight, left)
    a, b, matrix = coefficients(memory, writes, u, x, weight, delta, cache_layout, cache_values)
    decay = passive_decay(delta[:, None], (c.lambda_short, 0.), x)
    z = weight[:, None]*(norm(u, c.epsilon)@memory.summary.t())
    fresh = dict(H=scan(writes, a, b, left.H, matrix=matrix),
                 Z=scan(writes, decay[..., None], z[:, None].expand(-1, 2, -1), left.Z),
                 W=scan(writes, decay, weight[:, None].expand(-1, 2), left.W))
    bank = {n: torch.cat((getattr(left, n), t)) for n, t in fresh.items()}
    time = torch.cat((time, writes.times))
    end = torch.tensor(terminal_ids, dtype=torch.int64, device=content.device)
    terminal = replace(left, **{n: t.index_select(0, end) for n, t in bank.items()}, last_times=lasts, observations=counts)
    if cache_layout is not None:
        keep, offsets = [], [0]
        for owner, (a, z) in enumerate(cache_layout.spans):
            rows = torch.arange(a, z)
            if counts[owner] != left.observations[owner]:
                rows = rows[(lasts[owner]-cache_layout.times[a:z]) <= memory.conv.horizon]
            keep.append(rows); offsets.append(offsets[-1]+len(rows))
        keep = torch.cat(keep)
        terminal = replace(terminal, cache_layout=SequenceLayout(torch.tensor(offsets), cache_layout.times[keep]),
                           cache_values=cache_values.index_select(0, keep.to(cache_values.device)))
    return replace(result, bank=bank, times=time, cache_layout=cache_layout, cache_values=cache_values, terminal=terminal)
