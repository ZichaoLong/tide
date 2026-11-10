"""KV §2–4: shared numeric storage with independent snapshot adjoints.

Sharing is established by actual storage intervals, never value equality.
The bank is call-local; it neither caches a parameter version nor changes State
decoding. Saved original rows retain version guards. Independent row gradients
are returned even when their forward storage overlaps.
"""
from collections import defaultdict
import math
import torch
from . import packing_observation as observation


class Bank:
    def __init__(self, rows):
        self.lengths = [len(t) for t in rows]
        self.tail = rows[0].shape[1:]
        unit = math.prod(self.tail)
        self.spans, offsets, groups = [], [0]*len(rows), defaultdict(list)
        self.size = 0
        for i, t in enumerate(rows):
            start = t.storage_offset()//unit
            groups[t.untyped_storage()._cdata].append((start, start+len(t), i))
        for intervals in groups.values():
            merged = []
            for start, end, i in sorted(intervals):
                if not merged or start > merged[-1][1]:
                    merged.append([start, end, i, []])
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][3].append((start, i))
            for start, end, source, members in merged:
                self.spans.append((source, start*unit, end-start))
                for begin, i in members:
                    offsets[i] = self.size+begin-start
                self.size += end-start
        self.shared = self.size < sum(self.lengths)
        if self.shared:
            length = max(1, max(self.lengths))
            # Metadata only. No tensor values or device readiness decisions.
            self.index = torch.tensor([[offset+j if j < n else self.size for j in range(length)]
                for offset, n in zip(offsets, self.lengths)], device=rows[0].device, dtype=torch.int64)

    def values(self, rows):
        unit = math.prod(self.tail)
        pieces = [rows[i].as_strided((n, unit), (unit, 1), start)
                  for i, start, n in self.spans]
        values = torch.cat(pieces+[rows[0].new_zeros((1, unit))])
        return values.reshape(-1, *self.tail)

    def gather(self, values):
        return values.index_select(0, self.index.flatten()).reshape(*self.index.shape, *values.shape[1:])


def plan(rows):
    if len(rows) < 2 or not rows[0].ndim:
        return None
    ref, unit = rows[0], math.prod(rows[0].shape[1:])
    if not unit or any(t.shape[1:] != ref.shape[1:] or t.dtype != ref.dtype or t.device != ref.device
            or not t.is_contiguous() or t.is_conj() or t.is_neg() or t.storage_offset() % unit for t in rows):
        return None
    bank = Bank(rows)
    return bank if bank.shared else None


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bank, *rows):
        ctx.set_materialize_grads(False)
        ctx.lengths = bank.lengths
        ctx.save_for_backward(*rows)
        return bank.gather(bank.values(rows))

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("KV snapshot bank supports first-order VJP only")
        rows = ctx.saved_tensors
        return (None, *(bar[i, :n] if bar is not None and t.requires_grad else None
                        for i, (t, n) in enumerate(zip(rows, ctx.lengths))))


def gather(rows):
    bank = plan(rows)
    if bank is None:
        return None
    observation.projection("kv_unique_gather", bank.size)
    return _Gather.apply(bank, *rows)


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bank, visible, weight, *rows):
        from .packing import project
        ctx.set_materialize_grads(False)
        ctx.bank = bank
        values = bank.values(rows)
        # An ignored huge/NaN entry must be masked BEFORE projection. Count
        # visibility with int64; this does not atomically sum floating gradients.
        used = torch.zeros(bank.size+1, dtype=torch.int64, device=visible.device)
        used.index_add_(0, bank.index.flatten(), visible.to(torch.int64).flatten())
        safe = torch.where((used > 0)[:, None], values, torch.zeros_like(values))
        output = bank.gather(project(safe, weight))
        ctx.save_for_backward(visible, weight, safe, *rows)
        return torch.where(visible[..., None], output, torch.zeros_like(output))

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("KV snapshot bank supports first-order VJP only")
        visible, weight, safe, *rows = ctx.saved_tensors
        if bar is None:
            return (None, None, None, *([None]*len(rows)))
        grad = torch.where(visible[..., None], bar, torch.zeros_like(bar))
        flat = grad.reshape(-1, weight.shape[0])
        # Preserve independent snapshot roots. There is no gradient through a
        # hidden bank root and no zero-filled connection to future snapshots.
        inputs = flat@weight if any(t.requires_grad for t in rows) else None
        inputs = inputs.reshape(*grad.shape[:2], weight.shape[1]) if inputs is not None else None
        dw = flat.t()@ctx.bank.gather(safe).reshape(-1, weight.shape[1]) if weight.requires_grad else None
        return (None, None, dw, *(inputs[i, :n] if t.requires_grad else None
                for i, (t, n) in enumerate(zip(rows, ctx.bank.lengths))))


def project(rows, weight, visible):
    bank = plan(rows)
    if bank is None:
        return None
    observation.projection("kv_unique_projection", bank.size)
    return _Project.apply(bank, visible, weight, *rows)
