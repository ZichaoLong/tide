"""Windowed KV: one history bank, batched writes and selected Full queries."""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .attention import weighted_attention
from .layout import select_queries
from .parameters import FeedForward, HeadQuery, NodeLinear, norm, select_owners


@dataclass
class KVState:
    times: torch.Tensor
    weight: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    features: torch.Tensor


def rope(x, times, period):
    """x [N,B,H,Q,32]; pair coordinates as the textbook's 2x2 blocks."""
    dtype = torch.float64 if x.dtype==torch.float64 else torch.float32
    freq = torch.exp(torch.arange(0, 32, 2, device=x.device, dtype=dtype)
                     * (-math.log(10000.0) / 32))
    phase = times.to(dtype)[..., None] * (freq / period)
    cs, sn = phase.cos().to(x.dtype)[:, :, None], phase.sin().to(x.dtype)[:, :, None]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cs - b * sn, a * sn + b * cs), -1).flatten(-2)


def _take(x, order):
    extra = x.ndim - order.ndim
    index = order[(...,) + (None,) * extra].expand(*order.shape, *x.shape[order.ndim:])
    return x.gather(2, index)


def chronological_compaction(keep):
    """Stable partition of a chronological bank, without sorting timestamps.

    Valid entries in incoming state and new writes already follow int64 time.
    Keep invalid entries first in their original order, exactly like the old
    stable timestamp sort; callers retain the final finite-window slots.
    No timestamp is converted to floating point, including times above 2**53.
    """
    count = keep.to(torch.int64).cumsum(2)
    position = torch.arange(keep.shape[2], device=keep.device).expand_as(count)
    invalid = keep.shape[2] - count[:, :, -1:]
    destination = torch.where(keep, invalid + count - 1, position - count)
    return torch.empty_like(position).scatter(2, destination, position)


def cache_order(keep, times):
    # Avoid the device AiCPU timestamp sort on NPU. CPU end-to-end witnesses
    # favored the existing stable sort, so its implementation stays separate.
    if keep.device.type == "npu":
        return chronological_compaction(keep)
    return torch.where(keep, times, -torch.ones_like(times)).argsort(dim=2, stable=True)


class KVMemory(nn.Module):
    def __init__(self, nodes, d, groups, rank, period, settle=True, backend="auto", chunk=128, kv_groups=None):
        super().__init__()
        self.d, self.period, self.backend, self.chunk = d, period, backend, chunk
        self.unit_weights = False
        self.window = 128 * period
        self.slots = 128 if settle else self.window
        self.kv = NodeLinear(nodes, d, 2*d, groups if kv_groups is None else kv_groups, rank)
        self.summary = NodeLinear(nodes, d, d//4, groups, rank)
        self.query = HeadQuery(nodes, d, groups, rank)
        self.output = NodeLinear(nodes, d, d, groups, rank)
        self.ffn = FeedForward(nodes, d, groups, rank)

    def prepare(self, u, mass, times, owners, state=None):
        n, b, length, d = u.shape
        accumulation_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        # No reception means no write, including at the first-order boundary.
        # A padding slot must not become a differentiable key prior in a whole
        # window while disappearing from the saved continuation state.
        mass = torch.where(mass>0,mass,torch.zeros_like(mass)).to(accumulation_dtype)
        normal = norm(u)
        keys, values = self.kv(normal, owners).split(d, -1)
        # BF16 projection is genuine; recurrent weighted sums retain FP32.
        features = self.summary(normal, owners).to(accumulation_dtype)
        if state is None:
            state = KVState(times[:, :, :0], mass[:, :, :0], keys[:, :, :0], values[:, :, :0], features[:, :, :0])
        old = state.times.shape[2]
        bank = KVState(torch.cat((state.times, times), 2), torch.cat((state.weight, mass), 2),
                       torch.cat((state.keys, keys), 2), torch.cat((state.values, values), 2),
                       torch.cat((state.features, features), 2))
        short, long, ws, wl = [], [], [], []
        for start in range(0, length, self.chunk):
            end = min(length, start+self.chunk)
            lo, hi = max(0, old+start-self.slots), old+end
            delta = times[:, :, start:end, None] - bank.times[:, :, None, lo:hi]
            visible = (delta >= 0) & (delta < self.window)
            weight = bank.weight[:, :, None, lo:hi] * visible
            decay = torch.exp(delta.clamp_min(0).to(accumulation_dtype) * (-math.log(2)/(8*self.period)))
            for weights, outputs, amounts in ((weight, long, wl), (weight*decay, short, ws)):
                amount = weights.sum(-1)
                denominator = torch.where(amount <= 1, torch.ones_like(amount), amount)
                with torch.autocast(u.device.type, enabled=False):
                    outputs.append(torch.matmul(weights, bank.features[:, :, lo:hi]) / denominator[..., None])
                amounts.append(amount)
        last = torch.where(bank.weight > 0, bank.times, torch.zeros_like(bank.times)).amax(2, keepdim=True)
        keep = (bank.weight > 0) & ((last-bank.times) < self.window)
        order = cache_order(keep, bank.times)
        order = order[:, :, -min(self.slots, order.shape[2]):]
        newstate = KVState(_take(bank.times, order), _take(bank.weight*keep, order),
                           _take(bank.keys, order), _take(bank.values, order), _take(bank.features, order))
        description = (features, torch.cat(short, 2), torch.cat(long, 2), torch.cat(ws, 2), torch.cat(wl, 2))
        context = (u, times, owners, bank, old)
        return description, context, newstate

    def finish(self, context, control):
        u, times, owners, bank, old = context
        # A tile is an interval in the original input, not in the packed query
        # list. Its earliest query can see at most slots-1 preceding entries.
        # Integer masks still decide visibility when receptions have gaps.
        outputs = []
        for start in range(0, u.shape[2], self.chunk):
            end = min(u.shape[2], start+self.chunk)
            lo, hi = max(0, old+start-self.slots+1), old+end
            outputs.append(self._finish_tile(u[:, :, start:end], times[:, :, start:end], owners,
                                             bank, lo, hi, control[:, :, start:end]))
        if not outputs:
            return torch.zeros_like(u)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, 2)

    def _finish_tile(self, u, times, owners, bank, lo, hi, control):
        packed = select_queries(u, control, times)
        if packed is None:
            return torch.zeros_like(u)
        ids = select_owners(owners, packed.nodes)
        n, b, qlen, d = packed.data.shape
        queries = self.query(norm(packed.data), ids).reshape(n,b,qlen,d//32,32).transpose(2,3)
        anchor = packed.positions[:,:,:1]
        queries = rope(queries, packed.positions-anchor, self.period)
        keys = bank.keys[packed.nodes, :, lo:hi].reshape(n,b,-1,d//32,32).transpose(2,3)
        values = bank.values[packed.nodes, :, lo:hi].reshape(n,b,-1,d//32,32).transpose(2,3)
        key_times = bank.times[packed.nodes, :, lo:hi]
        keys = rope(keys, key_times-anchor, self.period)
        delta = packed.positions[..., None] - key_times[:, :, None, :]
        visible = (delta >= 0) & (delta < self.window) & packed.valid[..., None]
        w = bank.weight[packed.nodes, :, lo:hi]
        read = weighted_attention(queries, keys, values,
                                  weights=None if self.unit_weights else w,
                                  mask=visible[:, :, None], backend=self.backend)
        amount = (visible*w[:, :, None]).sum(-1).clamp(max=1)
        read = read.transpose(2,3).reshape(n,b,qlen,d) * amount[...,None]
        x = packed.data + self.output(read, ids)
        return packed.publish(self.ffn(x, ids))
