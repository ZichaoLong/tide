"""Windowed KV: one history bank, batched writes and selected Full queries."""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .attention import weighted_attention
from .ffn_packing import selected_ffn, validate_ffn_packing
from .layout import select_queries
from .parameters import FeedForward, HeadQuery, NodeLinear, _INHERIT_GROUPS, norm, select_owners


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
    def __init__(self, nodes, d, groups, rank, period, settle=True, backend="auto", chunk=128, kv_groups=None,
                 *, position_period=None, window=None, short_decay=None, kv_width=None, ffn_width=None,
                 ffn_groups=_INHERIT_GROUPS, ffn_packing="rectangular", ffn_weight_access="gather"):
        super().__init__()
        self.d, self.period, self.backend, self.chunk = d, period, backend, chunk
        self.ffn_packing = validate_ffn_packing(ffn_packing)
        self.kv_width = d if kv_width is None else kv_width
        if type(self.kv_width) is not int or not 0 < self.kv_width <= d or self.kv_width % 32:
            raise ValueError("kv_width must be a positive multiple of 32 <= d")
        self.heads = self.kv_width // 32
        self.unit_weights = False
        self.weak_log = False
        self.position_period = period if position_period is None else position_period
        self.window = 128 * period if window is None else window
        self.short_decay = math.log(2)/(8*period) if short_decay is None else short_decay
        # Settle writes are period-spaced; Timed can write at every logical tick.
        self.slots = (self.window + period - 1)//period if settle else self.window
        self.kv = NodeLinear(nodes, d, 2*self.kv_width, groups if kv_groups is None else kv_groups, rank)
        self.summary = NodeLinear(nodes, d, d//4, groups, rank)
        self.query = HeadQuery(nodes, d, groups, rank, out_features=self.kv_width)
        self.output = NodeLinear(nodes, self.kv_width, d, groups, rank)
        self.ffn = FeedForward(nodes, d, groups if ffn_groups is _INHERIT_GROUPS else ffn_groups,
                               rank, hidden_features=ffn_width, weight_access=ffn_weight_access)

    def prepare(self, u, mass, times, owners, state=None):
        n, b, length, d = u.shape
        accumulation_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        # No reception means no write, including at the first-order boundary.
        # A padding slot must not become a differentiable key prior in a whole
        # window while disappearing from the saved continuation state.
        present = mass != 0 if self.weak_log else mass > 0
        mass = torch.where(present,mass,torch.zeros_like(mass)).to(accumulation_dtype)
        normal = norm(u)
        keys, values = self.kv(normal, owners).split(self.kv_width, -1)
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
            codes = bank.weight[:, :, None, lo:hi]
            extended = self.weak_log and bool((codes < 0).any())
            features_tile = bank.features[:, :, lo:hi]
            if extended:
                from .weak_mass import above_one, decode, wide
                weight = decode(wide(codes)) * visible.cpu()
                features_tile = wide(features_tile)
            else:
                weight = codes * visible
            decay = torch.exp(delta.clamp_min(0).to(accumulation_dtype) * -self.short_decay)
            if extended:
                decay = decay.to(weight)
            for weights, outputs, amounts in ((weight, long, wl), (weight*decay, short, ws)):
                amount = weights.sum(-1)
                normalized = above_one(weights) if extended else amount > 1
                denominator = torch.where(normalized, amount, torch.ones_like(amount))
                with torch.autocast(u.device.type, enabled=False):
                    outputs.append((torch.matmul(weights, features_tile) / denominator[..., None]).to(features))
                amounts.append(amount.to(features))
        written = bank.weight != 0 if self.weak_log else bank.weight > 0
        last = torch.where(written, bank.times, torch.zeros_like(bank.times)).amax(2, keepdim=True)
        keep = written & ((last-bank.times) < self.window)
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
        queries = self.query(norm(packed.data), ids).reshape(n,b,qlen,self.heads,32).transpose(2,3)
        anchor = packed.positions[:,:,:1]
        queries = rope(queries, packed.positions-anchor, self.position_period)
        keys = bank.keys[packed.nodes, :, lo:hi].reshape(n,b,-1,self.heads,32).transpose(2,3)
        values = bank.values[packed.nodes, :, lo:hi].reshape(n,b,-1,self.heads,32).transpose(2,3)
        key_times = bank.times[packed.nodes, :, lo:hi]
        keys = rope(keys, key_times-anchor, self.position_period)
        delta = packed.positions[..., None] - key_times[:, :, None, :]
        visible = (delta >= 0) & (delta < self.window) & packed.valid[..., None]
        w = bank.weight[packed.nodes, :, lo:hi]
        if not self.unit_weights:
            # An unwritten slot has no key, including in the prior VJP.
            visible = visible & ((w != 0 if self.weak_log else w > 0)[:, :, None, :])
        read = weighted_attention(queries, keys, values,
                                  weights=None if self.unit_weights else w,
                                  mask=visible[:, :, None], backend=self.backend,
                                  mass_scaled=not self.unit_weights, _weak_log=self.weak_log)
        read = read.transpose(2,3).reshape(n,b,qlen,self.kv_width)
        if self.unit_weights:
            amount = (visible*w[:, :, None]).sum(-1).clamp(max=1)
            read = read * amount[...,None]
        x = packed.data + self.output(read, ids)
        return packed.publish(selected_ffn(self.ffn, x, packed, ids, packing=self.ffn_packing))
