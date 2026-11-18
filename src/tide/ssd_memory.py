"""Main SSD/state binding: batched preparation and selected Full content.

Equations (4)--(11) of the fixed SSD learning note, with the main profile's
32-wide heads/state, one B/C group, four convolution taps and zero passive
matrix decay. Summary scans use local time differences, never growing exponentials.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .layout import select_queries
from .parameters import FeedForward, NodeLinear, NodeVector, norm, owner_ids, select_owners
from .ssd import ConvState, logical_conv, ssd_scan


@dataclass
class SSDState:
    """Raw node-private state saved at each sample's last positive reception."""

    times: torch.Tensor
    matrix: torch.Tensor
    short: torch.Tensor
    long: torch.Tensor
    short_weight: torch.Tensor
    long_weight: torch.Tensor
    conv: ConvState


class SSDMemory(nn.Module):
    def __init__(self, nodes, d, groups, rank, period, settle=True, backend="auto", chunk=128,
                 *, short_decay=None, conv_stride=None):
        super().__init__()
        if d < 32 or d % 32:
            raise ValueError("SSD width must be a positive multiple of 32")
        if not isinstance(period, int) or period < 1:
            raise ValueError("SSD period must be a positive integer")
        if not isinstance(chunk, int) or chunk < 1:
            raise ValueError("SSD chunk must be a positive integer")
        self.d, self.heads, self.period = d, d // 32, period
        self.stride, self.chunk = period if settle else 1, min(chunk, 32)
        if conv_stride is not None:
            self.stride = conv_stride
        self.short_decay = math.log(2)/(8*period) if short_decay is None else short_decay
        self.backend = backend
        self.input = NodeLinear(nodes, d, 2*d + 64 + self.heads, groups, rank)
        self.summary = NodeLinear(nodes, d, d//4, groups, rank)
        self.read = NodeLinear(nodes, d, d//4, groups, rank)
        self.output = NodeLinear(nodes, d, d, groups, rank)
        self.ffn = FeedForward(nodes, d, groups, rank)
        self.a_log = NodeVector(nodes, self.heads, groups)
        self.dt_bias = NodeVector(nodes, self.heads, groups, value=-2)
        self.skip = NodeVector(nodes, self.heads, groups, value=1)
        self.rho = NodeVector(nodes, d, groups, value=1)
        self.conv_kernel = NodeVector(nodes, 4*(d+64), groups)
        self.conv_bias = NodeVector(nodes, d+64, groups)
        nn.init.normal_(self.conv_kernel.value, std=0.5)

    def _initial(self, u, times):
        n, b = u.shape[:2]
        base = u.new_empty((), dtype=torch.float64 if u.dtype == torch.float64 else torch.float32)
        return SSDState(times.new_zeros((n, b)), base.new_zeros((n, b, self.heads, 32, 32)),
                        base.new_zeros((n, b, self.d//4)), base.new_zeros((n, b, self.d//4)),
                        base.new_zeros((n, b)), base.new_zeros((n, b)),
                        ConvState(times[..., :0], base.new_zeros((n, b, 0, self.d+64))))

    def _summaries(self, features, weight, times, state):
        previous = torch.cat((state.times[..., None], times[..., :-1]), dim=-1)
        elapsed = (times - previous).clamp_min(0).to(features.dtype)
        short_decay = torch.exp(elapsed * -self.short_decay)
        decay = torch.stack((short_decay, torch.ones_like(short_decay)), dim=-1)
        writes = torch.cat((features * weight[..., None], weight[..., None]), dim=-1)
        writes = writes.unsqueeze(-2).expand(*writes.shape[:-1], 2, writes.shape[-1])
        initial = torch.stack((torch.cat((state.short, state.short_weight[..., None]), -1),
                               torch.cat((state.long, state.long_weight[..., None]), -1)), -2)
        ones = torch.ones_like(decay[..., None])
        effective, _ = ssd_scan(decay, writes, ones, ones, initial[..., None], chunk_size=self.chunk)
        return effective

    def prepare(self, u, mass, times, owners, state=None):
        """Prepare all positive receptions, returning description, context and raw state.

        Input axes are node, sample, logical position, channel. Times are int64,
        strictly increasing per node/sample and follow the incoming state. Owners
        map current node rows to parameter owners. Reception is clipped to [0,1];
        at nonpositive mass the update follows the explicit no-write branch.
        Low-precision projections keep matrix, summary and convolution state in
        FP32; only these state operations disable autocast. FP64 stays FP64.
        """
        if u.ndim != 4 or u.shape[-1] != self.d or mass.shape != u.shape[:-1]:
            raise ValueError("SSD expects u [N,B,L,d] and mass [N,B,L]")
        if times.shape != mass.shape or times.dtype != torch.int64 or times.device != u.device:
            raise ValueError("SSD times must be int64 [N,B,L] on the input device")
        mixed_mass = u.dtype in (torch.float16, torch.bfloat16) and mass.dtype == torch.float32
        if mass.device != u.device or mass.dtype != u.dtype and not mixed_mass:
            raise ValueError("SSD mass must share input dtype/device, or be FP32 with low-precision input")
        ids = owner_ids(owners)
        if ids.shape != (u.shape[0],) or ids.dtype != torch.int64 or ids.device != u.device:
            raise ValueError("SSD owners must be int64 [N] on the input device")
        n, b, length, d = u.shape
        state = self._initial(u, times) if state is None else state
        if length == 0:
            empty = u[..., :d//4]
            return (empty, empty, empty, mass, mass), (u, times, owners, u, u, u), state
        positive = mass > 0
        dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        weight = mass.to(dtype).clamp(max=1).masked_fill(~positive, 0)
        normalized = norm(u)
        gate, projected, dt_raw = self.input(normalized, owners).split((d, d+64, self.heads), -1)
        features = self.summary(normalized, owners)
        with torch.autocast(device_type=u.device.type, enabled=False):
            kernel = self.conv_kernel(owners).to(dtype).reshape(n, 1, 4, d+64)
            bias = self.conv_bias(owners).to(dtype)[:, None, :]
            convolved, cache = logical_conv(projected.to(dtype), weight, times, kernel, bias,
                                            state.conv, stride=self.stride)
            values, B, C = F.silu(convolved).split((d, 32, 32), -1)
            dt = weight[..., None] * F.softplus(dt_raw.to(dtype) + self.dt_bias(owners).to(dtype)[:, None, None, :])
            decay = torch.exp(-self.a_log(owners).to(dtype)[:, None, None, :].exp() * dt)
            write = dt[..., None] * values.reshape(n, b, length, self.heads, 32)
            B = B[..., None, :].expand(n, b, length, self.heads, 32)
            C = C[..., None, :].expand(n, b, length, self.heads, 32)
            reads, matrix = ssd_scan(decay, write, B, C, state.matrix, chunk_size=self.chunk)
            reads = reads.reshape(n, b, length, d)
            summaries = self._summaries(features.to(dtype), weight, times, state)
        short, long = summaries[..., 0, :-1], summaries[..., 1, :-1]
        ws, wl = summaries[..., 0, -1], summaries[..., 1, -1]
        denominator = torch.where(ws <= 1, torch.ones_like(ws), ws)
        description = (features, short / denominator[..., None], self.read(reads, owners), ws, wl)

        # Effective summaries at absent query positions are not raw saved state.
        order = torch.arange(length, device=u.device).expand(n, b, length)
        last = order.masked_fill(~positive, -1).amax(-1)
        index = last.clamp_min(0)[..., None]

        def raw(current, previous):
            if current.ndim == 4:
                picked = current.gather(2, index[..., None].expand(n, b, 1, current.shape[-1])).squeeze(2)
                return torch.where((last >= 0)[..., None], picked, previous)
            picked = current.gather(2, index).squeeze(2)
            return torch.where(last >= 0, picked, previous)

        newstate = SSDState(raw(times, state.times), matrix, raw(short, state.short), raw(long, state.long),
                            raw(ws, state.short_weight), raw(wl, state.long_weight), cache)
        context = (u, times, owners, reads, values, gate)
        return description, context, newstate

    def finish(self, context, control):
        """Run output projection and FFN only on the selected query pack."""
        u, times, owners, reads, values, gate = context
        packed = select_queries(u, control, times)
        if packed is None:
            return torch.zeros_like(u)
        ids = select_owners(owners, packed.nodes)
        index = (packed.nodes[packed.node], packed.batch, packed.time)
        rows = torch.cat((reads[index], values[index], gate[index]), dim=-1)
        selected = rows.new_zeros((*packed.data.shape[:-1], 3*self.d))
        selected = selected.index_put((packed.node, packed.batch, packed.slot), rows)
        read, value, output_gate = selected.split(self.d, -1)
        skip = self.skip(ids).repeat_interleave(32, dim=-1)[:, None, None, :]
        gated = (read + skip * value) * F.silu(output_gate)
        mix = self.output(norm(gated) * self.rho(ids)[:, None, None, :], ids)
        return packed.publish(self.ffn(packed.data + mix, ids))
