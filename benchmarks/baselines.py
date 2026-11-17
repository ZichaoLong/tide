"""Batched, mathematically different baselines for Tide measurements.

Transformer: pre-RMS residual attention (RoPE, 32-wide heads, causal window
including the current token), followed by pre-RMS residual 4d SwiGLU. SSD:
pre-RMS residual Mamba2Simple-style mixer, with expanded inner width, grouped
B/C, scalar-per-head decay, causal depthwise Conv1d and gate-before-RMS output.
Its default expansion is 2, head/state widths are 32, and it has no FFN. The
explicit ffn=True option adds a pre-RMS residual 4d SwiGLU to every SSD block.
Both stacks finish with learned RMSNorm unless final_norm=False is requested.

These are complete [B,L,d] blocks, without token embeddings, task heads, Tide
summaries, allocation or routing. Parameter counts come from actual tensors.
SSD uses the new Torch SSD block primitive, not a fused Mamba package kernel.
CPU/CUDA use native SDPA by default; supported NPU input uses the repository's
native fusion adapter. Model placement/runtime setup belongs to the harness.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from tide.attention import weighted_attention
from tide.ssd import ssd_scan


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


class _Baseline(nn.Module):
    @property
    def parameter_count(self):
        return parameter_count(self)

    def configuration(self):
        return {**self.spec, "parameter_count": self.parameter_count}

    def _check_input(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d:
            raise ValueError(f"expected [B,L,{self.d}] input")


def _positive(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class _SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_up = nn.Linear(d, 8*d, bias=False)
        self.down = nn.Linear(4*d, d, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(gate) * up)


def _rope(x, cosine, sine):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*cosine - odd*sine, even*sine + odd*cosine), -1).flatten(-2)


class _TransformerBlock(nn.Module):
    def __init__(self, d, backend):
        super().__init__()
        self.heads, self.backend = d // 32, backend
        self.attention_norm = nn.RMSNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
        self.ffn_norm = nn.RMSNorm(d, eps=1e-5)
        self.ffn = _SwiGLU(d)

    def forward(self, x, layout):
        batch, length, d = x.shape
        cosine, sine, tiles = layout
        q, k, v = (part.reshape(batch, length, self.heads, 32).transpose(1, 2)
                   for part in self.qkv(self.attention_norm(x)).chunk(3, -1))
        q, k = _rope(q, cosine, sine), _rope(k, cosine, sine)
        reads = []
        for start, stop, lo, mask in tiles:
            # Omitting weights selects the native unit-prior path directly.
            reads.append(weighted_attention(q[:, :, start:stop], k[:, :, lo:stop], v[:, :, lo:stop],
                                            mask=mask, backend=self.backend))
        attention = torch.cat(reads, -2).transpose(1, 2).reshape(batch, length, d)
        x = x + self.output(attention)
        return x + self.ffn(self.ffn_norm(x))


class TransformerBaseline(_Baseline):
    def __init__(self, d, layers, window=128, chunk=128, backend="auto", final_norm=True):
        super().__init__()
        for name, value in (("d", d), ("layers", layers), ("window", window), ("chunk", chunk)):
            _positive(name, value)
        if d % 32:
            raise ValueError("Transformer width must be a multiple of 32")
        if backend not in {"auto", "sdpa", "npu_fusion", "math", "tiled"}:
            raise ValueError(f"unknown attention backend: {backend}")
        self.d, self.window, self.chunk = d, window, chunk
        self.blocks = nn.ModuleList(_TransformerBlock(d, backend) for _ in range(layers))
        self.final_norm = nn.RMSNorm(d, eps=1e-5) if final_norm else nn.Identity()
        self._layout_cache = None
        self.spec = dict(family="transformer", d=d, layers=layers, heads=d//32, head_dim=32,
                         window=window, chunk=chunk, backend=backend, ffn_width=4*d,
                         final_norm=bool(final_norm))

    def _layout(self, x):
        key = (x.shape[1], x.device, x.dtype)
        if self._layout_cache is None or self._layout_cache[0] != key:
            length = x.shape[1]
            dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            positions = torch.arange(length, device=x.device, dtype=torch.int64)
            frequency = torch.exp(torch.arange(0, 32, 2, device=x.device, dtype=dtype)
                                  * (-math.log(10000)/32))
            phase = positions.to(dtype)[:, None] * frequency
            cosine, sine = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
            tiles = []
            for start in range(0, length, self.chunk):
                stop, lo = min(length, start+self.chunk), max(0, start-self.window+1)
                delta = positions[start:stop, None] - positions[None, lo:stop]
                tiles.append((start, stop, lo, (delta >= 0) & (delta < self.window)))
            self._layout_cache = key, (cosine, sine, tuple(tiles))
        return self._layout_cache[1]

    def forward(self, x):
        self._check_input(x)
        if x.shape[1] == 0:
            return self.final_norm(x)
        layout = self._layout(x)
        for block in self.blocks:
            x = block(x, layout)
        return self.final_norm(x)


class _SSDMixer(nn.Module):
    def __init__(self, d, expand, head_dim, state_dim, groups, conv_size, chunk):
        super().__init__()
        self.inner, self.head_dim, self.state_dim = d*expand, head_dim, state_dim
        self.heads, self.groups, self.chunk = self.inner//head_dim, groups, chunk
        conv_width = self.inner + 2*groups*state_dim
        self.input = nn.Linear(d, self.inner+conv_width+self.heads, bias=False)
        self.conv = nn.Conv1d(conv_width, conv_width, conv_size, groups=conv_width,
                              padding=conv_size-1, bias=True)
        self.a_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        dt = torch.exp(torch.empty(self.heads).uniform_(math.log(.001), math.log(.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.skip = nn.Parameter(torch.ones(self.heads))
        self.output_norm = nn.RMSNorm(self.inner, eps=1e-5)
        self.output = nn.Linear(self.inner, d, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        conv_width = self.inner + 2*self.groups*self.state_dim
        gate, projected, dt_raw = self.input(x).split((self.inner, conv_width, self.heads), -1)
        convolved = F.silu(self.conv(projected.transpose(1, 2))[:, :, :length].transpose(1, 2))
        values, B, C = convolved.split((self.inner, self.groups*self.state_dim,
                                        self.groups*self.state_dim), -1)
        values = values.reshape(batch, length, self.heads, self.head_dim)

        def heads(value):
            grouped = value.reshape(batch, length, self.groups, 1, self.state_dim)
            return grouped.expand(batch, length, self.groups, self.heads//self.groups,
                                  self.state_dim).reshape(batch, length, self.heads, self.state_dim)

        dt = F.softplus(dt_raw + self.dt_bias)
        decay = torch.exp(-self.a_log.exp() * dt)
        initial = x.new_zeros(batch, self.heads, self.head_dim, self.state_dim)
        read, _ = ssd_scan(decay, dt[..., None]*values, heads(B), heads(C), initial, chunk_size=self.chunk)
        y = (read + self.skip[None, None, :, None]*values).reshape(batch, length, self.inner)
        return self.output(self.output_norm(y * F.silu(gate)))


class _SSDBlock(nn.Module):
    def __init__(self, d, expand, head_dim, state_dim, groups, conv_size, chunk, ffn):
        super().__init__()
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.mixer = _SSDMixer(d, expand, head_dim, state_dim, groups, conv_size, chunk)
        self.ffn_norm = nn.RMSNorm(d, eps=1e-5) if ffn else None
        self.ffn = _SwiGLU(d) if ffn else None

    def forward(self, x):
        x = x + self.mixer(self.norm(x))
        return x + self.ffn(self.ffn_norm(x)) if self.ffn is not None else x


class SSDBaseline(_Baseline):
    def __init__(self, d, layers, expand=2, head_dim=32, state_dim=32, groups=1,
                 conv_size=4, chunk=32, ffn=False, final_norm=True):
        super().__init__()
        for name, value in (("d", d), ("layers", layers), ("expand", expand), ("head_dim", head_dim),
                            ("state_dim", state_dim), ("groups", groups), ("conv_size", conv_size), ("chunk", chunk)):
            _positive(name, value)
        if d*expand % head_dim or (d*expand//head_dim) % groups:
            raise ValueError("SSD inner width must divide into complete heads and B/C groups")
        self.d = d
        resolved_chunk = min(chunk, 32)
        self.blocks = nn.ModuleList(_SSDBlock(d, expand, head_dim, state_dim, groups, conv_size,
                                             resolved_chunk, ffn) for _ in range(layers))
        self.final_norm = nn.RMSNorm(d, eps=1e-5) if final_norm else nn.Identity()
        self.spec = dict(family="ssd", d=d, layers=layers, expand=expand, inner=d*expand,
                         heads=d*expand//head_dim, head_dim=head_dim, state_dim=state_dim,
                         groups=groups, conv_size=conv_size, requested_chunk=chunk, chunk=resolved_chunk,
                         implementation="torch_ssd_blocks", ffn_width=4*d if ffn else 0,
                         final_norm=bool(final_norm))

    def forward(self, x):
        self._check_input(x)
        if x.shape[1] == 0:
            return self.final_norm(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
