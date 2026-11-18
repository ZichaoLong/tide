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
from dataclasses import dataclass
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
    def __init__(self, d, width=None):
        super().__init__()
        width = 4*d if width is None else width
        self.gate_up = nn.Linear(d, 2*width, bias=False)
        self.down = nn.Linear(width, d, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(gate) * up)


def _rope(x, cosine, sine):
    cosine, sine = cosine.to(x.dtype), sine.to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*cosine - odd*sine, even*sine + odd*cosine), -1).flatten(-2)


@dataclass
class TransformerState:
    """Private, bounded per-layer KV cache; positions count actual input tokens.

    Keys already include RoPE at their original positions. State retains autograd
    until the caller detaches it; passing it never mutates an earlier state.
    """

    position: int
    keys: tuple
    values: tuple


class _TransformerBlock(nn.Module):
    def __init__(self, d, backend, head_dim=32, ffn_width=None):
        super().__init__()
        self.heads, self.head_dim, self.backend = d // head_dim, head_dim, backend
        self.attention_norm = nn.RMSNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
        self.ffn_norm = nn.RMSNorm(d, eps=1e-5)
        self.ffn = _SwiGLU(d, ffn_width)

    def forward(self, x, layout, cached=None, cache_size=None):
        batch, length, d = x.shape
        cosine, sine, tiles = layout
        q, k, v = (part.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for part in self.qkv(self.attention_norm(x)).chunk(3, -1))
        q, k = _rope(q, cosine, sine), _rope(k, cosine, sine)
        if cached is not None:
            old_k, old_v = cached
            for previous, current in ((old_k, k), (old_v, v)):
                if (previous.ndim != 4 or previous.shape[:2] != current.shape[:2]
                        or previous.shape[-1] != current.shape[-1]
                        or previous.dtype != current.dtype or previous.device != current.device):
                    raise ValueError("Transformer cache shape/dtype/device differs from projected input")
            k, v = torch.cat((old_k, k), -2), torch.cat((old_v, v), -2)
        reads = []
        for start, stop, lo, hi, mask in tiles:
            # Only the new query rows are projected/read. Keys are real history.
            reads.append(weighted_attention(q[:, :, start:stop], k[:, :, lo:hi], v[:, :, lo:hi],
                                            mask=mask, backend=self.backend))
        attention = torch.cat(reads, -2).transpose(1, 2).reshape(batch, length, d)
        x = x + self.output(attention)
        x = x + self.ffn(self.ffn_norm(x))
        saved = None
        if cache_size is not None:
            # clone bounds the retained storage as well as the visible shape.
            saved = ((k[..., -cache_size:, :].clone(), v[..., -cache_size:, :].clone())
                     if cache_size else (k[..., :0, :].clone(), v[..., :0, :].clone()))
        return x, saved


class TransformerBaseline(_Baseline):
    def __init__(self, d, layers, window=128, chunk=128, backend="auto", final_norm=True,
                 ffn_width=None, head_dim=32):
        super().__init__()
        ffn_width = 4*d if ffn_width is None else ffn_width
        for name, value in (("d", d), ("layers", layers), ("window", window), ("chunk", chunk),
                            ("ffn_width", ffn_width), ("head_dim", head_dim)):
            _positive(name, value)
        if head_dim % 2 or d % head_dim:
            raise ValueError("Transformer head width must be even and divide model width")
        if backend not in {"auto", "sdpa", "npu_fusion", "math", "tiled"}:
            raise ValueError(f"unknown attention backend: {backend}")
        self.d, self.window, self.chunk, self.head_dim = d, window, chunk, head_dim
        self.blocks = nn.ModuleList(_TransformerBlock(d, backend, head_dim, ffn_width)
                                    for _ in range(layers))
        self.final_norm = nn.RMSNorm(d, eps=1e-5) if final_norm else nn.Identity()
        self._layout_cache = None
        self.spec = dict(family="transformer", d=d, layers=layers, heads=d//head_dim, head_dim=head_dim,
                         window=window, chunk=chunk, backend=backend, ffn_width=ffn_width,
                         final_norm=bool(final_norm))

    def _layout(self, x, position=0, history=0):
        key = (x.shape[1], position, history, x.device, x.dtype)
        if self._layout_cache is None or self._layout_cache[0] != key:
            length = x.shape[1]
            dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            positions = torch.arange(position, position+length, device=x.device, dtype=torch.int64)
            key_positions = torch.arange(position-history, position+length, device=x.device, dtype=torch.int64)
            frequency = torch.exp(torch.arange(0, self.head_dim, 2, device=x.device, dtype=dtype)
                                  * (-math.log(10000)/self.head_dim))
            phase = positions.to(dtype)[:, None] * frequency
            cosine, sine = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
            tiles = []
            for start in range(0, length, self.chunk):
                stop = min(length, start+self.chunk)
                lo, hi = max(0, history+start-self.window+1), history+stop
                delta = positions[start:stop, None] - key_positions[None, lo:hi]
                tiles.append((start, stop, lo, hi, (delta >= 0) & (delta < self.window)))
            self._layout_cache = key, (cosine, sine, tuple(tiles))
        return self._layout_cache[1]

    def forward(self, x, state=None, *, return_state=False):
        """Evaluate new inputs, optionally continuing a real incremental KV cache.

        The default returns the original tensor API. ``return_state=True`` returns
        ``(output, state)``; detach at optimizer boundaries for truncated training.
        Input history is not recomputed. The caller owns the returned cache.
        """
        self._check_input(x)
        position, history = 0, 0
        if state is not None:
            position = state.position
            if type(position) is not int or position < 0:
                raise ValueError("Transformer state.position must be a nonnegative int")
            if len(state.keys) != len(self.blocks) or len(state.values) != len(self.blocks):
                raise ValueError("Transformer cache layer count differs from model")
            history = state.keys[0].shape[-2]
            if history != min(position, self.window-1) or any(
                    tensor.shape[-2] != history for tensor in (*state.keys, *state.values)):
                raise ValueError("Transformer cache length differs from its original position/window")
        if position+x.shape[1] > torch.iinfo(torch.int64).max:
            raise OverflowError("Transformer position exceeds int64")
        if x.shape[1] == 0:
            if state is None:
                dtype = (torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type)
                         and x.dtype == torch.float32 else x.dtype)
                empty = tuple(x.new_empty((x.shape[0], self.d//self.head_dim, 0, self.head_dim), dtype=dtype)
                              for _ in self.blocks)
                state = TransformerState(0, empty, tuple(t.clone() for t in empty))
            result = self.final_norm(x)
            return (result, state) if return_state else result
        layout = self._layout(x, position, history)
        keys, values = [], []
        retained = min(position+x.shape[1], self.window-1) if return_state else None
        for index, block in enumerate(self.blocks):
            cached = None if state is None else (state.keys[index], state.values[index])
            x, saved = block(x, layout, cached, retained)
            if saved is not None:
                keys.append(saved[0]); values.append(saved[1])
        result = self.final_norm(x)
        state = TransformerState(position+x.shape[1], tuple(keys), tuple(values))
        return (result, state) if return_state else result


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

        # BF16 projections/convolution, explicit FP32 recurrence and state.
        # Keep scan arguments in one accumulation dtype; FP64 witnesses remain FP64.
        accumulation = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            values, B, C = values.to(accumulation), B.to(accumulation), C.to(accumulation)
            dt = F.softplus(dt_raw.to(accumulation) + self.dt_bias.to(accumulation))
            decay = torch.exp(-self.a_log.to(accumulation).exp() * dt)
            initial = x.new_zeros(batch, self.heads, self.head_dim, self.state_dim, dtype=accumulation)
            read, _ = ssd_scan(decay, dt[..., None]*values, heads(B), heads(C), initial, chunk_size=self.chunk)
            y = (read + self.skip.to(accumulation)[None, None, :, None]*values).reshape(batch, length, self.inner)
            normalized = self.output_norm(y * F.silu(gate.to(accumulation)))
        return self.output(normalized)


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
