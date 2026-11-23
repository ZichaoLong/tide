"""Language-model adapters for the current, explicitly pinned pilot.

The Tide adapter leaves the installed backbone's initialization and mathematics
unchanged. Timed language logits use the last tick *before* the next external
input. This is a causal observation policy, not an assertion that all work caused
by the current token has settled. Continuous calls never insert flush periods.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import copy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LMOutput:
    logits: torch.Tensor
    state: Any
    work: dict | None = None


@dataclass
class AttentionState:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class TransformerState:
    position: int
    layers: tuple[AttentionState, ...]


def detach_state(value: Any) -> Any:
    """Detach an application/backbone state without changing its ownership."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: detach_state(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {k: detach_state(v) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(detach_state(v) for v in value))
    if isinstance(value, tuple):
        return tuple(detach_state(v) for v in value)
    if isinstance(value, list):
        return [detach_state(v) for v in value]
    return value


def _check_tokens(tokens: torch.Tensor) -> None:
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
        raise ValueError("tokens must be nonempty [batch, positions]")
    if tokens.dtype != torch.int64:
        raise TypeError("tokens must use torch.int64")


class TideLM(nn.Module):
    def __init__(self, spec: dict, vocab_size: int):
        super().__init__()
        from tide import ModelConfig, TideModel

        config = ModelConfig.from_dict(copy.deepcopy(spec["config"]))
        if config.input_dim not in (None, config.d):
            raise ValueError("pilot embedding width must equal the Tide content width")
        self.model_spec = copy.deepcopy(spec)
        self.family = "tide"
        self.vocab_size, self.d = vocab_size, config.d
        self.embedding = nn.Embedding(vocab_size, config.d)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.core = TideModel(config)
        self.output_norm = nn.RMSNorm(config.d, eps=1e-5)
        self.readout_policy = (
            "settle_per_input_position" if self.core.settle
            else "timed_last_tick_before_next_external_input"
        )

    def forward(self, tokens, state=None, record_work=False):
        _check_tokens(tokens)
        result = self.core(self.embedding(tokens), state, record_work=record_work)
        hidden = result.readout()
        if not self.core.settle:
            b, length = tokens.shape
            hidden = hidden.reshape(b, length, self.core.graph.period, self.d)[:, :, -1]
        logits = F.linear(self.output_norm(hidden), self.embedding.weight)
        work = None
        if record_work:
            strengths = result.strengths.sum(0)
            all_empty = (strengths <= 0).sum().item()
            if not self.core.settle:
                strengths = strengths.reshape(*tokens.shape, self.core.graph.period)[:, :, -1]
            work = dict(result.work or {})
            work.update(
                family="tide",
                readout_policy=self.readout_policy,
                input_positions=tokens.numel(),
                empty_output_positions=int((strengths <= 0).sum().item()),
                empty_logical_output_slots=int(all_empty),
                output_positions=tokens.numel(),
                logical_ticks=int(result.times.numel()),
                period=int(self.core.graph.period),
            )
        return LMOutput(logits, result.state, work)


class CausalAttention(nn.Module):
    """RoPE attention with one causal sliding-window function in every mode."""
    def __init__(self, d: int, heads: int, window: int):
        super().__init__()
        if d % heads or (d // heads) % 2:
            raise ValueError("attention requires an even head width and heads dividing d")
        self.d, self.heads, self.head_dim, self.window = d, heads, d // heads, window
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        inverse_frequency = 10000.0 ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def _rotate(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        phase = positions.to(self.inverse_frequency.dtype)[:, None] * self.inverse_frequency[None, :]
        cosine, sine = phase.cos()[None, None], phase.sin()[None, None]
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)

    def forward(self, x: torch.Tensor, state: AttentionState | None, start: int):
        b, length, _ = x.shape
        parts = self.qkv(x).reshape(b, length, 3, self.heads, self.head_dim)
        q, k, v = (part.transpose(1, 2) for part in parts.unbind(2))
        positions = torch.arange(start, start + length, device=x.device, dtype=torch.int64)
        q, k = self._rotate(q, positions), self._rotate(k, positions)
        old_length = 0
        if state is not None:
            if state.key.ndim != 4 or state.value.shape != state.key.shape:
                raise ValueError("invalid attention cache shape")
            if state.key.shape[:2] != (b, self.heads) or state.key.shape[-1] != self.head_dim:
                raise ValueError("attention cache belongs to a different batch/model")
            old_length = state.key.shape[-2]
            if old_length > min(start, self.window - 1):
                raise ValueError("attention cache exceeds the declared history window")
            k, v = torch.cat((state.key, k), -2), torch.cat((state.value, v), -2)
        key_positions = torch.arange(start - old_length, start + length, device=x.device, dtype=torch.int64)
        distance = positions[:, None] - key_positions[None, :]
        visible = (distance >= 0) & (distance < self.window)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=visible, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(b, length, self.d)
        keep = min(self.window - 1, k.shape[-2])
        # clone owns the bounded history storage while retaining autograd edges.
        following = AttentionState(k[:, :, -keep:].clone(), v[:, :, -keep:].clone()) if keep else AttentionState(
            k[:, :, :0].clone(), v[:, :, :0].clone())
        return self.out(attended), following


class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn_width: int, window: int):
        super().__init__()
        self.attention_norm = nn.RMSNorm(d, eps=1e-5)
        self.attention = CausalAttention(d, heads, window)
        self.ffn_norm = nn.RMSNorm(d, eps=1e-5)
        self.gate_up = nn.Linear(d, 2 * ffn_width, bias=False)
        self.down = nn.Linear(ffn_width, d, bias=False)

    def forward(self, x, state, start):
        attention, following = self.attention(self.attention_norm(x), state, start)
        x = x + attention
        gate, up = self.gate_up(self.ffn_norm(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up), following


class TransformerLM(nn.Module):
    def __init__(self, spec: dict, vocab_size: int):
        super().__init__()
        self.model_spec = copy.deepcopy(spec)
        self.family, self.vocab_size = "transformer", vocab_size
        self.d = int(spec.get("d", 128))
        count, heads = int(spec.get("layers", 4)), int(spec.get("heads", 4))
        ffn_width, self.window = int(spec.get("ffn_width", 4 * self.d)), int(spec.get("window", 128))
        if min(self.d, count, heads, ffn_width, self.window) < 1:
            raise ValueError("Transformer dimensions and window must be positive")
        self.embedding = nn.Embedding(vocab_size, self.d)
        self.blocks = nn.ModuleList(TransformerBlock(self.d, heads, ffn_width, self.window) for _ in range(count))
        self.output_norm = nn.RMSNorm(self.d, eps=1e-5)
        self.readout_policy = "causal_per_input_position"
        self.apply(self._initialize)
        with torch.no_grad():
            for block in self.blocks:
                block.attention.out.weight.div_(math.sqrt(2 * count))
                block.down.weight.div_(math.sqrt(2 * count))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, tokens, state=None, record_work=False):
        _check_tokens(tokens)
        if state is not None and not isinstance(state, TransformerState):
            raise TypeError("expected TransformerState")
        if state is not None and len(state.layers) != len(self.blocks):
            raise ValueError("Transformer cache has a different number of layers")
        start = 0 if state is None else state.position
        if type(start) is not int or start < 0:
            raise ValueError("state position must be a nonnegative integer")
        hidden, following = self.embedding(tokens), []
        for index, block in enumerate(self.blocks):
            hidden, cache = block(hidden, None if state is None else state.layers[index], start)
            following.append(cache)
        logits = F.linear(self.output_norm(hidden), self.embedding.weight)
        work = None
        if record_work:
            b, length = tokens.shape
            available = sum(min(self.window, start + t + 1) for t in range(length))
            work = dict(family="transformer", input_positions=tokens.numel(), output_positions=tokens.numel(),
                        empty_output_positions=0, readout_policy=self.readout_policy,
                        attention_visible_pairs=b * available * sum(block.attention.heads for block in self.blocks),
                        cached_positions=following[0].key.shape[-2])
        return LMOutput(logits, TransformerState(start + tokens.shape[1], tuple(following)), work)


def build_model(model_spec: dict, vocab_size: int) -> nn.Module:
    """Build a causal FP32 LM with a common tied embedding/output boundary."""
    if type(vocab_size) is not int or vocab_size < 2:
        raise ValueError("vocab_size must be an integer >= 2")
    family = model_spec.get("family")
    if family == "tide":
        return TideLM(model_spec, vocab_size)
    if family == "transformer":
        return TransformerLM(model_spec, vocab_size)
    raise ValueError(f"unsupported model family: {family!r}")
