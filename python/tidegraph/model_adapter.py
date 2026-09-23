"""Finite PyTorch model adapter: RMSNorm, explicit-position RoPE GQA and SwiGLU.

This module-level example consumes Tide's representative attention/Full weights.
It does not change the aggregated-event or same-fiber graph profiles. Occurrences
and positions are explicit separate interfaces, never inferred from graph time.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .graph import Graph, Node, Region
from .ops import Model


@dataclass(frozen=True)
class DecoderCache:
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    positions: tuple[torch.Tensor, ...]
    occurrences: tuple[tuple[int, ...], ...]
    parameter_identity: tuple

    def detach(self):
        return DecoderCache(tuple(x.detach().clone() for x in self.keys),
                            tuple(x.detach().clone() for x in self.values),
                            tuple(x.clone() for x in self.positions), self.occurrences,
                            self.parameter_identity)


def rms_norm(x, scale, epsilon):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * scale


def rotary(x, positions, base):
    """Interleaved even/odd pairs; positions are explicit int64, not token indices."""
    inv = base ** (-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype) / x.shape[-1])
    angle = positions.to(x.dtype)[:, None, None] * inv
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a*angle.cos()-b*angle.sin(), a*angle.sin()+b*angle.cos()), -1).flatten(-2)


class TinyDecoderAdapter(nn.Module):
    profile = "tiny-rms-rope-gqa-swiglu-v1"

    def __init__(self, width=8, query_heads=2, kv_heads=1, *, window=0,
                 epsilon=1e-5, rope_base=10000., seed=7, dtype=torch.float64):
        super().__init__()
        if (width < 1 or query_heads < 1 or kv_heads < 1 or width % query_heads
                or query_heads % kv_heads or (width // query_heads) % 2 or window < 0
                or not math.isfinite(epsilon) or epsilon <= 0
                or not math.isfinite(rope_base) or rope_base <= 0):
            raise ValueError("invalid tiny decoder dimensions/norm/RoPE policy")
        graph = Graph((Node(0, memory="attention", full="swiglu", query_heads=query_heads,
                            kv_heads=kv_heads, window=window),), (), (Region(1),), (0,), (0,))
        self.weights = Model(graph, width, seed, dtype).nodes[0]
        self.norm_attention = nn.Parameter(torch.ones(width, dtype=dtype))
        self.norm_ffn = nn.Parameter(torch.ones(width, dtype=dtype))
        self.width, self.query_heads, self.kv_heads = width, query_heads, kv_heads
        self.window, self.epsilon, self.rope_base = window, epsilon, rope_base

    def _identity(self):
        # Scope: same live adapter only. Checkpoints must reconstruct/re-prefill
        # this example's cache; it is not a portable graph continuation schema.
        return tuple((id(p), p._version) for p in self.parameters())

    def initial(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("positive batch required")
        w = self.norm_attention
        shape = (0, self.kv_heads, self.width // self.query_heads)
        return DecoderCache(tuple(w.new_empty(shape) for _ in range(batch)),
                            tuple(w.new_empty(shape) for _ in range(batch)),
                            tuple(torch.empty(0, device=w.device, dtype=torch.int64) for _ in range(batch)),
                            ((),)*batch, self._identity())

    def _validate(self, x, positions, occurrences, valid, cache):
        if (x.ndim != 3 or x.shape[-1] != self.width or x.dtype != self.norm_attention.dtype
                or x.device != self.norm_attention.device or not torch.isfinite(x).all()):
            raise ValueError("invalid input shape/device/dtype/value")
        for tensor in (positions, occurrences):
            if tensor.shape != x.shape[:2] or tensor.dtype != torch.int64 or tensor.device != x.device:
                raise ValueError("explicit int64 [B,T] positions and occurrences required")
        if valid.shape != x.shape[:2] or valid.dtype != torch.bool or valid.device != x.device:
            raise ValueError("boolean [B,T] valid mask required")
        if (cache.parameter_identity != self._identity() or len(cache.keys) != len(x)
                or len(cache.values) != len(x) or len(cache.positions) != len(x)
                or len(cache.occurrences) != len(x)):
            raise ValueError("cache owner/parameter update/batch mismatch; reset and prefill")
        d = self.width // self.query_heads
        for b in range(len(x)):
            k, v, p, ledger = cache.keys[b], cache.values[b], cache.positions[b], cache.occurrences[b]
            if (k.shape != v.shape or k.shape != (len(p), self.kv_heads, d)
                    or k.dtype != x.dtype or v.dtype != x.dtype or k.device != x.device or v.device != x.device
                    or p.dtype != torch.int64 or p.device != x.device or p.ndim != 1
                    or not torch.isfinite(k).all() or not torch.isfinite(v).all()
                    or len(p) != min(len(ledger), self.window or len(ledger))
                    or ledger != tuple(range(len(ledger)))):
                raise ValueError("invalid cache shape or occurrence ledger")
            pos = positions[b, valid[b]]; ids = occurrences[b, valid[b]].tolist()
            all_pos = torch.cat((p, pos))
            if (all_pos < 0).any() or (all_pos[1:] <= all_pos[:-1]).any():
                raise ValueError("valid positions must strictly increase")
            if ids != list(range(len(ledger), len(ledger) + len(ids))):
                raise ValueError("input occurrences must continue the explicit ledger")

    def forward(self, x, positions, occurrences, valid, cache=None, *, layout="BTD"):
        if layout not in {"BTD", "TBD"}:
            raise ValueError("layout must be BTD or TBD")
        x = x if layout == "BTD" else x.transpose(0, 1)
        cache = self.initial(len(x)) if cache is None else cache
        self._validate(x, positions, occurrences, valid, cache)
        d, repeat = self.width // self.query_heads, self.query_heads // self.kv_heads
        w = self.weights.extra
        rows, keys, values, saved_positions, ledgers = [], [], [], [], []
        stats = dict(attention_sequence_calls=0, max_sequence=0, valid_events=0, padded_events=int((~valid).sum()))
        for b in range(len(x)):
            index = valid[b].nonzero().flatten(); length = len(index)
            if not length:
                rows.append(x[b]*0); keys.append(cache.keys[b]); values.append(cache.values[b])
                saved_positions.append(cache.positions[b]); ledgers.append(cache.occurrences[b]); continue
            h = x[b, index]; pos = positions[b, index]
            normalized = rms_norm(h, self.norm_attention, self.epsilon)
            q = rotary((normalized @ w["attn_q"]).reshape(length, self.query_heads, d), pos, self.rope_base)
            k = rotary((normalized @ w["attn_k"]).reshape(length, self.kv_heads, d), pos, self.rope_base)
            v = (normalized @ w["attn_v"]).reshape(length, self.kv_heads, d)
            count = len(cache.keys[b]); k = torch.cat((cache.keys[b], k)); v = torch.cat((cache.values[b], v))
            scores = q.transpose(0, 1) @ k.repeat_interleave(repeat, 1).permute(1, 2, 0) / math.sqrt(d)
            end = count + torch.arange(length, device=x.device)[:, None]
            key_index = torch.arange(len(k), device=x.device)[None, :]
            mask = key_index <= end
            if self.window: mask = mask & (key_index > end - self.window)
            attention = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v.repeat_interleave(repeat, 1).transpose(0, 1)
            h = h + attention.transpose(0, 1).reshape(length, self.width) @ w["attn_out"]
            normalized = rms_norm(h, self.norm_ffn, self.epsilon)
            h = h + (torch.nn.functional.silu(normalized @ w["ffn_gate"])
                     * (normalized @ w["ffn_up"])) @ w["ffn_down"]
            rows.append((x[b]*0).index_copy(0, index, h))
            start = max(0, len(k)-self.window) if self.window else 0
            keys.append(k[start:].clone()); values.append(v[start:].clone())
            saved_positions.append(torch.cat((cache.positions[b], pos))[start:].clone())
            ledgers.append(cache.occurrences[b] + tuple(occurrences[b, index].tolist()))
            stats["attention_sequence_calls"] += 1
            stats["max_sequence"] = max(stats["max_sequence"], length)
            stats["valid_events"] += length
        output = torch.stack(rows)
        new_cache = DecoderCache(tuple(keys), tuple(values), tuple(saved_positions), tuple(ledgers), self._identity())
        return output if layout == "BTD" else output.transpose(0, 1), new_cache, stats
