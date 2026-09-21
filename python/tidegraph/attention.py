"""Causal GQA over aggregated event observations; no implicit token-position policy."""
from collections import defaultdict
import math
import torch
from .history import increment
from .records import State
from .packing import PackedSequence
from .content import as_content, Content


class Attention:
    sequence_contract = True
    joint_sequence = True

    def __init__(self, query_heads, kv_heads, window):
        self.query_heads, self.kv_heads, self.window = query_heads, kv_heads, window

    def initial(self, w):
        shape = (0, self.kv_heads, len(w.bias) // self.query_heads)
        return State(torch.zeros_like(w.bias), slots={k: w.bias.new_zeros(shape) for k in ("key", "value")})

    def step(self, w, old, h, time):
        """Readable independent oracle: one query head at a time, no dense causal mask."""
        h = as_content(h).value
        d = len(w.bias) // self.query_heads
        q = (h @ w.extra["attn_q"]).reshape(self.query_heads, d)
        k = (h @ w.extra["attn_k"]).reshape(1, self.kv_heads, d)
        v = (h @ w.extra["attn_v"]).reshape(1, self.kv_heads, d)
        k = torch.cat((old.slots["key"], k)); v = torch.cat((old.slots["value"], v))
        if self.window:
            k, v = k[-self.window:], v[-self.window:]
        heads = []
        for i in range(self.query_heads):
            j = i // (self.query_heads // self.kv_heads)
            p = (k[:, j] @ q[i] / math.sqrt(d)).softmax(0)
            heads.append(p @ v[:, j])
        value = torch.cat(heads) @ w.extra["attn_out"]
        return State(value, time, increment(old.observations), {"key": k.clone(), "value": v.clone()})

    def sequence(self, w, old, h, times, views=None):
        batch = PackedSequence(h, [0, len(times)], [(0, 0)], times, [Content(v) for v in h] if views is None else views)
        return self.packed_sequence(w, [old], batch)[0]

    def packed_sequence(self, w, old, batch):
        """Group exact cache/sequence lengths: [B,Hq,T,K+T], never cross-sample scores."""
        batch.validate()
        if len(old) != len(batch.owners):
            raise ValueError("packed initial-state count mismatch")
        d = len(w.bias) // self.query_heads
        q = (batch.contents @ w.extra["attn_q"]).reshape(-1, self.query_heads, d)
        k = (batch.contents @ w.extra["attn_k"]).reshape(-1, self.kv_heads, d)
        v = (batch.contents @ w.extra["attn_v"]).reshape(-1, self.kv_heads, d)
        groups = defaultdict(list)
        for i, state in enumerate(old):
            a, b = batch.offsets[i:i+2]
            groups[len(state.slots["key"]), b-a].append(i)
        states = [None] * len(batch.contents)
        for (cache, length), ids in groups.items():
            qs, ks, vs = [], [], []
            for i in ids:
                a, b = batch.offsets[i:i+2]
                qs.append(q[a:b]); ks.append(torch.cat((old[i].slots["key"], k[a:b])))
                vs.append(torch.cat((old[i].slots["value"], v[a:b])))
            keys, values = torch.stack(ks), torch.stack(vs)
            group = self.query_heads // self.kv_heads
            scores = torch.stack(qs).transpose(1, 2) @ keys.repeat_interleave(group, 2).permute(0, 2, 3, 1)
            scores = scores / math.sqrt(d)
            end = cache + torch.arange(length, device=q.device)[:, None]
            index = torch.arange(cache + length, device=q.device)[None, :]
            mask = index <= end
            if self.window:
                mask = mask & (index > end - self.window)
            probs = scores.masked_fill(~mask, -torch.inf).softmax(-1)
            out = (probs @ values.repeat_interleave(group, 2).transpose(1, 2)).transpose(1, 2)
            out = out.reshape(len(ids), length, -1) @ w.extra["attn_out"]
            for row, i in enumerate(ids):
                for t in range(length):
                    end = cache + t + 1; begin = max(0, end - self.window) if self.window else 0
                    slots = {"key": keys[row, begin:end], "value": values[row, begin:end]}
                    if t == length - 1:
                        slots = {name: x.clone() for name, x in slots.items()}
                    j = batch.offsets[i] + t
                    value = out[row, t].clone() if t == length - 1 else out[row, t]
                    states[j] = State(value, batch.times[j], increment(old[i].observations, t + 1), slots)
        return states, len(groups)

    @staticmethod
    def reset(state):
        # Empty clone releases old storage in inference; backward still supplies connected zeros.
        return State(state.value * 0, state.last_time, state.observations,
                     {k: x[:0].clone() for k, x in state.slots.items()})

    def validate(self, w, state):
        if set(state.slots) != {"key", "value"}:
            raise ValueError("attention requires key/value slots")
        k, v = state.slots["key"], state.slots["value"]
        if (k.ndim != 3 or k.shape[1:] != (self.kv_heads, len(w.bias) // self.query_heads)
                or k.shape != v.shape or len(k) > state.observations or (self.window and len(k) > self.window)):
            raise ValueError("invalid attention cache shape/length")
