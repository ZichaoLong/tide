"""KV §3–4: compact query algebra; shared history is projected once per bank."""
import math
import torch
from torch.nn import functional as F
from .tensor_describe import Algebra, describe, fields
from .column_attention import logit_attention
from .tensors import norm, rotate_relative
from .numeric_time import passive_decay, time_values


def rotate(value, coordinates, frequencies, scale):
    if not any(frequencies):
        return value
    t = time_values(coordinates, value)/scale
    # Only head configuration is traversed; N/B/Q/K remain tensor axes.
    return torch.stack([rotate_relative(value[:, :, h], t, f)
                        for h, f in enumerate(frequencies)], 2)


def attend(logits, values, weights):
    g, b, h, q, k = logits.shape
    value = logit_attention(logits.reshape(g*b, h, q, k),
        values.reshape(g*b, h, k, values.shape[-1]), weights.reshape(g*b, q, k))
    return value.reshape(g, b, h, q, values.shape[-1])


class Formula:
    def __init__(self, config, regional, metadata, active, kind):
        self.c, self.regional, self.metadata = config, regional, metadata
        self.active, self.kind = active, kind
        self.shared = set()
        self.fields = fields(regional) if kind == "read" else ("content",)

    def __call__(self, values):
        a, c, m = Algebra(values, self.shared), self.c, self.metadata
        u, weight = values["u"], values["weights"]
        weight = torch.where(m["visible"], weight[..., None, :], 0)
        normal = norm(u, c.epsilon)
        if self.kind == "read":
            short = weight*passive_decay(m["age"].clamp_min(0), c.beta_short, u)
            st, lt = short.sum(-1), weight.sum(-1)
            if c.memory_read == "sum":
                memory = ((short@values["summaries"])/st.clamp_min(1)[..., None],
                          (weight@values["summaries"])/lt.clamp_min(1)[..., None])
            elif self.active:
                q = a.linear(normal, "kernel.read_q")[:, :, None]
                k = a.linear(values["keys"], "kernel.read_k")[:, :, None]
                v = a.linear(values["values"], "kernel.read_v")[:, :, None]
                if c.read_frequencies:
                    q = rotate(q, m["qt"], (c.read_frequencies,), c.position_scale)
                    k = rotate(k, m["kt"], (c.read_frequencies,), c.position_scale)
                logits = (q@k.transpose(-1, -2))/math.sqrt(c.summary_width)
                # One QK map, two weight channels; VJP accumulates both uses.
                memory = (attend(logits, v, short)[:, :, 0]*st.clamp_max(1)[..., None],
                          attend(logits, v, weight)[:, :, 0]*lt.clamp_max(1)[..., None])
            else:
                memory = (u.new_zeros((*u.shape[:-1], c.summary_width)),)*2
            return describe(a, self.regional, c.epsilon, u, values["mass"], (*memory, st, lt))
        g, b, qlen, _ = u.shape
        if self.active:
            q = a.linear(normal, "full.query").reshape(g, b, qlen, c.heads, -1).transpose(2, 3)
            k, v = (values[key].reshape(g, b, -1, c.heads, c.width//c.heads).transpose(2, 3)
                    for key in ("keys", "values"))
            if any(c.full_frequencies):
                q = rotate(q, m["qt"], c.full_frequencies, c.position_scale)
                k = rotate(k, m["kt"], c.full_frequencies, c.position_scale)
            logits = (q@k.transpose(-1, -2))/math.sqrt(c.width//c.heads)
            if any(c.distance_slopes):
                delta = -time_values(m["age"], u)/c.position_scale
                logits = logits+u.new_tensor(c.distance_slopes)[None, None, :, None, None]*delta[:, :, None]
            mixed = attend(logits, v, weight).transpose(2, 3).reshape_as(u)
        else:
            mixed = torch.zeros_like(u)
        x = u+c.alpha_att*a.linear(mixed*weight.sum(-1).clamp_max(1)[..., None], "full.out")
        normal = norm(x, c.epsilon)
        y = a.linear(F.silu(a.linear(normal, "full.gate"))*a.linear(normal, "full.up"), "full.down")
        return dict(content=x+c.alpha_ff*y)
