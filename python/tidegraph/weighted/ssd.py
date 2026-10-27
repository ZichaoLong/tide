"""0770593 SSD §1–3: grouped B/C, dt outer write, gate before global RMS."""
import torch
from .recurrent import RecurrentMemory, RecurrentContent
from .tensors import norm, parameter


class SSDMemory(RecurrentMemory):
    def __init__(self, config, profile, generator, dtype):
        super().__init__(config, profile, generator, dtype)
        c = config
        self.projection = parameter((c.heads*c.value_width+c.conv_width+c.heads, c.width), generator, dtype)
        self.delta_bias = parameter((c.heads,), generator, dtype)
        self.log_rate = parameter((c.heads,), generator, dtype)

    def features(self, state, time, u):
        c = self.config
        gate, raw, delta = (self.projection@norm(u, c.epsilon)).split((c.heads*c.value_width, c.conv_width, c.heads))
        value, b, query = torch.nn.functional.silu(self.conv(self.cache(state), time, raw)).split(
            (c.heads*c.value_width, c.groups*c.key_width, c.groups*c.key_width))
        b = b.reshape(c.groups, c.key_width).repeat_interleave(c.heads//c.groups, dim=0)
        query = query.reshape(c.groups, c.key_width).repeat_interleave(c.heads//c.groups, dim=0)
        delta = torch.nn.functional.softplus(delta+self.delta_bias)
        return raw, gate, value.reshape(c.heads, c.value_width), b, query, delta

    def update_matrix(self, state, time, u, weight, H):
        raw, _, value, b, _, delta = self.features(state, time, u)
        step = weight*delta
        retain = torch.exp(-self.log_rate.exp()*step)
        return retain[:, None, None]*H+step[:, None, None]*value.unsqueeze(-1)*b.unsqueeze(-2), raw

    def matrix_read(self, state, time, u):
        _, _, _, _, query, _ = self.features(state, time, u)
        H, _, _ = self.effective(state, time)
        return (H@query.unsqueeze(-1)).squeeze(-1)


class SSDContent(RecurrentContent):
    def __init__(self, config, generator, dtype):
        super().__init__(config, generator, dtype)
        c = config
        self.direct = parameter((c.heads,), generator, dtype)
        self.rho = parameter((c.heads*c.value_width,), generator, dtype)
        self.out = parameter((c.width, c.heads*c.value_width), generator, dtype)

    def forward(self, memory, state, time, u):
        c = self.config
        _, gate, value, _, query, _ = memory.features(state, time, u)
        H, _, _ = memory.effective(state, time)
        read = (H@query.unsqueeze(-1)).squeeze(-1)
        y = (read+self.direct[:, None]*value).flatten()*torch.nn.functional.silu(gate)
        mix = self.out@(self.rho*y/torch.sqrt(c.output_epsilon+y.square().mean()))
        return self.finish(u, mix)
