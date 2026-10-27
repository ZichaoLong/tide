"""0770593 gated-delta §1–3: multihead Delta, fixed-time conv and gated RMS."""
import math
import torch
from .recurrent import RecurrentMemory, RecurrentContent
from .tensors import norm, parameter


class GatedDeltaMemory(RecurrentMemory):
    def __init__(self, config, profile, generator, dtype):
        super().__init__(config, profile, generator, dtype)
        c = config
        for name, shape in (("projection", (c.conv_width, c.width)), ("delta", (c.heads, c.width)),
                            ("delta_bias", (c.heads,)), ("write", (c.heads, c.width)),
                            ("write_bias", (c.heads,)), ("log_rate", (c.heads,))):
            setattr(self, name, parameter(shape, generator, dtype))

    def features(self, state, time, u):
        c = self.config
        raw = self.projection@norm(u, c.epsilon)
        projected = torch.nn.functional.silu(self.conv(self.cache(state), time, raw))
        q, k, value = projected.reshape(c.heads, -1).split((c.key_width, c.key_width, c.value_width), dim=-1)
        q = q/torch.sqrt(c.qk_epsilon+q.square().sum(-1, keepdim=True))/math.sqrt(c.key_width)
        k = k/torch.sqrt(c.qk_epsilon+k.square().sum(-1, keepdim=True))
        return raw, q, k, value

    def update_matrix(self, state, time, u, weight, H):
        raw, _, key, value = self.features(state, time, u)
        x = norm(u, self.config.epsilon)
        delta = torch.nn.functional.softplus(self.delta@x+self.delta_bias)
        retain = torch.exp(-weight*self.log_rate.exp()*delta)
        beta = weight*torch.sigmoid(self.write@x+self.write_bias)
        retained = retain[:, None, None]*H
        residual = value-(retained@key.unsqueeze(-1)).squeeze(-1)
        return retained+beta[:, None, None]*residual.unsqueeze(-1)*key.unsqueeze(-2), raw

    def matrix_read(self, state, time, u):
        _, query, _, _ = self.features(state, time, u)
        H, _, _ = self.effective(state, time)
        return (H@query.unsqueeze(-1)).squeeze(-1)


class GatedDeltaContent(RecurrentContent):
    def __init__(self, config, generator, dtype):
        super().__init__(config, generator, dtype)
        c = config
        self.output_gate = parameter((c.heads*c.value_width, c.width), generator, dtype)
        self.rho = parameter((c.heads, c.value_width), generator, dtype)
        self.out = parameter((c.width, c.heads*c.value_width), generator, dtype)

    def forward(self, memory, state, time, u):
        c = self.config
        read = memory.matrix_read(state, time, u)
        rms = self.rho*read/torch.sqrt(c.output_epsilon+read.square().mean(-1, keepdim=True))
        gate = torch.nn.functional.silu(self.output_gate@norm(u, c.epsilon))
        return self.finish(u, self.out@(rms.flatten()*gate))
