"""0770593 affine-delta §1–4; common exact recurrent state and summaries.

`time` stores mathematical s (initially 0). State.last_time is the engine's
last positive observation (-1 before any write). They are not floating slots.
"""
import torch
from ..history import increment
from ..coordinates import integers
from ..records import State
from ..state_program import StateProgram
from .convolution import ConvCache, FixedTimeConv
from .tensors import Received, norm, parameter


class RecurrentMemory(StateProgram):
    sequence_contract = True

    def __init__(self, config, profile, generator, dtype):
        super().__init__()
        self.config, self.profile = config, profile
        self.summary = parameter((config.summary_width, config.width), generator, dtype)
        self.integer_slots = frozenset({"time"} if config.basic else {"time", "conv_times"})
        if not config.basic:
            self.conv = FixedTimeConv(config.conv_width, config.conv_size, config.conv_stride,
                                      generator=generator, dtype=dtype)
            if config.memory_read == "state":
                self.read_out = parameter((config.summary_width, config.heads*config.value_width), generator, dtype)

    def initial(self, w):
        c, ref = self.config, w.bias
        slots = {"time": torch.zeros((), dtype=torch.int64, device=ref.device),
                 "H": ref.new_zeros(c.matrix_shape), "Z": ref.new_zeros((2, c.summary_width)),
                 "W": ref.new_zeros(2)}
        if not c.basic:
            slots.update(conv_times=torch.empty(0, dtype=torch.int64, device=ref.device),
                         conv_values=ref.new_empty((0, c.conv_width)))
        return State(torch.zeros_like(ref), slots=slots)

    @staticmethod
    def cache(state):
        return ConvCache(state.slots["conv_times"], state.slots["conv_values"])

    def current(self, u):
        return self.summary@norm(u, self.config.epsilon)

    def effective(self, state, time):
        integers("recurrent query time", time)
        if time < 0:
            raise ValueError("negative recurrent query time")
        c, s = self.config, state.slots
        delta = (s["time"].new_tensor(time)-s["time"]).clamp_min(0).to(s["H"].dtype)
        rates = (c.lambda_short, 0.) if c.basic else c.matrix_rates
        matrix_decay = torch.exp(-s["H"].new_tensor(rates)*delta)
        summary_decay = torch.exp(-s["Z"].new_tensor((c.lambda_short, 0.))*delta)
        return (matrix_decay[:, None, None]*s["H"], summary_decay[:, None]*s["Z"], summary_decay*s["W"])

    def step(self, w, old, content, time):
        h = Received.decode(content.value)
        if h.mass <= 0:
            return old
        if time <= old.last_time:
            raise ValueError("recurrent write times must strictly increase")
        H, Z, W = self.effective(old, time)
        weight = h.mass.clamp(max=1)
        matrix, raw = self.update_matrix(old, time, h.content, weight, H)
        slots = {"time": old.slots["time"].new_tensor(time), "H": matrix,
                 "Z": Z+weight*self.current(h.content), "W": W+weight}
        if not self.config.basic:
            cache = self.conv.write(self.cache(old), time, weight, raw)
            slots.update(conv_times=cache.times, conv_values=cache.values)
        return State(old.value, time, increment(old.observations), slots)

    def mem(self, state, time, u):
        H, Z, W = self.effective(state, time)
        values = Z/W.clamp(min=1).unsqueeze(-1)
        if self.config.memory_read == "state":
            values = (values[0], self.read_out@self.matrix_read(state, time, u).flatten())
        return values[0], values[1], W[0], W[1]

    def validate(self, w, state):
        c, s = self.config, state.slots
        shapes = {"time": (), "H": c.matrix_shape, "Z": (2, c.summary_width), "W": (2,)}
        if not c.basic:
            shapes.update(conv_times=(s.get("conv_times", w.bias).numel(),),
                          conv_values=(s.get("conv_times", w.bias).numel(), c.conv_width))
        if s.keys() != shapes.keys() or any(s[k].shape != shape for k, shape in shapes.items()):
            raise ValueError("invalid recurrent state field domain/shape")
        if (s["time"].dtype != torch.int64 or s["time"].requires_grad or s["time"] != max(0, state.last_time)
                or (s["W"] < 0).any() or state.value.count_nonzero()
                or (state.observations == 0) != (state.last_time == -1)):
            raise ValueError("invalid recurrent timestamp/metadata state")
        if not c.basic:
            self.cache(state).validate(c.conv_width, self.conv.horizon)
            if (s["conv_times"] > state.last_time).any() or len(s["conv_times"]) > state.observations:
                raise ValueError("invalid recurrent convolution clock/count")


class BasicMemory(RecurrentMemory):
    def __init__(self, config, profile, generator, dtype):
        super().__init__(config, profile, generator, dtype)
        c = config
        for name, shape in (("key", (c.key_width, c.width)), ("value", (c.value_width, c.width)),
                            ("retain", (c.width,)), ("retain_bias", ())):
            setattr(self, name, parameter(shape, generator, dtype))
        if c.memory_family == "delta":
            self.write = parameter((c.width,), generator, dtype)
            self.write_bias = parameter((), generator, dtype)
        if c.memory_read == "state":
            self.read_query = parameter((c.key_width, c.width), generator, dtype)
            self.read_out = parameter((2, c.summary_width, c.value_width), generator, dtype)

    def update_matrix(self, state, time, u, weight, H):
        x = norm(u, self.config.epsilon)
        key = self.key@x
        key = key/key.norm().clamp(min=1)
        value = self.value@x
        retained = torch.exp(-weight*torch.nn.functional.softplus(self.retain@x+self.retain_bias))*H
        if self.config.memory_family == "affine":
            return retained+weight*value.unsqueeze(-1)*key, None
        beta = weight*torch.sigmoid(self.write@x+self.write_bias)
        return retained+beta*(value-retained@key).unsqueeze(-1)*key, None

    def mem(self, state, time, u):
        if self.config.memory_read == "sum":
            return super().mem(state, time, u)
        H, _, W = self.effective(state, time)
        values = (self.read_out@(H@(self.read_query@norm(u, self.config.epsilon))).unsqueeze(-1)).squeeze(-1)
        return values[0], values[1], W[0], W[1]


class RecurrentContent(torch.nn.Module):
    def __init__(self, config, generator, dtype):
        super().__init__()
        self.config = config
        for name, shape in (("gate", (config.ffn_width, config.width)),
                            ("up", (config.ffn_width, config.width)), ("down", (config.width, config.ffn_width))):
            setattr(self, name, parameter(shape, generator, dtype))

    def finish(self, u, mix):
        c = self.config
        x = u+c.alpha_mem*mix
        normal = norm(x, c.epsilon)
        return x+c.alpha_ff*(self.down@(torch.nn.functional.silu(self.gate@normal)*(self.up@normal)))


class BasicContent(RecurrentContent):
    def __init__(self, config, generator, dtype):
        super().__init__(config, generator, dtype)
        self.query = parameter((config.key_width, config.width), generator, dtype)
        self.out = parameter((2, config.width, config.value_width), generator, dtype)

    def forward(self, memory, state, time, u):
        H, _, _ = memory.effective(state, time)
        reads = H@(self.query@norm(u, self.config.epsilon))
        mix = (self.out@reads.unsqueeze(-1)).squeeze(-1).sum(0)
        return self.finish(u, mix)
