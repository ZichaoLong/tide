"""GDN/SSD §2 and §4: input-only coefficients including fixed-time conv."""
import torch
from .convolution_packed import convolve
from .packing import project
from .tensors import norm
from .normalization import epsilon_l2
from .numeric_time import passive_decay


def coefficients(memory, plan, old, x, gap):
    c = memory.config
    raw = project(x, memory.projection)
    if c.memory_family == "ssd":
        _, raw, dt = raw.split((c.heads*c.value_width, c.conv_width, c.heads), -1)
        dt = torch.nn.functional.softplus(dt+memory.delta_bias)
    convolved, caches = convolve(memory.conv, plan, old, raw)
    projected = torch.nn.functional.silu(convolved)
    decay = passive_decay(gap[:, None], c.matrix_rates, x)
    if c.memory_family == "gated_delta":
        _, key, value = projected.reshape(-1, c.heads, 2*c.key_width+c.value_width).split(
            (c.key_width, c.key_width, c.value_width), -1)
        key = epsilon_l2(key, c.qk_epsilon)
        control = norm(plan.content, c.epsilon)
        dt = torch.nn.functional.softplus(project(control, memory.delta)+memory.delta_bias)
        gain = plan.weight[:, None]*torch.sigmoid(project(control, memory.write)+memory.write_bias)
        retain = torch.exp(-plan.weight[:, None]*memory.log_rate.exp()*dt)
        a = torch.eye(c.key_width, dtype=x.dtype, device=x.device)-gain[..., None, None]*key.unsqueeze(-1)*key.unsqueeze(-2)
        a = a*(decay*retain)[..., None, None]
        b = gain[..., None, None]*value.unsqueeze(-1)*key.unsqueeze(-2)
        return a, b, True, caches
    value, key, _ = projected.split((c.heads*c.value_width, c.groups*c.key_width, c.groups*c.key_width), -1)
    value = value.reshape(-1, c.heads, c.value_width)
    key = key.reshape(-1, c.groups, c.key_width).repeat_interleave(c.heads//c.groups, dim=1)
    step = plan.weight[:, None]*dt
    a = (decay*torch.exp(-memory.log_rate.exp()*step))[..., None, None]
    b = step[..., None, None]*value.unsqueeze(-1)*key.unsqueeze(-2)
    return a, b, False, caches
