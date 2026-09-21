"""State-kernel programs. Scheduling sees State and exact block capabilities."""
import torch
from torch.nn.functional import softplus
from .records import State
from .scan import affine_scan


class EMA:
    sequence_contract = True

    def initial(self, w):
        return State(torch.zeros_like(w.bias))

    def step(self, w, old, h, time):
        return State(w.decay.sigmoid() * old.value + h, time, old.observations + 1)

    def sequence(self, w, old, h, times):
        values = affine_scan(w.decay.sigmoid().expand_as(h), h, old.value)
        return [State(v, t, old.observations + i + 1) for i, (v, t) in enumerate(zip(values, times))]

    def validate(self, w, state):
        if state.slots:
            raise ValueError("EMA state has unexpected slots")


class DiagonalSSM:
    """Input-selective diagonal recurrence, observation-clock discretization.

    dt=softplus(h W_dt); a=exp(-softplus(A)*dt)
    m'=a*m+dt*(h W_b); value=(h W_c)*m'+skip*h.
    """
    sequence_contract = True

    def initial(self, w):
        return State(torch.zeros_like(w.bias), slots={"memory": torch.zeros_like(w.bias)})

    def coefficients(self, w, h):
        dt = softplus(h @ w.extra["ssm_dt"])
        a = torch.exp(-softplus(w.extra["ssm_a"]) * dt)
        return a, dt * (h @ w.extra["ssm_b"])

    def step(self, w, old, h, time):
        a, b = self.coefficients(w, h)
        memory = a * old.slots["memory"] + b
        value = (h @ w.extra["ssm_c"]) * memory + w.extra["ssm_skip"] * h
        return State(value, time, old.observations + 1, {"memory": memory})

    def sequence(self, w, old, h, times):
        a, b = self.coefficients(w, h)
        memory = affine_scan(a, b, old.slots["memory"])
        values = (h @ w.extra["ssm_c"]) * memory + w.extra["ssm_skip"] * h
        return [State(v, t, old.observations + i + 1, {"memory": m})
                for i, (v, m, t) in enumerate(zip(values, memory, times))]

    def validate(self, w, state):
        if set(state.slots) != {"memory"} or state.slots["memory"].shape != w.bias.shape:
            raise ValueError("SSM requires one width-sized memory slot")


def kernel(name):
    if name == "ema":
        return EMA()
    if name == "ssm":
        return DiagonalSSM()
    raise ValueError(f"unknown state kernel: {name}")


def reset(state):
    return State(state.value * 0, state.last_time, state.observations,
                 {name: value * 0 for name, value in state.slots.items()})
