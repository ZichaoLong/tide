"""Linear attention and gated DeltaRule with independent recurrence/scan paths."""
import torch
from torch.nn.functional import elu
from .records import State


def matrix_scan(a, b, initial):
    """Exact dense affine-transform contract; a correctness scan, not a fast delta kernel."""
    stride = 1
    while stride < len(b):
        b = torch.cat((b[:stride], b[stride:] + a[stride:] @ b[:-stride]))
        a = torch.cat((a[:stride], a[stride:] @ a[:-stride]))
        stride *= 2
    return b + a @ initial


class MatrixMemory:
    sequence_contract = True
    epsilon = 1e-6

    def __init__(self, kind):
        self.kind = kind

    def initial(self, w):
        d = len(w.bias)
        slots = {"matrix": w.bias.new_zeros((d, d))}
        if self.kind == "linear":
            slots["normalizer"] = torch.zeros_like(w.bias)
        return State(torch.zeros_like(w.bias), slots=slots)

    def project(self, w, h):
        q, k, v = (h @ w.extra[name] for name in ("mem_q", "mem_k", "mem_v"))
        if self.kind == "linear":
            return elu(q) + 1, elu(k) + 1, v
        def normalize(x):
            return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(self.epsilon)
        return normalize(q), normalize(k), v

    def output(self, w, q, matrix, normalizer=None):
        value = (q.unsqueeze(-2) @ matrix).squeeze(-2)
        if normalizer is not None:
            value = value / ((q * normalizer).sum(-1, keepdim=True) + self.epsilon)
        return value @ w.extra["mem_out"]

    def step(self, w, old, h, time):
        q, k, v = self.project(w, h)
        if self.kind == "linear":
            matrix = old.slots["matrix"] + k.unsqueeze(-1) * v.unsqueeze(-2)
            z = old.slots["normalizer"] + k
            slots = {"matrix": matrix, "normalizer": z}
            value = self.output(w, q, matrix, z)
        else:
            beta = (h @ w.extra["mem_beta"]).sigmoid()
            a = (h @ w.extra["mem_decay"]).sigmoid()
            decayed = a * old.slots["matrix"]
            error = v - k @ decayed
            matrix = decayed + beta * k.unsqueeze(-1) * error.unsqueeze(-2)
            slots = {"matrix": matrix}
            value = self.output(w, q, matrix)
        return State(value, time, old.observations + 1, slots)

    def sequence(self, w, old, h, times):
        q, k, v = self.project(w, h)
        if self.kind == "linear":
            matrix = (k.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(0) + old.slots["matrix"]
            z = k.cumsum(0) + old.slots["normalizer"]
            values = self.output(w, q, matrix, z)
        else:
            beta = (h @ w.extra["mem_beta"]).sigmoid()[:, None, None]
            decay = (h @ w.extra["mem_decay"]).sigmoid()[:, None, None]
            eye = torch.eye(h.shape[-1], dtype=h.dtype, device=h.device)
            a = decay * (eye - beta * k.unsqueeze(-1) * k.unsqueeze(-2))
            b = beta * k.unsqueeze(-1) * v.unsqueeze(-2)
            matrix = matrix_scan(a, b, old.slots["matrix"])
            values = self.output(w, q, matrix)
        states = []
        for i, time in enumerate(times):
            slots = {"matrix": matrix[i]}
            if self.kind == "linear":
                slots["normalizer"] = z[i]
            states.append(State(values[i], time, old.observations + i + 1, slots))
        return states

    def validate(self, w, state):
        expected = {"matrix", "normalizer"} if self.kind == "linear" else {"matrix"}
        d = len(w.bias)
        if set(state.slots) != expected or state.slots["matrix"].shape != (d, d):
            raise ValueError("invalid matrix-memory slots")
        if self.kind == "linear":
            z = state.slots["normalizer"]
            if z.shape != (d,) or (z < 0).any():
                raise ValueError("linear attention needs a nonnegative width-sized normalizer")
