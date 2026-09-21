"""Readable LH same-fiber attention, sum pooling and ordered log-bias decay."""
import math
import torch
from .content import as_content
from .history import increment, int64
from .records import State
from .state_program import StateProgram

PROFILE = "lh-fiber-attention-sum-repeat-v1"


def advance_bias(bias, rate, ticks):
    # Structural emptiness skips absent memory; numerical zeros do not.
    if bias.numel():
        for _ in range(ticks):
            bias = bias - rate
    return bias


class FiberAttention(StateProgram):
    profile = PROFILE
    sequence_contract = True
    joint_sequence = False  # Baseline only; packed attention is a separate gate.

    def __init__(self, spec):
        super().__init__()
        self.heads = spec.query_heads
        if spec.kv_heads != self.heads or spec.window or spec.aggregation != "sum":
            raise ValueError("LH fiber attention requires equal heads, no eviction and sum Aggregate")

    def initial(self, w):
        shape = (0, self.heads, len(w.bias)//self.heads)
        return State(torch.zeros_like(w.bias), slots={"key": w.bias.new_zeros(shape),
                     "value": w.bias.new_zeros(shape), "log_bias": w.bias.new_zeros((0,))})

    def step(self, w, old, content, time):
        content = as_content(content)
        if not content.sources:
            raise ValueError("LH fiber attention requires complete source rows")
        if not int64(time) or not int64(old.last_time) or not -1 <= old.last_time < time:
            raise ValueError("fiber attention requires increasing tick times")
        observations = increment(old.observations)
        # Original LH gathers by incoming local source index, including appended
        # bridge/token slots. The public trace retains its own canonical order.
        sources = sorted(content.sources, key=lambda s: s.slot)
        x = torch.stack([s.atom.value*s.scale for s in sources])
        width = len(w.bias); d = width//self.heads
        qkv = torch.nn.functional.linear(x, w.extra["fiber_qkv"].t(), w.extra["fiber_qkv_bias"])
        q, k, v = [t.reshape(len(x), self.heads, d) for t in qkv.split(width, -1)]
        q = q * (1/math.sqrt(d))
        k = torch.cat((old.slots["key"], k)); v = torch.cat((old.slots["value"], v))
        bias = advance_bias(old.slots["log_bias"], w.extra["fiber_decay"], time-old.last_time)
        bias = torch.cat((bias, x.new_zeros((len(x),))))
        rows = []
        for query in q:
            heads = []
            for head in range(self.heads):
                probability = (k[:, head] @ query[head] + bias).softmax(0)
                heads.append(probability @ v[:, head])
            rows.append(torch.cat(heads))
        pooled = torch.stack(rows).sum(0)
        value = torch.nn.functional.linear(pooled, w.extra["fiber_out"].t(), w.extra["fiber_out_bias"])
        return State(value, time, observations, {"key": k, "value": v, "log_bias": bias})

    @staticmethod
    def reset(state):
        return State(state.value*0, state.last_time, state.observations,
                     {name: t[:0].clone() for name, t in state.slots.items()})

    def validate_weights(self, w):
        d = len(w.bias)
        if self.heads < 1 or d % self.heads:
            raise ValueError("invalid fiber attention head/width policy")
        for name, shape in (("fiber_qkv", (d, 3*d)), ("fiber_qkv_bias", (3*d,)),
                            ("fiber_out", (d, d)), ("fiber_out_bias", (d,)), ("fiber_decay", ())):
            p = w.extra.get(name)
            if (p is None or p.shape != shape or p.dtype != w.bias.dtype or p.device != w.bias.device
                    or not torch.isfinite(p).all()):
                raise ValueError("invalid fiber attention parameter")

    def validate(self, w, state):
        if set(state.slots) != {"key", "value", "log_bias"}:
            raise ValueError("fiber attention requires key/value/log_bias slots")
        k, v, bias = (state.slots[n] for n in ("key", "value", "log_bias"))
        if (k.ndim != 3 or k.shape[1:] != (self.heads, len(w.bias)//self.heads) or v.shape != k.shape
                or bias.shape != (len(k),)):
            raise ValueError("invalid fiber attention cache shape")


def decode_bias(w, state, cut):
    if not isinstance(w.kernel, FiberAttention):
        raise ValueError("fiber bias decoder requires the fiber attention profile")
    w.kernel.validate_weights(w); w.kernel.validate(w, state)
    if (not int64(cut) or not int64(state.last_time) or not int64(state.observations)
            or not -1 <= state.last_time < cut or state.observations < 0):
        raise ValueError("invalid fiber attention cut/state clock")
    if state.value.shape != w.bias.shape:
        raise ValueError("invalid fiber attention state value shape")
    for t in [state.value, *state.slots.values()]:
        if t.dtype != w.bias.dtype or t.device != w.bias.device or not torch.isfinite(t).all():
            raise ValueError("invalid fiber attention state tensor")
    return advance_bias(state.slots["log_bias"], w.extra["fiber_decay"], cut-1-state.last_time)
