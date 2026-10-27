"""0770593 KV §1–4: unrotated private entries, original-time windows and Read.

State.value is an inert transport-width zero; exact mathematical state is in
times/int64, keys, values, summaries and weights. Decode never replays updates.
"""
import math
import torch
from ..history import increment
from ..records import State
from ..state_program import StateProgram
from .tensors import Received, norm, parameter, rotate_relative
from .attention_mean import attention_mean


class KVMemory(StateProgram):
    sequence_contract = True  # Given complete local driving content/time sequence.
    joint_sequence = True
    integer_slots = frozenset({"times"})

    def __init__(self, config, profile, base, generator, dtype):
        super().__init__()
        self.config, self.profile, self.base = config, profile, base
        d, ds, rank = config.width, config.summary_width, config.adapter_rank
        self.adapter_u = parameter((2*d, rank), generator, dtype)
        self.adapter_v = parameter((rank, d), generator, dtype)
        self.summary = parameter((ds, d), generator, dtype)
        if config.memory_read == "qry":
            self.read_q = parameter((ds, d), generator, dtype)
            self.read_k = parameter((ds, d), generator, dtype)
            self.read_v = parameter((ds, d), generator, dtype)

    def initial(self, w):
        c = self.config
        return State(torch.zeros_like(w.bias), slots={
            "times": torch.empty(0, device=w.bias.device, dtype=torch.int64),
            "keys": w.bias.new_empty((0, c.width)), "values": w.bias.new_empty((0, c.width)),
            "summaries": w.bias.new_empty((0, c.summary_width)), "weights": w.bias.new_empty(0)})

    def visible(self, state, time):
        age = time-state.slots["times"]
        mask = age >= 0
        if self.config.window:
            mask = mask & (age < self.config.window)
        return mask, age

    def current(self, u):
        return norm(u, self.config.epsilon) @ self.summary.t()

    def packed_sequence(self, w, old, batch):
        from .kv_packed import packed
        return packed(self, old, batch)

    def preparation_key(self, content):
        from .kv_projection import key
        return key(self, content)

    def prepare_shared(self, content):
        from .kv_projection import prepare
        return prepare(self, content)

    def batch_autograd_key(self, old, view, time):
        return (bool(view.value[0] > 0),)

    def sequence(self, w, old, values, times, views=None):
        from .packing import sequence
        return sequence(self, w, old, values, times, views)

    def step(self, w, old, content, time):
        h = Received.decode(content.value)
        if h.mass <= 0:
            return old  # Includes expired entries: exact state, not effective projection.
        if time <= old.last_time:
            raise ValueError("KV write times must strictly increase")
        mask, _ = self.visible(old, time)
        nu = norm(h.content, self.config.epsilon)
        key, value = ((self.base+self.adapter_u@self.adapter_v)@nu).chunk(2)
        entries = {"keys": key, "values": value, "summaries": self.current(h.content),
                   "weights": h.mass.clamp(max=1),
                   "times": torch.tensor(time, dtype=torch.int64, device=h.mass.device)}
        slots = {name: torch.cat((old.slots[name][mask], entry.unsqueeze(0))) for name, entry in entries.items()}
        return State(old.value, time, increment(old.observations), slots)

    def mem(self, state, time, u):
        c = self.config
        mask, age = self.visible(state, time)
        s = {name: value[mask] for name, value in state.slots.items()}
        age = age[mask].to(u.dtype)
        weights = (s["weights"]*torch.exp(-c.beta_short*age), s["weights"])
        totals = tuple(w.sum() for w in weights)
        if c.memory_read == "sum":
            values = tuple((w.unsqueeze(-1)*s["summaries"]).sum(0)/total.clamp(min=1)
                           for w, total in zip(weights, totals))
        else:
            q = self.read_q@norm(u, c.epsilon)
            k = rotate_relative(s["keys"]@self.read_k.t(), -age/c.position_scale, c.read_frequencies)
            logits = k@q/math.sqrt(c.summary_width)
            v = s["values"]@self.read_v.t()
            values = tuple(weighted_read(logits, v, w, total) for w, total in zip(weights, totals))
        return *values, *totals

    def validate(self, w, state):
        slots, c = state.slots, self.config
        if slots.keys() != {"times", "keys", "values", "summaries", "weights"}:
            raise ValueError("invalid KV state field domain")
        n = slots["times"].numel()
        shapes = {"times": (n,), "keys": (n,c.width), "values": (n,c.width),
                  "summaries": (n,c.summary_width), "weights": (n,)}
        if any(slots[k].shape != shape for k, shape in shapes.items()):
            raise ValueError("invalid KV state field shape")
        stamps = slots["times"]
        if (stamps.dtype != torch.int64 or (stamps < 0).any() or (stamps > state.last_time).any()
                or (stamps[1:] <= stamps[:-1]).any() or state.observations < n
                or c.window and n > c.window or state.value.count_nonzero()):
            raise ValueError("invalid KV timestamp/metadata state")
        if ((slots["weights"] < 0) | (slots["weights"] > 1)).any():
            raise ValueError("invalid KV write weights")


def weighted_read(logits, values, weights, total):
    positive = weights > 0
    if not positive.any():
        return values.new_zeros(values.shape[-1])
    return total.clamp(max=1)*attention_mean(logits[positive]+weights[positive].log(), values[positive])


class KVContent(torch.nn.Module):
    """KV §4: Full reads proposal independently of the selected Read snapshot."""
    def __init__(self, config, generator, dtype):
        super().__init__()
        self.config = config
        d, df = config.width, config.ffn_width
        for name, shape in (("query", (d,d)), ("out", (d,d)), ("gate", (df,d)),
                            ("up", (df,d)), ("down", (d,df))):
            setattr(self, name, parameter(shape, generator, dtype))

    def forward(self, memory, state, time, u):
        c = self.config
        mask, age = memory.visible(state, time)
        s = state.slots
        weights = s["weights"][mask]
        total = weights.sum()
        keys = s["keys"][mask].reshape(-1, c.heads, c.width//c.heads)
        values = s["values"][mask].reshape_as(keys)
        queries = (self.query@norm(u,c.epsilon)).reshape(c.heads,-1)
        displacement = -age[mask].to(u.dtype)/c.position_scale
        reads = []
        for h in range(c.heads):
            key = rotate_relative(keys[:,h], displacement, c.full_frequencies[h])
            logits = key@queries[h]/math.sqrt(c.width//c.heads)+c.distance_slopes[h]*displacement
            reads.append(weighted_read(logits, values[:,h], weights, total))
        x = u+c.alpha_att*(self.out@torch.cat(reads))
        normal = norm(x,c.epsilon)
        return x+c.alpha_ff*(self.down@(torch.nn.functional.silu(self.gate@normal)*(self.up@normal)))
