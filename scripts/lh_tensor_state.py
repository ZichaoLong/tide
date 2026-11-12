"""Independent padded KV bank: actual writes, exact int64 clocks and decoding.

Padding is representation only. Raw terminal retention uses the last actual
positive write, so a silent window does not discard previously expired entries.
The bank is a baseline projection, not the public graph checkpoint format.
"""
import torch
from lh_tensor_math import norm, project

FLOATS = ("keys", "values", "summaries", "weights")


def empty(nodes, x):
    n, b, d = len(nodes), x.shape[0], x.shape[-1]
    return dict(times=torch.empty(0, dtype=torch.int64, device=x.device),
        valid=torch.empty((n, b, 0), dtype=torch.bool, device=x.device),
        keys=x.new_empty(n, b, 0, d), values=x.new_empty(n, b, 0, d),
        summaries=x.new_empty(n, b, 0, nodes[0].kernel.config.summary_width),
        weights=x.new_empty(n, b, 0), last_times=torch.full((n, b), -1, dtype=torch.int64, device=x.device),
        observations=torch.zeros((n, b), dtype=torch.int64, device=x.device))


def validate(bank, nodes, x, first):
    expected = empty(nodes, x)
    if bank.keys() != expected.keys():
        raise ValueError("LH raw bank field domain differs")
    stamps = bank["times"]
    if stamps.ndim != 1:
        raise ValueError("LH raw bank times must be an int64 vector")
    for name, value in bank.items():
        shape = list(expected[name].shape)
        if name == "times":
            shape[0] = len(stamps)
        elif name in (*FLOATS, "valid"):
            shape[2] = len(stamps)
        if (list(value.shape) != shape or value.device != x.device or value.dtype != expected[name].dtype
                or not torch.isfinite(value).all()):
            raise ValueError("LH raw bank shape/dtype/device/value differs: "+name)
    if ((stamps < 0).any() or (stamps >= first).any() or (stamps[1:] <= stamps[:-1]).any()
            or (bank["last_times"] >= first).any() or (bank["last_times"] < -1).any()
            or (bank["observations"] < bank["valid"].sum(-1)).any()
            or ((bank["last_times"] == -1) != (bank["observations"] == 0)).any()
            or ((stamps > bank["last_times"][..., None]) & bank["valid"]).any()
            or (bank["weights"] < 0).any() or (bank["weights"] > 1).any()):
        raise ValueError("LH raw bank timestamps, counts or weights outside domain")
    return bank


def append(nodes, mass, u, times, left):
    memories = [n.kernel for n in nodes]
    c = memories[0].config
    normal = norm(u, c.epsilon)
    kv = project(normal, [m.base for m in memories])+project(
        project(normal, [m.adapter_v for m in memories]), [m.adapter_u for m in memories])
    keys, values = kv.chunk(2, -1)
    fresh = dict(keys=keys, values=values, summaries=project(normal, [m.summary for m in memories]),
                 weights=mass.clamp_max(1), valid=mass.detach() > 0)
    bank = {name: torch.cat((left[name], fresh[name]), 2) for name in (*FLOATS, "valid")}
    bank["times"] = torch.cat((left["times"], times))
    last = times.expand_as(mass).masked_fill(~fresh["valid"], -1).amax(-1)
    bank["last_times"] = torch.maximum(left["last_times"], last)
    count = fresh["valid"].sum(-1)
    if (left["observations"] > 2**63-1-count).any():
        raise ValueError("LH raw bank observation count overflows int64")
    bank["observations"] = left["observations"]+count
    terminal = dict(bank)
    if c.window:
        actual = fresh["valid"].any(-1)
        retain = bank["last_times"][..., None]-bank["times"] < c.window
        terminal["valid"] = bank["valid"] & (~actual[..., None] | retain)
    # Remove common all-padding columns only; no per-event autograd objects.
    keep = terminal["valid"].any(dim=(0, 1)).nonzero().flatten()
    terminal = {name: value.index_select(0 if name == "times" else 2, keep)
                if name in (*FLOATS, "valid", "times") else value for name, value in terminal.items()}
    return bank, terminal


def decode(banks, members):
    """Return actual raw slots/metadata at (sample,node), with gradients intact."""
    result = {}
    for region, nodes in enumerate(members):
        bank = banks[region]
        for i, node in enumerate(nodes):
            for b in range(bank["valid"].shape[1]):
                valid = bank["valid"][i, b]
                result[b, node] = dict(slots=dict(times=bank["times"][valid],
                    **{name: bank[name][i, b, valid] for name in FLOATS}),
                    last_time=int(bank["last_times"][i, b]), observations=int(bank["observations"][i, b]))
    return result


def detached(banks):
    return {r: {k: v.detach().clone() for k, v in bank.items()} for r, bank in banks.items()}
