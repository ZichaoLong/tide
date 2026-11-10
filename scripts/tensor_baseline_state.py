"""Raw tensor state helpers for the restricted additive baselines.

No graph schedule, reference recurrence, effective-state decoding or hidden
parameter cache. Integer fields have one common time layout across samples;
floating fields retain their B axis and independent differentiable roots.
"""
import torch


def check_position(spec, node, values, start):
    if type(start) is not int or start < 0:
        raise ValueError("tensor baseline start must be a nonnegative int64 position")
    last = (start+values.shape[1]-1)*spec.stride+spec.rank(node)
    if last > 2**63-1:
        raise ValueError("tensor baseline logical time overflows int64")


def times(spec, node, values, start):
    check_position(spec, node, values, start)
    return (torch.arange(values.shape[1], device=values.device, dtype=torch.int64)+start)*spec.stride+spec.rank(node)


def validate(raw, config, values, first_time):
    """Validate a supplied exact raw state without copying or detaching it."""
    if not isinstance(raw, dict):
        raise ValueError("tensor baseline raw state must be a field dictionary")
    batch = values.shape[0]
    stamp_name = "times" if hasattr(config, "window") else "conv_times"
    if not isinstance(raw.get(stamp_name), torch.Tensor):
        raise ValueError("tensor baseline raw state timestamp field missing")
    n = raw[stamp_name].numel()
    if stamp_name == "times":
        shapes = dict(times=(n,), keys=(batch, n, config.width), values=(batch, n, config.width),
                      summaries=(batch, n, config.summary_width), weights=(batch, n))
        integer = {"times"}
    else:
        shapes = dict(time=(), H=(batch, *config.matrix_shape), Z=(batch, 2, config.summary_width),
                      W=(batch, 2), conv_times=(n,), conv_values=(batch, n, config.conv_width))
        integer = {"time", "conv_times"}
    if raw.keys() != shapes.keys():
        raise ValueError("tensor baseline raw state field domain differs")
    for name, shape in shapes.items():
        value = raw[name]
        dtype = torch.int64 if name in integer else values.dtype
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype
                or value.device != values.device or name in integer and value.requires_grad
                or not torch.isfinite(value).all()):
            raise ValueError("tensor baseline raw state shape/dtype/device/value differs: "+name)
    stamps = raw["times"] if "times" in raw else raw["conv_times"]
    if ((stamps < 0).any() or (stamps >= first_time).any()
            or (stamps[1:] <= stamps[:-1]).any()):
        raise ValueError("tensor baseline left timestamps must precede the window and increase")
    if "times" in raw:
        if ((raw["weights"] < 0).any() or (raw["weights"] > 1).any()
                or config.window and n > config.window):
            raise ValueError("invalid tensor baseline KV weights/cache")
    elif (raw["time"] < 0 or raw["time"] >= first_time or (stamps > raw["time"]).any()
          or (raw["W"] < 0).any() or n > (config.conv_size-1)*config.conv_stride+1):
        raise ValueError("invalid tensor baseline recurrent clock/count/cache")
    return raw


def kv_join(config, times, key, value, summary, weight, left):
    raw = dict(times=times, keys=key, values=value, summaries=summary, weights=weight)
    if left is not None:
        # A positive write prunes entries expired at the first current time.
        # Preserve all still-visible history, including zero-weight entries.
        if config.window:
            keep = times[0]-left["times"] < config.window
            left = {name: value[keep] if name == "times" else value[:, keep] for name, value in left.items()}
        raw = {name: torch.cat((left[name], current), 0 if name == "times" else 1)
               for name, current in raw.items()}
    age = times[:, None]-raw["times"][None, :]
    mask = age >= 0
    if config.window:
        mask = mask & (age < config.window)
    return raw, age, mask


def kv_terminal(config, raw, last):
    keep = torch.ones_like(raw["times"], dtype=torch.bool) if not config.window else last-raw["times"] < config.window
    return {name: value[keep] if name == "times" else value[:, keep] for name, value in raw.items()}


def detached(states):
    """Explicit truncated-BPTT/checkpoint boundary; int64 fields remain int64."""
    if isinstance(states, dict):
        return {key: detached(value) for key, value in states.items()}
    if isinstance(states, (tuple, list)):
        return type(states)(detached(value) for value in states)
    return states.detach().clone()
