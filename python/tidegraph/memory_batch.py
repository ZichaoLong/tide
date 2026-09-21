"""Joint [time,batch,width] scans over equal-length, nonempty event segments."""
from collections import defaultdict
import torch
from .history import increment
from .records import State
from .scan import affine_scan


def diagonal_batch(weights, old, batch, ssm=None):
    batch.validate()
    if len(old) != len(batch.owners):
        raise ValueError("packed initial-state count mismatch")
    groups = defaultdict(list)
    for i in range(len(old)):
        groups[batch.offsets[i+1] - batch.offsets[i]].append(i)
    states = [None] * len(batch.contents)
    for length, ids in groups.items():
        h = torch.stack([batch.contents[batch.offsets[i]:batch.offsets[i+1]] for i in ids], dim=1)
        if ssm is None:
            initial = torch.stack([old[i].value for i in ids])
            values = affine_scan(weights.decay.sigmoid().expand_as(h), h, initial)
        else:
            initial = torch.stack([old[i].slots["memory"] for i in ids])
            a, b = ssm.coefficients(weights, h)
            memory = affine_scan(a, b, initial)
            values = (h @ weights.extra["ssm_c"]) * memory + weights.extra["ssm_skip"] * h
        for row, i in enumerate(ids):
            for t in range(length):
                value = values[t, row]; slots = {} if ssm is None else {"memory": memory[t, row]}
                if t == length - 1:
                    value = value.clone(); slots = {k: v.clone() for k, v in slots.items()}
                j = batch.offsets[i] + t
                states[j] = State(value, batch.times[j], increment(old[i].observations, t + 1), slots)
    return states, len(groups)
