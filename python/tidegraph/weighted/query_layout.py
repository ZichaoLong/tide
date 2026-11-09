"""KV §2–4: bounded batches of actual snapshots, with int64 visibility.

Lengths describe storage, not received-event time. Padding has no message or
gradient identity; the custom query boundary still owns independent field roots.
The byte target limits multirow packing. A single snapshot can exceed it;
it is neither a peak-memory guarantee nor a cumulative history limit.
"""
from collections import defaultdict
import torch
from . import packing_observation as observation

WORKSPACE_BYTES = 32 << 20


def chunks(states, width, *, budget=WORKSPACE_BYTES):
    groups = defaultdict(list)
    for i, state in enumerate(states):
        length = state.slots["weights"].numel()
        groups[length.bit_length()].append(i)
    for ids in groups.values():
        length = max(1, max(states[i].slots["weights"].numel() for i in ids))
        # Payload/gather/adjoint headroom; never duplicate model matrices.
        row_bytes = length*(6*width*states[ids[0]].value.element_size()+64)
        count = max(1, budget//max(1, row_bytes))
        for start in range(0, len(ids), count):
            yield ids[start:start+count]


class Layout:
    def __init__(self, states, times, window):
        self.states = states
        lengths = [s.slots["weights"].numel() for s in states]
        offsets, total = [], 0
        for length in lengths:
            offsets.append(total)
            total += length
        ref = states[0].slots["times"]
        n, length = len(states), max(1, max(lengths))
        col = torch.arange(length, dtype=torch.int64, device=ref.device)[None, :]
        present = col < ref.new_tensor(lengths)[:, None]
        self.indices = torch.where(present, ref.new_tensor(offsets)[:, None]+col, total)
        self.shape = (n, length)
        stamps = self.slot("times")
        age = ref.new_tensor(times)[:, None]-stamps
        self.visible = present & (age >= 0)
        if window:
            self.visible = self.visible & (age < window)
        self.age = age.clamp_min(0)
        observation.bucket("kv_snapshot_layout", length, n)

    def slot(self, name):
        rows = [s.slots[name] for s in self.states]
        tail = rows[0].shape[1:]
        joined = torch.cat((*rows, rows[0].new_zeros((1, *tail))))
        return joined.index_select(0, self.indices.flatten()).reshape(*self.shape, *tail)

    def weights(self):
        raw = self.slot("weights")
        return torch.where(self.visible, raw, torch.zeros_like(raw))

    def visible_slot(self, name):
        raw = self.slot(name)
        mask = self.visible.reshape(*self.shape, *([1]*(raw.ndim-2)))
        return torch.where(mask, raw, torch.zeros_like(raw))
