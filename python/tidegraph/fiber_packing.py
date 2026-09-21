"""Ragged source-row packing, per-sample event masks and post-attention pooling."""
from collections import defaultdict
import math
import torch
from .fiber_attention import advance_bias
from .history import increment
from .records import State
from .fiber_pool import pool_rows


def packed_sequence(w, heads, pool, old, batch):
    batch.validate()
    if len(old) != len(batch.owners):
        raise ValueError("packed initial-state count mismatch")
    rows, offsets, source_slots = [], [0], []
    for view in batch.views:
        if not view.sources:
            raise ValueError("fiber attention requires complete source rows")
        sources = sorted(view.sources, key=lambda s: s.slot)
        rows.extend(s.atom.value*s.scale for s in sources); source_slots.extend(s.slot for s in sources)
        offsets.append(len(rows))
    x = torch.stack(rows); width = len(w.bias); d = width//heads
    qkv = torch.nn.functional.linear(x, w.extra["fiber_qkv"].t(), w.extra["fiber_qkv_bias"])
    q, k, v = [part.reshape(len(x), heads, d) for part in qkv.split(width, -1)]
    q = q*(1/math.sqrt(d))
    groups = defaultdict(list)
    for i, state in enumerate(old):
        a, b = batch.offsets[i:i+2]
        groups[len(state.slots["key"]), offsets[b]-offsets[a]].append(i)
    states = [None]*len(batch.times); pooled = [None]*len(batch.times)
    for (cache, queries), ids in groups.items():
        qs, ks, vs, bs, ms = [], [], [], [], []
        key_index = torch.arange(cache+queries, device=x.device)
        for i in ids:
            a, b = batch.offsets[i:i+2]; begin, end = offsets[a], offsets[b]
            qs.append(q[begin:end]); ks.append(torch.cat((old[i].slots["key"], k[begin:end])))
            vs.append(torch.cat((old[i].slots["value"], v[begin:end])))
            bias = old[i].slots["log_bias"]; last = old[i].last_time
            bias_rows, masks = [], []
            for j in range(a, b):
                time = batch.times[j]; size = offsets[j+1]-offsets[j]
                if last < -1 or time <= last:
                    raise ValueError("fiber attention requires increasing tick times")
                bias = torch.cat((advance_bias(bias, w.extra["fiber_decay"], time-last), x.new_zeros(size)))
                bias_rows.append(torch.cat((bias, x.new_zeros(cache+queries-len(bias)))).expand(size, -1))
                masks.append((key_index < len(bias)).expand(size, -1)); last = time
                states[j] = State(None, time, increment(old[i].observations, j-a+1), {"log_bias": bias})
            bs.append(torch.cat(bias_rows)); ms.append(torch.cat(masks))
        keys, values = torch.stack(ks), torch.stack(vs)
        scores = torch.stack(qs).transpose(1, 2) @ keys.permute(0, 2, 3, 1)
        scores = scores+torch.stack(bs).unsqueeze(1)
        probabilities = scores.masked_fill(~torch.stack(ms).unsqueeze(1), -torch.inf).softmax(-1)
        outputs = (probabilities @ values.transpose(1, 2)).transpose(1, 2).reshape(len(ids), queries, width)
        for row, i in enumerate(ids):
            a, b = batch.offsets[i:i+2]; start = offsets[a]
            for j in range(a, b):
                begin, end = offsets[j]-start, offsets[j+1]-start
                pooled[j] = pool_rows(w, pool, source_slots[offsets[j]:offsets[j+1]], outputs[row, begin:end])
                slots = {"key": keys[row, :cache+end], "value": values[row, :cache+end]}
                if j == b-1:
                    slots = {name: t.clone() for name, t in slots.items()}
                states[j].slots.update(slots)
    y = torch.nn.functional.linear(torch.stack(pooled), w.extra["fiber_out"].t(), w.extra["fiber_out_bias"])
    for j, state in enumerate(states):
        state.value = y[j].clone()
    return states, len(groups)
