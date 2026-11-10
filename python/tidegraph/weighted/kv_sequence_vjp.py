"""KV §2: bulk VJP when intermediate State handles are not publicly returned.

All output messages, boundary states and input/parameter roots remain public.
Tracing retains the step chain so later states can be differentiated with
respect to an intermediate state handle. This boundary never replaces tracing.
"""
from dataclasses import replace
import torch

FIELDS = ("keys", "values", "summaries", "weights")
PACK_BYTES = 16*1024**2


def reduce_prefixes(bars, spans, size, reference):
    """Deterministic gather/reduce; no colliding scatter/atomic accumulation."""
    used = [(bar, span) for bar, span in zip(bars, spans) if bar is not None]
    if not used:
        return None, -1
    values = torch.cat([bar for bar, _ in used] + [reference.new_zeros((1, *reference.shape[1:]))])
    zero = len(values)-1
    offsets, offset = [], 0
    for bar, (start, end) in used:
        offsets.append((offset, start, end)); offset += len(bar)
    width = max(1, reference[0:1].numel())
    chunk = max(1, PACK_BYTES//(len(used)*(8+width*reference.element_size())))
    pieces = []
    for begin in range(0, size, chunk):
        end = min(size, begin+chunk)
        index = [[offset+i-start if start <= i < stop else zero
                  for offset, start, stop in offsets] for i in range(begin, end)]
        ids = torch.tensor(index, device=values.device, dtype=torch.int64)
        gathered = values.index_select(0, ids.flatten()).reshape(end-begin, len(used), *values.shape[1:])
        pieces.append(gathered.sum(1))
    return torch.cat(pieces) if pieces else reference[:0], max(end for _, (_, end) in used)


class _Sequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, *inputs):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*inputs)
        numeric, owners = plan
        ctx.owners = owners
        outputs, differentiable = [], []
        # Concatenate once per field, then return disjoint views of actual data.
        for field, name in enumerate(FIELDS):
            rows, flags = [], []
            for old_ids, writes, spans, events in owners:
                connected = inputs[old_ids[field]].requires_grad
                for (_, value_id, summary_id, content_id), event in zip(writes, events):
                    source = summary_id if field == 2 else content_id if field == 3 else value_id
                    connected |= inputs[source].requires_grad
                    rows.append(numeric[event].slots[name]); flags.append(connected)
            if rows:
                joined = torch.cat(rows)
                outputs.extend(joined.split([len(row) for row in rows])); differentiable.extend(flags)
        ctx.mark_non_differentiable(*(x for x, flag in zip(outputs, differentiable) if not flag))
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("KV sequence supports first-order VJP only")
        inputs = ctx.saved_tensors  # Validate original parameter/state/content versions.
        owners = ctx.owners
        total = sum(len(events) for *_, events in owners)
        result, offset = [None]*len(inputs), 0
        for old_ids, writes, spans, events in owners:
            n = len(inputs[old_ids[0]])
            reduced, last = [], []
            for field, old_id in enumerate(old_ids):
                rowbars = bars[field*total+offset:field*total+offset+len(events)]
                value, end = reduce_prefixes(rowbars, spans, n+len(writes), inputs[old_id])
                reduced.append(value); last.append(end)
                if value is not None and inputs[old_id].requires_grad:
                    result[old_id] = value[:n]
            for i, (_, value_id, summary_id, content_id) in enumerate(writes):
                index = n+i
                if inputs[value_id].requires_grad and any(index < last[f] for f in (0, 1)):
                    result[value_id] = torch.cat([reduced[f][index] if index < last[f]
                        else inputs[value_id].new_zeros(inputs[value_id].numel()//2) for f in (0, 1)])
                if inputs[summary_id].requires_grad and index < last[2]:
                    result[summary_id] = reduced[2][index]
            active = [(i, row[3]) for i, row in enumerate(writes)
                      if n+i < last[3] and inputs[row[3]].requires_grad]
            if active:
                contents = torch.stack([inputs[ident] for _, ident in active])
                mass = torch.stack([reduced[3][n+i] for i, _ in active])*(contents[:, 0] <= 1)
                gradients = torch.cat((mass[:, None], torch.zeros_like(contents[:, 1:])), 1)
                for (_, ident), grad in zip(active, gradients.unbind()):
                    result[ident] = grad
            offset += len(events)
        return None, *result


def bind(weights, old, batch, numeric):
    """Decline unprepared/custom paths; callers retain the generic step VJP."""
    from .kv import KVMemory
    if type(weights.kernel) is not KVMemory or not batch.prepared:
        return None
    positives = [[j for j in range(a, b) if numeric[j].last_time == batch.times[j]]
                 for a, b in zip(batch.offsets, batch.offsets[1:])]
    if any(set(batch.prepared[j]) != {"values", "summaries"} for ids in positives for j in ids):
        return None
    # Preserve the former boundary's version checks, including unused parameters.
    from ..state_vjp import _weights, _rows
    inputs = []
    def capture(t):
        inputs.append(t); return t
    _weights(weights, capture)
    for i, state in enumerate(old):
        _rows([(state, batch.views[j], batch.times[j], batch.prepared[j]) for j in
               range(batch.offsets[i], batch.offsets[i+1])], capture)
    def add(t):
        inputs.append(t); return len(inputs)-1
    owners = []
    for state, ids in zip(old, positives):
        old_ids = [add(state.slots[name]) for name in FIELDS]
        writes, spans = [], []
        n = len(state.slots["times"])
        for k, j in enumerate(ids):
            p = batch.prepared[j]
            writes.append((j, add(p["values"]), add(p["summaries"]), add(batch.views[j].value)))
            end = n+k+1
            spans.append((end-len(numeric[j].slots["times"]), end))
        owners.append((old_ids, writes, spans, ids))
    if not any(positives):
        return [old[i] for i in range(len(old)) for _ in range(batch.offsets[i], batch.offsets[i+1])]
    outputs = iter(_Sequence.apply((numeric, owners), *inputs))
    updated = {j: dict(numeric[j].slots) for ids in positives for j in ids}
    for name in FIELDS:
        for ids in positives:
            for j in ids:
                updated[j][name] = next(outputs)
    previous = []
    for i, state in enumerate(old):
        for j in range(batch.offsets[i], batch.offsets[i+1]):
            previous.append(state)
            if j in updated:
                state = replace(numeric[j], value=state.value, slots=updated[j])
            numeric[j] = state
    return previous
