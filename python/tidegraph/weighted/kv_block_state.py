"""Call-local provenance for hidden KV prefixes; never checkpoint metadata.

The final adopted state remains on its public independent-root path. A view
records actual prepared writes, not an inferred common storage/parameter alias.
"""
from dataclasses import dataclass


@dataclass(eq=False)
class Bank:
    old: object
    values: tuple
    summaries: tuple
    contents: tuple
    times: tuple
    guards: tuple = ()


@dataclass(eq=False)
class View:
    bank: Bank
    end: int
    slots: dict
    value: object

    def valid(self, state):
        return (state is not None and state.value is self.value and state.slots.keys() == self.slots.keys()
                and all(state.slots[k] is v for k, v in self.slots.items()))


def view(state):
    result = getattr(state, '_kv_block_view', None)
    return result if isinstance(result, View) and result.valid(state) else None


def attach(weights, old, batch, states):
    """Attach only hidden, unexpired prepared prefixes with a future write.

    Terminal states, traces, unrelated kernels, and relocated/detached slots
    retain ordinary query boundaries. This metadata contains no numeric oracle.
    """
    from .kv import KVMemory
    if type(weights.kernel) is not KVMemory or not batch.prepared:
        return
    for owner, (a, b) in enumerate(zip(batch.offsets, batch.offsets[1:])):
        ids = [j for j in range(a, b) if states[j].last_time == batch.times[j]]
        if len(ids) < 2 or any(set(batch.prepared[j]) != {'values', 'summaries'} for j in ids):
            continue
        initial = old[owner]
        stamps = tuple(initial.slots['times'].cpu().tolist()) + tuple(batch.times[j] for j in ids)
        bank = Bank(initial, tuple(batch.prepared[j]['values'] for j in ids),
                    tuple(batch.prepared[j]['summaries'] for j in ids),
                    tuple(batch.views[j].value for j in ids), stamps,
                    (*weights.parameters(), initial.value, *initial.slots.values()))
        count = len(initial.slots['times'])
        terminal = states[ids[-1]]
        positive = set(ids)
        for j in range(a, b):
            if j in positive:
                count += 1
            state = states[j]
            # Do not attach to the public left state or final positive prefix.
            if (state is not initial and state is not terminal and j < ids[-1]
                    and state.slots['times'].numel() == count):
                state._kv_block_view = View(bank, count, dict(state.slots), state.value)
