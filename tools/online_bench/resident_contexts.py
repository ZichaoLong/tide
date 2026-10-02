"""Bound all saved sample continuations before their device payload allocation."""
import torch


class ContextPool:
    def __init__(self, session, count, devices, capacity, compact):
        self.session, self.compact = session, compact
        self.handles = [None]*count if count > 1 else []
        self.budgets = {torch.device(d).index: c['components']['saved_contexts']
                        for d,c in zip(devices,capacity['devices'])}
        self.used = dict.fromkeys(self.budgets,0)
        self.peak = dict(self.used)
        if self.handles:
            initial = self.snapshot()
            # Shared empty handles are charged per logical slice. This is a
            # conservative bound; they become distinct after execution.
            for i in range(count):
                self.store(i,initial)

    def __bool__(self):
        return bool(self.handles)

    def __getitem__(self, index):
        return self.handles[index]

    def snapshot(self):
        remaining = {d: limit-self.used[d] for d,limit in self.budgets.items()}
        if sum(remaining.values()) < 1:
            raise ValueError('saved continuation pool budget exhausted')
        return self.session.snapshot_device(max_bytes=sum(remaining.values()),
                                           compact=self.compact,device_budgets=remaining)

    def release(self, index):
        value = self.handles[index]
        for d,n in value.device_bytes.items():
            self.used[d] -= n
        self.handles[index] = None

    def store(self, index, value):
        if self.handles[index] is not None:
            raise RuntimeError('release saved continuation before replacing it')
        sizes = value.device_bytes
        if any(self.used[d]+n > self.budgets[d] for d,n in sizes.items()):
            raise ValueError('saved continuation pool budget exceeded')
        self.handles[index] = value
        for d,n in sizes.items():
            self.used[d] += n
            self.peak[d] = max(self.peak[d],self.used[d])

    def save(self, index):
        self.store(index,self.snapshot())

    def clear(self):
        self.handles.clear()
        self.used = dict.fromkeys(self.budgets,0)

    def record(self):
        return dict(policy='compact' if self.compact else 'dense',
                    devices=[dict(device_index=d,budget_bytes=limit,peak_saved_bytes=self.peak[d])
                             for d,limit in sorted(self.budgets.items())])
