"""Optional profile counters using already available host metadata only.

Counts describe actual eager local calls, including private forward replay in
training. They are neither unique logical work nor device timing. Observation
does not read tensors or enter a model, continuation or checkpoint.
"""
from contextlib import contextmanager
from contextvars import ContextVar

MAX_KEYS = 4096
_current = ContextVar("weighted_packing_observation", default=None)


def enabled():
    return _current.get() is not None


class Observation:
    def __init__(self):
        self.counts = {}
        self.dropped_updates = 0

    def add(self, key, value):
        if key in self.counts or len(self.counts) < MAX_KEYS:
            self.counts[key] = self.counts.get(key, 0) + value
        else:
            self.dropped_updates += 1

    def metrics(self):
        return {**self.counts, "packing/schema_version": 1,
                "packing/dropped_updates": self.dropped_updates}


@contextmanager
def observe():
    """Profile one Python executor; nested scopes remain independent."""
    observation = Observation()
    token = _current.set(observation)
    try:
        yield observation
    finally:
        _current.reset(token)


def bucket(kind, length, rows):
    current = _current.get()
    if current is None:
        return
    if type(length) is not int or type(rows) is not int or length < 0 or rows < 1:
        raise ValueError("invalid packing bucket metadata")
    base = f"packing/bucket/{kind}"
    for suffix, value in (("calls", 1), ("rows", rows), ("elements", length*rows),
                          (f"length/{length}/rows/{rows}/calls", 1)):
        current.add(f"{base}/{suffix}", value)


def projection(kind, rows):
    current = _current.get()
    if current is None:
        return
    if type(rows) is not int or rows < 0:
        raise ValueError("invalid projection batch metadata")
    base = f"packing/projection/{kind}"
    for suffix, value in (("calls", 1), ("rows", rows), (f"rows/{rows}/calls", 1)):
        current.add(f"{base}/{suffix}", value)


def skipped(reason, rows):
    current = _current.get()
    if current is None:
        return
    if type(rows) is not int or rows < 0:
        raise ValueError("invalid skipped packing metadata")
    if not rows:
        return
    base = f"packing/skipped/{reason}"
    current.add(f"{base}/calls", 1)
    current.add(f"{base}/rows", rows)
