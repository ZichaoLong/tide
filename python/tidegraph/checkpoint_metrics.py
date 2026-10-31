"""Optional wall-time spans outside execution timers; no tensor observation."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_current = ContextVar("tide_checkpoint_metrics", default=None)


@contextmanager
def capture():
    record = dict(schema="tide-checkpoint-cost-v1", spans={},
                  scope="inclusive wall-time spans; nested spans must not be summed; no extra device synchronization")
    token = _current.set(record)
    began = time.perf_counter()
    try:
        yield record
    finally:
        record["total_seconds"] = time.perf_counter()-began
        _current.reset(token)


@contextmanager
def phase(name):
    record = _current.get()
    if record is None:
        yield
        return
    began = time.perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        row = record["spans"].setdefault(name, dict(seconds=0.0, calls=0, failed_calls=0))
        row["seconds"] += time.perf_counter()-began
        row["calls"] += 1
        row["failed_calls"] += int(failed)
