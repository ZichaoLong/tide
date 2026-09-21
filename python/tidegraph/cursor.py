"""Owned native streaming state. Advancing does not serialize the continuation."""
from .native_records import from_continuation, to_continuation, window_records
from .records import AdvanceResult


class NativeCursor:
    def __init__(self, native, continuation):
        self.native = native  # Keep model/program/pool alive.
        q = to_continuation(native.core, native.graph, native.compiled, continuation)
        self.cursor = native.core.StreamingCursor(native.engine, q)

    def advance(self, external, stop, *, sealed_until):
        xs = [self.native.core.External(x.batch, x.port, x.position, x.time, x.value) for x in external]
        r = self.cursor.advance(xs, stop, sealed_until)
        return AdvanceResult(r.cut, *window_records(r))

    def snapshot(self):
        return from_continuation(self.native.graph, self.cursor.snapshot())

    def detach(self):
        self.cursor.detach()

    @property
    def cut(self):
        return self.cursor.cut

    @property
    def failed(self):
        return self.cursor.failed
