"""Conservative per-node temporary-workspace admission for column windows.

This is separate from whole-model, retained-autograd and device-memory admission.
Its bound depends on live rows/history and dimensions, never absolute clock time.
"""
from .config import KVConfig


def check(memory, left, domain, limit, element_size):
    c = memory.config
    rows = len(domain.times)
    if isinstance(c, KVConfig):
        queries = max(domain.offsets.diff().tolist(), default=0)
        past = max(left.layout.offsets.diff().tolist(), default=0)
        elements = domain.samples*queries*(past+queries)*(8+3*c.heads)
        elements += rows*(4*c.width+c.summary_width)
    else:
        matrices = c.matrix_shape[0]
        elements = 8*rows*matrices*(c.key_width*c.key_width+c.key_width*c.value_width)
        elements += rows*(8*c.width+8*c.summary_width)
    requested = elements*element_size
    if requested > limit:
        raise MemoryError(f"column local workspace estimate {requested} exceeds budget {limit}")
    return requested
