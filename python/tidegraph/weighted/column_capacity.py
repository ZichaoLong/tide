"""Conservative per-node temporary-workspace admission for column windows.

This is separate from whole-model, retained-autograd and device-memory admission.
Its bound depends on live rows/history and dimensions, never absolute clock time.
"""
from .config import KVConfig


def check(memory, left, domain, limit, element_size):
    c = memory.config
    rows = len(domain.times)
    metadata_bytes = 0
    if isinstance(c, KVConfig):
        queries = max(domain.offsets.diff().tolist(), default=0)
        past = max(left.layout.offsets.diff().tolist(), default=0)
        elements = domain.samples*queries*(past+queries)*(8+3*c.heads)
        if c.read_frequencies or any(c.full_frequencies):
            elements += 8*domain.samples*queries*(past+queries)*(c.width+c.summary_width)
        elements += rows*(4*c.width+c.summary_width)
    else:
        matrices = c.matrix_shape[0]
        padded_rows = domain.samples*max(domain.offsets.diff().tolist(), default=0)
        elements = 8*padded_rows*matrices*(c.key_width*c.key_width+c.key_width*c.value_width)
        elements += padded_rows*(8*c.width+8*c.summary_width)
        if not c.basic:
            bank = len(left.cache_layout.times)+rows
            taps = c.conv_size-1
            # Forward taps plus the ordered transpose's bank×lag cotangents.
            elements += 4*(bank+rows*c.conv_size+bank*taps)*c.conv_width
            metadata_bytes = 16*(rows+bank)*taps  # host/device int64 layouts
    requested = elements*element_size+metadata_bytes
    if requested > limit:
        raise MemoryError(f"column local workspace estimate {requested} exceeds budget {limit}")
    return requested
