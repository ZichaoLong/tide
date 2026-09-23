# Optional native Streaming transport

Implementation candidate; qualification and performance are pending. Both
`packed_sources` and `batch_next` default to false and require packed native
Streaming. They are independent of exact/single attention and earlier fiber
options, and are outside graph/checkpoint identity. Frontier and independent
specializations reject these currently unsupported policies explicitly.

## Source rows

Built-in Aggregate programs can stack all present payloads/scales once into
`SourceBatch.values [present rows,D]`, with event offsets and local source slots.
Pattern groups gather these rows for the unchanged canonical Aggregate fold.
The result retains one `[events,D]` content matrix for state batch input. Source
metadata, optional contributions and public per-event values remain available.
Fiber attention reuses the scaled rows and only permutes if slot order differs
from Aggregate's canonical atom order. It never substitutes the summary as a
query, drops present zero sources, or changes the active/all-source denominator.

The batch storage is immutable and numeric. Scalar state/Read/Full replay still
reads original atoms, source scales and bound contributions. No private cache
is serialized. A custom Aggregate defaults to the existing batch/scalar path;
`packed_source_fallback_events` records that choice. Custom source-batch programs
must promise exactly `atom.value * scale` in the specified row order; metadata
is validated before exposure. Custom consumers can ignore the transport hint.

This does not eliminate all event objects or batch persistent KV caches. Public
trace and state boundaries still materialize views/records. No dense allocation
over all nodes or dormant sample/node pairs is introduced.

## Next

`adopt-v1` supports batch adoption. Selected clear uses the state's `reset_batch`
capability: fiber attention batches value-zeroing and shares immutable empty
cache tensors, preserving each owner's clocks. Other state programs use their
scalar reset. Control-blend/custom Next default to scalar execution, counted by
`next_scalar_fallback_steps`. Comparison remains unchanged for current Full.

Grad-enabled batch Next binds every value/slot to its independent scalar
transition using the existing semantic replay. This preserves isolated roots,
None versus connected-zero, sharing and the declared first-order VJP; it is not
an optimized-backward claim. `semantic_next_replays` exposes that cost. Snapshot
export continues to clone storage; later execution cannot mutate retained state.

`next_steps` is a logical event count. `next_batches`, `next_reset_batches` and
scalar fallback/reset counters describe execution. New operator counters
`fiber_scale_elements` and `fiber_reused_elements` distinguish performed source
multiplications from reused values. Equal semantics do not require identical
physical operation counts; performance reports must disclose their changes.

The scale binary and portable `run_pdg.py` expose `--packed-sources 0|1` and
`--batch-next 0|1`; run records include both. Existing defaults remain the
reference for controlled comparisons.
