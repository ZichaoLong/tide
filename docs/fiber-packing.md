# Packed same-fiber attention

The scalar profile and independent analytic anchors remain in `fiber-attention.md`.
Python `fiber_packing.py` implements frontier batch/sequence work; native
`fiber_packing.cpp` serves streaming independent-batch and frontier batch/sequence
calls. `FiberAttention.step` remains the scalar schedule/reference path.

## Data and visibility

Existing `PackedSequence` has event contents, sample/owner event offsets, logical
times and complete source views. A second offset vector maps each event to its
real source rows, sorted by local slot. Source tags, physical IDs and input
positions remain in those views; no padding source or synthetic candidate exists.
Raw scaled sources are flattened for one joint Q/K/V projection.

Group samples by `(initial_cache_rows, total_source_rows)`. The score shape is
`[batch, heads, source_rows, initial_cache_rows + source_rows]`. Different samples
never share a score matrix. Equal total row counts may still have different fiber
boundaries, event counts and times; each sample has its own event mask and bias.

Every query sees prior events and every row in its own event. Later events are
masked, irrespective of row position inside a fiber. Ordered repeated bias
subtraction is still performed along each sample's clock; biases are padded only
for score construction and masking. Pool each event's query outputs, then jointly
project event outputs with one output bias per event. Slice prefix K/V into the
corresponding event state; clone final persistent K/V and values to release the
larger group storage in inference. Log-bias snapshots already own compact storage.

## Training and capability boundaries

Schedulers compute packed values under no-grad and replay scalar steps to preserve
the declared first-order public-root connectivity (`packed-autograd.md`). A first
output must not acquire a new path to decay, absent sources, future events or other
samples merely because they participated in a packed projection. Inference does
not replay. Raw low-level packed kernel methods are internal numeric producers;
the scheduler's replay boundary is part of the public training contract.

Native independent-batch and both languages' exact sequence capabilities now
advertise joint work. Clear, selected-only adoption and incompatible Next continue
to prevent sequence prefill. Disabling prefill keeps causal execution available.
`state_sequence_calls`, `max_state_batch`, `max_state_sequence`,
`attention_score_elements` and scalar fallback/replay counters expose work.
Sequence length counts events; score elements count actual source rows.

This is exact-shape bucketing, not a fused ragged SDPA kernel. Tick-repeat bias
preparation still has serial clock loops and long idle-gap cost. Training replay,
score-matrix allocation and repeated cache concatenation remain measured-performance
work, not hidden speed claims. The bounded original oracle now includes CROSSBATCH
via its actual batch cache and eager tick clock, in addition to the five per-sample
modes; consult immutable evidence for completed qualifications.
