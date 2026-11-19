# Interfaces, state and autograd

Use [architecture](../../../../docs/architecture.md) for current symbols and
[semantics](../../../../docs/semantics.md) for the selected mathematics. These
are design techniques for the active instances, not a second public API.

Follow B, sequence/logical time, node ownership and ragged offsets through
reception, preparation, regional decisions, selected Full, state and publication.
Keep these axes in tensors across boundaries. Chunk when dependencies, memory
or operator efficiency require it; an outer window alone does not make inner
per-event work batched. Pack once around repeated projections where useful.

Consider actual work as well as contiguous storage. Rectangular packing can
process many more rows than are selected. Bucketing, grouped operations or
segmented indexes may reduce this cost, but compare their indexing and launch
costs on useful shapes. Separate parameter groups when it preserves efficient
matrix operations; sharing does not require one universal tensor format.

Give persistent and differentiable state explicit ownership. A common KV bank
with indexed/range queries avoids materializing every prefix. Preserve original
int64 times, window visibility, zero-reception writes and the selected Read/Full
snapshots. Reuse a projection only for the same parameters and inputs. Shared
weights do not make node state shared.

Distinguish persistent state, read-only inputs, scratch and checkpoint data.
Scratch reuse waits for all readers, including device work and backward.
Checkpoint decoding restores the actual saved state; a reduced observation
cannot stand in for exact state recovery.

Design backward together with forward. Cached values detached from autograd
need an explicit, correct derivative path. Batch gather/scatter transposes and
sum every shared use. A per-row view can create repeated full-buffer gradient
allocations even when forward storage is packed. Block or segmented adjoints
can help, with accumulation order and connectivity checked explicitly.

A custom Function is useful when it removes measured structural overhead or
controls saved intermediates. Re-enabling autograd for many private leaves and
calling nested grad may merely hide that overhead. Ordinary autograd is often
appropriate for a small number of large GEMM/Attention operations. Attention
backward can recompute tiles instead of retaining the full score matrix while
preserving weighted-prior derivatives and normalization across chunks.

Expose the output and state the current consumer needs. Layout and API changes
may be appropriate when they enable efficient computation; document migration
and verify the new observable behavior. Instance/backend implementations can
be separate when their useful layouts differ.

Select checks for the changed boundary: repeated/shared inputs, absent versus
zero gradients, short/long histories, large logical times, window cuts, optimizer
state and restored execution. Relevant mechanism checks justify a local change;
whole-path observations establish its integration cost. Numerical policies are
owned by [precision](../../../../docs/precision.md), including FP32/BF16 and
bounded CPU FP64 diagnosis.
