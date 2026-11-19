# Applicable performance experience

Use these mechanisms where their preconditions hold. They apply to scheduling,
local math, state, autograd, checks and publication. They are tools for choosing work,
not a required sequence or a permanent obligation to reproduce old implementations.
The current [instance observations](performance-lessons-instances.md) and their
[fixed report](evidence/instances-fp32-20261118.md) supply current numbers.

## Tensor granularity and ownership

- Preserve B/L and compatible-node batches across producer/consumer boundaries.
  A contiguous buffer alone does not create a GEMM batch. Inspect loops, gathers,
  copies, actual matrix shapes and selected-versus-padded work.
- Shared bases can use flattened GEMM; independent/private adapters may use BMM.
  Share computation only for the same parameters and inputs. Sum every use's
  gradient while keeping node state private; a shared weight reduces capacity,
  not necessarily per-token arithmetic.
- Pack selected queries once, borrow a uniquely owned KV bank, and avoid copying
  whole history per query. Only pad the operation requiring rectangular shape.
  A direct baseline itself may still contain B/L loops and repeated history work.
- Indexing/sparse operators and custom Autograd can reduce graph construction.
  Count real computation and inner graphs as well as the visible autograd graph;
  fewer outer nodes or submissions do not alone imply faster training.
- Borrowed/cache storage needs clear ownership, version and lifetime. Do not
  mutate tensors required by backward. Check repeated owners, empty rows, ragged
  lengths, final state and actual restored updates.

## Mathematics, precision and hardware

Treat KV, reception and regional Attention separately. Native fusion, ordinary
batched operations and tiled/recomputed backward compete by shape, device, dtype
and weighted-prior derivatives; a fast API name does not prove a fused kernel.

Near-null gradients can amplify in AdamW. Check updates and optimizer state as well
as local gradient error. Stable algebra/reduction order can help without changing
epsilon; promoting only the final reduction cannot recover prior rounded inputs.
An FP64/summed independent witness may show the low-precision reference is itself
inaccurate. Preserve original deviations and the scope of any explanation.

Logical support is separate from floating underflow. Positive-weight normalization,
attention and allocation derivatives must be considered together. A small local
stabilization does not establish stability of all compositions or scales.

CPU threads, acceleration and shared memory have shape-dependent costs. Count
synchronization, host/device movement, padding, forward/backward, optimizer and
allocator peaks. Workspace size and captured device-control task counts are
separate resource limits. Communication/weight sharing does not by itself fix
asynchronous event ownership or deterministic gradient accumulation.

## Additive and subtractive investigation

A compact same-function computation estimates an attainable point; it is not proof
of optimality. Add needed state, sharing and interface responsibilities deliberately,
and measure meaningful increments. Combine this with static code inspection and
stage/operator diagnostics. Distinguish changed mathematics from extra work for
the same function. Transformer/SSD provide different-function compute references.

Use local evidence to adopt a well-scoped optimization, then representative complete
calls to assess integration. Stop a finite investigation with an adopted change,
a rejected candidate or a concrete remaining opportunity. Avoid both compulsory
second implementations and indefinite local improvement without a useful conclusion.

## Historical observations retained as summaries

The following source identifiers are provenance, not prompts to retrieve retired code.
Original pages remain in the pre-governance commit `728d19a470d16d392d5dd6a2452161d1a3e9959e`.

| Observation | Scope and lesson | Original report path |
|---|---|---|
| Core-3 batched Full VJP reduced grad-forward 110.439→10.499 ms/sample-token | 9.468B Add, CPU56; grad-forward is not full training | docs/evidence/full-batched-autograd.md |
| Aggregate batching improved small complete Add 1.442→1.048 s and Attention 5.477→4.985 s | Local batching can help training, but the different large-mode results cannot be combined | docs/evidence/aggregate-batched-autograd.md |
| Direct FC removed its B loop: CPU training 1.365→1.044 s, NPU .674→.291 s | Old FC16 B8/L128; inspect the baseline's own granularity. SDPA alone added no gain | docs/evidence/fc-batch-full-20261112.md |
| LH custom batching/indexing used packed history and grouped backward | Its activation clock/selector differ from current Tide; transfer the storage/graph lesson, not its semantics | docs/performance-lessons-batching.md, fixed LH a10fdb1883fccd63ec21e36cc9cffa294c63c9e2 |
| Widening only an outer reduction did not resolve the whole FP32 discrepancy | Follow the arithmetic chain; higher precision is a diagnostic, not a universal fix | docs/evidence/core5-nongpu-boundaries-20261029.md |
| Shared-use gradient arrival order affected repeated updates | Explicitly aggregating known users stabilized that witness, not every asynchronous backend | docs/evidence/column-placement-20261111.md |

Record new experience with observation, mechanism, applicability, positive/negative
outcome and source. Measured gains certify only their configuration. Old gates, APIs,
standalone/resident matrices and full-record duties do not accompany these lessons.
