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

## Capacity and region execution, 2026-11-18

- Compatible region batching must retain independent Attention/allocation axes,
  owner sharing and private memory shapes. Measured Settle/Timed training time
  ratios were 0.695–0.792, with slightly higher peaks; activity changes prevent a
  universal same-trajectory speedup claim. See [execution evidence](evidence/capacity-execution-20261118.md).
- Non-reentrant regional recomputation reduced measured peaks to 0.148–0.202 of
  baseline but increased same-shape time to 1.738–1.813. Combined batching and
  recomputation fit B64 in the selected 32 GiB search; its best observed throughput
  was 0.9364 of baseline B16. Saving memory is not by itself a throughput gain.
- A Timed last-tick view can retain a whole window's storage. Clone compact tails
  and release banks after their last consumer without detaching BPTT. State payload,
  autograd saved tensors and allocator peaks overlap; never add separate phase peaks.
- BF16 layout changes can alter Linear decomposition or matrix reduction and
  cross a real allocation boundary. A zero-weight bias probe establishes only
  that case; use actual inputs plus an independent scalar/allocation witness.
  [The bounded investigation](evidence/capacity-rounding-20261118.md) preserves
  both implementation fixes and unresolved finite differences.
- Compare language overhead, outer workers and intra-op threads separately.
  [The standalone regional slice](evidence/native-regional-performance-20261118.md)
  helped small NPU dispatch-heavy shapes; larger shapes did not benefit uniformly
  from C++ workers. It does not justify a second complete model implementation.

## Training choices and attribution, 2026-11-18

- Reception may save large edge activations despite a short forward. Local
  non-reentrant recomputation reduced the selected peaks about22–25% for a
  roughly1–5% time cost; whole-region recomputation saved more but did not win
  same-budget throughput. Saved-site storage aliases other sites and includes
  checkpoint inputs; measure the unique union and allocator peaks separately.
- K=1 SSD vector summaries need not use the matrix-state block scan. Parallel
  affine composition avoids logs/inverse decay and preserves zero decay;
  selected normal update times fell about6–9%. The matrix recurrence still has
  a substantial device Cumprod cost, including observed Ascend AICPU kernels.
  This is a bounded next opportunity, not proof that a new scan will be faster.
- A/B parameter and block matching does not match arithmetic. In the measured
  Settle point, Tide NodeLinear alone performs about123 G forward multiply-adds,
  versus about32 G for Transformer A/C linear layers. Region batching reduces
  dispatch but can increase packed padding. Record both selected and executed
  work before attributing an entire gap to Python or Attention.
- Give the comparison model the same finite optimization opportunity. Increasing
  Transformer query tiles128→256 improved A/B at L256 despite more masked
  arithmetic; contiguous QKV did not yield a clear general gain. Keep noise and
  actual operator identity, and do not call either baseline globally optimal.
- Choose training settings using continuing history and reserved memory as well
  as single-window allocated peaks. Settle FP32 B24 and BF16 B32 reached similar
  twenty-step throughput, while FP32 had a lower allocated/reserved peak.
  Inference, first backbone output and cached decode need separate choices.

These observations are scoped to [search](evidence/training-search-20261118.md),
[costs and real traces](evidence/training-costs-20261118.md),
[continuous training](evidence/training-continuous-20261118.md) and
[other modes](evidence/training-modes-20261118.md). Cross-path activity changes
and retained numerical failures prevent strict equal-trajectory speedup claims.

## Reception and SSD, 2026-11-18

- A short matrix-state scan can spend more time in a device Cumprod than in
  its matrix products. Bounded doubling removed 128 observed Ascend AICPU
  Cumprod tasks, with about 19% higher complete FP32 throughput at the selected
  small point. More multiplication tasks and saved intermediates remain;
  primitive gains and memory savings cannot be assumed to scale together.
- A structural Reception mask can enable native fusion while preserving real
  zero-strength derivatives, yet the measured FP32 padded microcases became
  slower and larger. QKV fusion helped light64 more than the wider profiles.
  Keep per-site shape and whole-model evidence separate.
- Static complete owner order can avoid parameter gather materialization.
  A possible-owner superset is insufficient; repeated/nonconsecutive rows and
  dynamic Full must retain gathers. Cached metadata must survive inference-mode
  initialization before training, without caching weights or activations.
- Near-equal parameters/blocks need not imply near-equal physical work. The
  0.5B point had K packing execute about 5.25 times its selected rows, while
  Transformer A had 1712 narrow sequential blocks. Both packing and launch gaps
  are concrete opportunities; neither model's entire gap is just Attention.
- At the 17B inference point, serial region batch1→4 changed one Full action,
  whereas batch4 serial→per_device retained identical recorded work, packing,
  output support and time hashes. The latter observed about 64% higher decode
  throughput across different physical-card pairs. Separate batching from
  concurrency before attributing an activity change to threading.
- Static views gave about 9.5% higher 9B B1 sustained decode throughput with
  essentially unchanged allocated peak and slightly higher reserved peak.
  Cumulative eliminated index outputs are neither peak memory nor measured
  hardware bandwidth; the candidate also caches static IDs. Source/card scope
  and separate training deviations remain part of that observation.
- Profiler kernel-interval unions and synchronized stage observations describe
  their intrusive runs. Never add nested events, or divide them by ordinary
  timing to manufacture a normal-run utilization percentage.

These results and their negative candidates are scoped in
[Reception/SSD costs](evidence/reception-performance-20261118.md),
[parameter indexing](parameter-indexing.md), [resident inference](evidence/reception-inference-20261118.md)
and the separate [continuous training](evidence/reception-training-20261118.md) record.

## Fixed Transformer depth and inference batch, 2026-11-18

The [supplementary measurements](evidence/transformer-depth-batch-20261118.md) retain
FP32, prompt128/chunk32/decode16 and five normal samples per new point. Fixed total
parameters at 32/48 layers change width, head width and FFN allocation: 9B L32 was
faster than the older five-layer C, while 17B L48 was faster than L32. Nearest parameter
matching is not an optimization over Transformer throughput or a pure depth experiment.

Increasing Tide's input batch to eight improved aggregate throughput to 12.34 positions/s
on 9B/one NPU and 10.07 on 17B/two NPUs, with higher per-step latency and allocated peaks
near the chosen 48 GiB budget. Actual touched parameters per input changed little in the
B2/B4 records, while packing overhead increased; lower activity did not explain the gain.
These short-request shapes do not establish long-history capacity or the optimal batch.

At 17B B4/B8, four-device region placement was slower than two devices, despite roughly
halving per-device peaks. B4 K calls rose from 300 to 480 and S calls from 160 to 320;
static remote predecessor-plan entries rose from 16 to 54. Count these as grouping/transport
work indicators, not a measured communication percentage. B4 had one Full/support change;
B8 work/support/time summaries matched but packing differed. Device count, parameter balance
and a per-device thread policy alone do not establish efficient scaling. Replicas, transfer
paths, communication-aware partitions and CPU scheduling require their own finite comparisons.

## Per-budget decode batch search, 2026-11-18

The [twelve-series follow-up](evidence/decode-capacity-20261118.md) keeps model,
precision, request and per-card allocated/reserved budgets fixed. Same-batch and
independently selected budget-best throughput answer different questions. At 17B,
four-card Tide B34 reached 24.28 positions/s versus
two-card B8 at 9.90, even though four cards were slower
at the previously measured common B8. Capacity can outweigh poorer same-batch
efficiency; this is not a same-work concurrency speedup.

The largest passing batch did not always win. For the 17B four-card 48-layer
Transformer, B128 formally reached 333.577 positions/s versus 324.383 at B155,
with lower allocated/reserved peaks. For the four-card 32-layer model, B197 and
B198 reached 284.166 and 284.197; that 0.011% difference is not a demonstrated
performance advantage. Retain the alternate batch when memory margin matters.

Track both allocator quantities. A reserved-budget refusal is not hardware OOM,
and observed reserved peaks need not increase monotonically with batch: the 17B
two-card 48-layer Transformer fell from 47.314 GiB at B32 to 43.098 GiB at B49,
while allocated grew from 38.205 to 41.988 GiB. Keep physical-device/allocator
history identities and avoid assigning the change to an unprofiled kernel.
An adjacent refused batch is a finite stopping observation, not proof that every
larger batch fails. Larger-batch holes, other prefill chunks and allocator choices
were not exhaustively searched. Final private-state storage is an all-device
snapshot and cannot replace the measured whole-request per-device peaks.
