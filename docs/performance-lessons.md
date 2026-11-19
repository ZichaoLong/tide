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

A constrained solver can satisfy total mass on the wrong active face. The
[shared9B witness](evidence/cleanup-numerics-20261119.md) lost a representable positive
control after breakpoint-prefix cancellation; the lost score VJP was substantial.
Keep the actual batch shape and stride in a replay, check optimality bounds as well
as mass, and distinguish a solver error on identical rounded inputs from upstream
scores crossing a true boundary. Added checks/fallbacks need new whole-path timing;
a corrected local formula does not inherit old throughput.

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

## Memory width and ownership, 2026-11-19

The [five-family evidence](evidence/memory-candidates-20261119.md) holds graph,
content, summary and FFN dimensions fixed before a separate near-parameter comparison.
Narrowing total Q/K/V width changes the memory function and parameter count; it
is neither GQA nor an equivalent execution optimization. Actual head counts must
drive capacity, attention work and fused-query accounting.

With 848 nodes, d768 and FFN3072, independent Full FFNs alone contain 6.002B
parameters. Memory-only regional sharing lowers total parameters to 6.936B but
retains those FFNs, private histories and their activations. The measured medium
shared case saves weights/AdamW storage without improving continuous training
throughput; parameter savings alone do not establish a useful training speedup.

Shared storage needs a compatible execution grouping to become a larger GEMM.
The measured region-batch4 shared case executes 120 base BMMs and no base MM
across all NodeLinear modules in its zero-state training work audit. Shared
memory matrices account for 48 of those BMMs and 752 repeated base rows. Per-head query
adapters also remain distinct after projection fusion. Inspect real operands,
packing and dispatch before attributing a gain to weight reuse.

At 9B B8, reducing the shared region group from four to one eliminates repeated
base rows but increases calls and lowers measured decode throughput from 12.288
to 7.948 positions/s. It also changes output support despite matching parameter
and input hashes; the cause is unlocated. Less padding and a true base MM do not
by themselves establish whole-model speed or numerical equivalence.

The real-history X/KV128 decode diagnostics both perform 128 KV compactions and
640 field gathers. Narrowing lowers their payload, while summary width and int64
window handling stay fixed. State joining materializes another substantial set
of banks. Repeated logical output bytes characterize this work; they are not
measured device traffic, peak memory or a bandwidth estimate.

All-SSD improves the measured large cached decode yet is slower and larger in
the medium zero-state whole-window training point. SSD preparation includes its
convolution, state scan and descriptor reads before Full selection. Historical
KV handling, query padding and workload boundaries can reverse the ranking;
zero-state timing and continued chunked training are separate observations.


## Narrow KV paired cost diagnosis, 2026-11-19

The [paired diagnosis](evidence/kv128-transformer-breakdown-20261119.md) separates
ordinary timing, synchronized modules and synchronized dispatches. Fixed-FFN KV128
still holds 6.002B independent FFN parameters. In the large B8 decode operator pass,
FFN weight indexing takes much more observed time than its BMMs; narrowing memory
does not remove those indexed weight copies. Actual publish indices and FFN operands
show about 5x executed/selected KV rows in that pass, with separate prefill/training ratios.
Storage sharing, mathematical sparsity and actual rectangular work remain different facts.

The medium continuing-training gap is mostly forward and backward, with a small
AdamW difference. Its backward includes many index writes, gather backwards,
tensor creations and pointwise operations; matrices are not the only cost.
Unscoped autograd operators cannot be assigned to forward modules from shape alone.
Entry synchronization must finish before a child timer starts; parent remainders
still include unwrapped work and waiting. These diagnostics perturb scheduling,
so neither their shares nor their logical output bytes identify ordinary wall-time
shares, bus traffic or recoverable speedup. Optimization requires a new full-path
measurement and unchanged numerical/continuation checks at the intended scale.

## FFN interventions and CPU/NPU comparison, 2026-11-19

The [finite execution comparison](evidence/ffn-execution-devices-20261119.md)
keeps declared FFN formulas and ownership while separating direct bank access W,
owner-row packing P and their combination. Three paired medium NPU training
repeats gave throughput ratios 0.991/0.854/0.813, despite lower allocated peaks.
P reduced KV FFN executed rows about 51% but increased calls 28→131; it retained
the same logical parameter-gather output bytes across more calls. Reduced padded
arithmetic alone did not pay for bucket construction, indexing and dispatch.

W borrows weight views but still organizes activations, builds owner counts and
assembles gradients. One owner plan serves three ordinary FFN projections; placed
wrappers currently rebuild it. Power2 buckets can also span many empty weight
groups. A single ragged grouped contraction with reused owner metadata is a future
candidate, not an observed speedup. Original numerical failures remain, and W/P/WP
were not extended to 9B; same-candidate restoration cannot qualify another path.

The large same-B CPU decode gap is smaller than the NPU gap while both absolute
CPU throughputs are lower. CPU/NPU parameter and input hashes match, but allocation
fallback work and attention backends differ. CPU NUMA3 diagnostic updates also
differ substantially from NUMA6 normal timing; later frequency/local-page snapshots
do not explain past latency. Do not label these ratios a pure hardware effect.

Synchronized NPU medium allocation requested 324 fallback rows while executing 49,152.
Compacting only requested rows is a bounded opportunity if thresholds, solution
faces and VJPs remain valid. Large inference separately exposes KV compaction,
field gathers, pre-Full SSD preparation and state joins. These stages mix necessary
math with organization; no unique additive percentage separates "unstructured"
from "complex arithmetic". Parent remainders and logical bytes do not establish
recoverable wall time or hardware bandwidth.
