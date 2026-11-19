# Current instance implementation: measured lessons

The first observations concern the implementation at c17c613f and the
[FP32 delivery report](evidence/instances-fp32-20261118.md); the later measured
extensions below retain their own sources. They guide a relevant
change; they are not a compulsory experiment matrix or a reason to recover old
code. Semantic correctness comes first, performance second.

## Preserve tensor flow and mathematical ownership

The public evaluator is already a direct region-window computation. B and L
remain tensor axes through reception, memory preparation, regional decisions,
selected Full and publication. There is no token/event object reconstruction
between those stages. Region loops and necessary memory chunks remain visible.

A uniform shared base can multiply all node/B/position rows with one GEMM,
while private low-rank adapters use node BMMs. Static owner supersets avoid a
host read of dynamic device IDs. Resolve sharing independently inside each
module: the mixed-region KV base exception differs from other parameter groups.
Independent tests compare explicit matrices and all gradients, including
repeated owners. Mixed groups conservatively use BMM. Shared parameters do not
make private state shared or eliminate per-node computations.

The paired Lattice change (GEMM plus quad solve) measured CPU inference
2,261→2,303 and training 334→387 positions/s; shared NPU inference 3,956→5,531
and training 2,020→2,217. These are finite observations, not universal gains;
NPU training's audit work differs slightly. Historical before/after numbers
with different inputs/affinities must not be attributed to this one change.

## Finite allocation solve must preserve support

Quad has a piecewise-linear mass response. Sorting its enter/leave breakpoints
allows a finite O(n log n), O(n)-storage solve. The analytic first-order VJP
belongs to the solved mapping, not to the sorting/bisection iterations.
Keep a safeguarded fallback for collapsed or ill-conditioned floating intervals.
Shannon's current implementation still uses its bounded iterative solve.

Prefix subtraction initially created 2.22e-16 at an exact zero endpoint. The
weighted output looked close but it triggered an extra Full. Rebuild the
selected face from its remaining mass with a local score origin and exact
endpoint handling. Do not solve this with arbitrary epsilon clipping, which
would erase real tiny positive senders. See test_allocation.py and the
independent FC/Timed tests; the first integrated failures are recorded.

## Count selected and padded Full rows

A tensor labeled packed can still contain material padding. The optional
`python -m benchmarks.profile` reports selected_query_packing alongside stage
times. At B8/L128, the measured FC-K packs 9,894 selected rows into 19,968
rectangular rows (2.02×); Lattice packs 74,665 into 80,000 (1.07×). Query grouping
or ragged kernels may help the former more than the latter. Preserve original
logical time and weighted Attention visibility while changing layouts.

Likewise k/n is a sending-mass ratio, not a hard Full fraction. Eight equal
scores with k1 can yield eight 1/8 senders. FC-S and shared-base Lattice here
perform 99.9% and 91.3% Full/candidate, respectively. Actual sparse workloads
must be demonstrated, not inferred from configuration names or parameter count.

## Treat the three Attention sites separately

Historical KV has temporal windows; reception attends over same-tick incoming
messages; regional Attention attends over candidates. Their Q/K dimensions and
prior gradients differ. Compare native fusion, ordinary batched math and tiled
math on the actual shapes. The present auto policy chooses by backend/prior
applicability; it is not an exhaustive shape-tuned performance policy.

Positive trainable NPU key priors can be encoded as an extra Q/K feature while
retaining the original scale and native Q/K backward. Visible trainable zero
priors need the weighted quotient's right derivative: the tiled/custom backward
preserves it, whereas silently taking log(0) or dropping that derivative does not.
Tiled math recomputes 64-query/256-key blocks in backward rather than retaining a
full score matrix. These observations qualified FP32 only. The new BF16 work and policy are in
[ROADMAP](ROADMAP.md) and [precision](precision.md).

CANN's null-PSE mode is an actual backend contract: no PSE tensor requires the
default pse_type. Both the original runtime failure and the fixed 10-case NPU
qualification are retained. Compilation or CPU fake-adapter checks alone did
not establish this device behavior.

NPU logs also place int64 Argsort on device AiCPU for KV cache ordering. This
does not mean a host CPU transfer, but it is not AiCore execution. A stable
compaction alternative may help; blindly casting timestamps to FP32 would lose
their required integer ordering. Preserve such backend warnings in evidence.

## Bound numerical diagnosis and performance work

Check support/metadata/None gradients/exact checkpoint separately from numerical
tolerance. Preserve finite deviations and use a small FP64 replay when it can
identify arithmetic sensitivity. In the worst Timed witness, both CPU/NPU FP32
had small deviations from CPU FP64, all parameter gradients passed against FP64,
and FP64 continuation differed by 4.22e-14. This supports a finite FP32 diagnosis;
it neither relaxes the original threshold nor substitutes for independent formulas.
`python -m scripts.diagnose_precision --config configs/precision-timed-x.json`
is the reproducible entry (supply admitted device and a new output path).

Exact checkpoint data/forward and a bitwise subsequent optimizer trajectory
are separate claims. The installed FC-K replay performs the next update too:
CPU/NPU values remain finite, but a common regional bias retains small parameter
tolerance deviations (observed maxima across parameters 1.20e-5/1.79e-5).
AdamW can amplify tiny near-null gradient discrepancies. Preserve the declared
threshold and the deviation, rather than calling a successful process an exact
replay. `examples/train.py` now reports this distinction directly.

Use an additive direct computation/comparison when it clarifies a ceiling, and
intrusive stage/operator observation to explain a remaining gap. A different
Transformer/SSD function is a compute reference, not an equivalence oracle.
The current CPU Lattice profile is dominated by large shared FFN gradients and
private-adapter BMMs; the NPU profile identifies Full and allocation cost. Stop
with a concrete measured opportunity or a documented limit instead of starting
an open-ended fusion campaign. Larger modules or downstream needs can reopen it.

## Pair inputs and record real resources

A seed alone does not pair CPU and NPU random generators. Construct inputs and
initial parameters on CPU, retain hashes, then transfer outside timing. Sort
role names before seeded parameter construction. Keep model-source and
harness-source identities separate when comparing earlier implementations.
Training can subsequently diverge in floating routing; retain audit work too.

Inference, grad-enabled forward and complete AdamW training are distinct modes.
Fresh prefill throughput is B×L/time; it gives neither decode nor first-token
latency. Keep Timed pending/flush boundaries explicit. Record raw samples:
short CPU measurements here fluctuate and shared NPU measurements are not
exclusive timings. An 8-thread Lattice probe was slower than 32 here; available
CPU count is permission, not a universally optimal thread count.

Distinguish process CPU affinity, Torch threads and memory enforcement. This
host's user cgroup MemoryMax requests were not hard limits; scripts/job.py uses
a sampled process-tree RSS bound. The current jobs stayed well below their
bounds, with ample disk quota headroom. Check storage before large runs and
report a real blocker rather than silently serializing or abandoning work.

## Measured extensions in the five-workstream programme

- Specialize after whole-path measurements. NPU stable KV compaction and compatible
  head-query fusion reduced the measured training times; CPU retained stable sort
  and separate heads. Full bucketing regressed despite reducing padded work, and
  extra gate/up fusion did not have convincing whole-model benefit. Some later
  routing differed, so the observations are not strict same-trajectory speedups.
  See [the finite candidate record](evidence/roadmap-hotspots-20261118.md).
- Low Full support leaves other required work. The legal constructed 3.125% and
  9.375% fixtures still perform candidate preparation and regional computation;
  their construction is not evidence of learned sparsity. Measure these costs and
  sharing/time deduplication explicitly in [four-class comparisons](benchmarking.md).
- Attention shape matters. Long unit-prior KV favored native fusion; small
  reception/regional shapes favored ordinary math in the post-projection witness.
  The latter remains a candidate for integrated validation, not a blanket dispatch
  change. Zero-prior math/tiled has a memory/time tradeoff. See
  [the site comparison](evidence/roadmap-attention-20261118.md).
- Precision changes can alter allocation support before the solve. Independent
  Decimal solutions agreed with each rounded input in the examined crossings;
  promoting only the final solve cannot recover earlier scores. Separately,
  keeping BF16 inference/training Attention representations consistent repaired a
  mode-dependent discrepancy. [The numerical reports](evidence/roadmap-numerics-followup-20261118.md)
  retain cross-precision support differences and finite whole/chunk errors.
- Exact checkpoint data and forward are weaker than exact training continuation.
  FC-S BF16 backward varied even on the same owner. Selected actual AdamW update
  differences agree with scalar FP64 predictions from the measured gradients;
  a restore-only or common-bias-only explanation would be too narrow.
- BF16 arithmetic does not halve FP32-master AdamW storage. Actual 0.5B/1B
  short-window training peaks remain near 9.2/18.2–18.6 GiB in
  [the scale witnesses](evidence/roadmap-scale-small-20261118.md).
  Budget the widest layer and temporary storage, not only aggregate device RAM.
- An isolated differentiable copy is insufficient for a multi-device model.
  Complete HB backward exposed an event failure; explicit CPU host staging
  completed updates and restore checks with retained finite deviations. This is
  a transport choice with real CPU cost, not a peer-DMA or HCCL claim.
- A fullgraph capture blocker is local evidence. The observed static-role guard
  can be moved out of capture; it does not prove resident execution impossible.
  A backend callback count of zero establishes no backend lowering or autonomous
  device-controlled progression. Keep the [bounded probe](evidence/roadmap-resident-20261118.md)
  distinct from a resident implementation.
