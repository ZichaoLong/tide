# Precision policy and recorded qualification

The current implementation has FP32 and explicit BF16 arithmetic paths, exercised
on CPU and NPU in [fixed numerical qualification](evidence/roadmap-numerics-followup-20261118.md).
The [initial BF16 investigation](evidence/roadmap-numerics-initial-20261118.md) and
[original FP32 report](evidence/instances-fp32-20261118.md) retain their failures.
Implementation, completed execution and numerical equivalence are separate facts;
[STATUS](STATUS.md) owns the current handoff. CPU FP64 is available for bounded
accuracy diagnosis. FP16 is outside this cycle. Actual training and subsequent
training optimization prioritize FP32; BF16 remains a secondary option, as agreed
in [project scope](project-scope.md). This priority does not alter arithmetic
policies, thresholds or the qualification of existing records.

## Implemented arithmetic policy

Keep model parameters and floating inputs in FP32, import `precision_context`
from `tide.precision`, and use `with precision_context(device, "bf16"):` around
forward/loss; run backward and AdamW outside that context. `"fp32"` explicitly disables outer
autocast. Bare `model.to(bfloat16)` is not this policy. Matrix projections use
actual BF16 operations, while mass, allocation, sensitive scalar score contraction,
KV summaries and SSD recurrence/state use FP32. KV keys/values use BF16; original
times remain int64. Parameter gradients and AdamW moments remain FP32, with the
parameters themselves serving as master weights. Autocast's cast cache is disabled.

Explicit math/tiled Attention promotes BF16 Q/K/V to FP32 for its score,
normalization and value products, then returns the payload dtype. Native SDPA
arithmetic depends on its selected backend: the recorded CPU site audits
(including auto→SDPA) observed FP32 `bmm` for BF16 payloads. Native NPU fusion
has separate observed inputs/operators. See the [Attention evidence](evidence/roadmap-attention-20261118.md);
a payload dtype or API name alone does not identify the executed matrix kernel.
The integrated models still perform genuine BF16 projections outside these
explicitly stable subcalculations.

BF16 `auto` Attention uses tiled math whenever a visible key prior is zero,
with or without a recorded gradient. Positive explicit NPU priors use the same
rank-one encoding in inference and training; omitted unit priors retain the
native unit path. This avoids changing the arithmetic representation merely by
enabling gradients. Explicit backend choices remain separately declared paths;
FP32 retains its previous dispatch. Cross-precision routing may still change
when rounded scores cross a true allocation boundary.

`precision_policy` serializes this policy and `ArithmeticAudit` observes actual
operator/input/output dtypes in a separate intrusive call. Neither certifies an
unobserved hardware accumulator or kernel. `scripts/qualify_precision.py` compares
independent attention formulas and frozen integrated CPU fixtures, retaining
metadata, support, connectivity, nonfinite and numerical outcomes separately.

The production context accepts `fp32` and `bf16`. CPU `fp64` options in the
qualification/continuation diagnostics use explicit float64 tensors with autocast
disabled; they are not a third production autocast policy.

## Recorded qualification and boundaries

The seven-case CPU/single-NPU records cover inference, grad-enabled forward,
three continuing AdamW updates, saved data and an actual restored update.
BF16 inference/grad-forward values, support and state are exact in those records,
but all four HB comparisons against FP32 retain support/KV timestamp failures.
FP32 device/reference and some restored updates also retain finite deviations;
BF16 FC restored-update deviations remain. The fixed report gives each case's
source, threshold and outcome rather than an all-model precision pass.

Whole/chunk full-backward witnesses pass their existing thresholds in CPU FP64
and CPU/NPU FP32. BF16 retains finite value/state differences in FC-K, Exchange
and Timed, including with forced tiled Attention; their support, logical metadata
and gradient-connectivity checks remain distinct from numerical tolerance checks.
See the [bounded investigations](evidence/roadmap-numerics-followup-20261118.md)
for the scope of rounded-input allocation and repeated-backward/AdamW explanations.

Optional two-NPU [host-staged placement](placement.md) has separate
[FP32/BF16 evidence and finite failures](evidence/roadmap-placement-20261118.md).
The direct NPU transport's full-graph backward event failure remains recorded;
an isolated copy success does not qualify that path. CUDA hardware is unqualified.
The [TorchAir probe](evidence/roadmap-resident-20261118.md) selected the real backend
but stopped at a frontend static-role guard before invoking it; no resident
scheduler or compiled training was qualified.

## BF16 delivery contract

Cover inference, grad-enabled forward, full backward/AdamW, multiple updates,
private state continuation, checkpoint and an actual update after restore.
Declare separately parameter storage, matrix arithmetic, reductions, persistent
state, gradients, master weights and optimizer state. Record requested/effective
operator and device paths. A BF16 label must include genuine BF16 computation.

FP32 accumulation, normalization, regional solving or other stable subcalculations
are legitimate explicit policies; record their reason and cost. Preserve int64
logical times, counters, indices and memory timestamps. No silent whole-model FP32
substitution, arbitrary support clipping or dropped derivatives.

Expose non-finite values, cancellation, boundary sensitivity, under/overflow,
error accumulation, optimizer amplification, chunking and restored-update errors.
Use stable formulas, reduction organization and justified accumulation precision.
Three Attention families select kernels for their actual shapes/priors/dtypes;
Flash/fusion capability does not follow from the dtype name alone.

## Interpretation and bounded investigation

Keep values, support, metadata, disconnected gradients and checkpoint data checks
separate. Compare formula witnesses, same-device repeats, CPU/device and finite
FP64 references as needed. A higher-precision replay of the same code diagnoses
arithmetic; independent formulas check mathematics.

Record thresholds, original failures and the policy actually tested. Structural,
causal, solver and state errors need fixes; numerical thresholds are not weakened
merely to relabel failures. A branch switch near a rounded boundary needs an
explanation, not an automatic claim of identical support or an implementation bug.

Finite rounding differences receive bounded investigation and do not indefinitely
block unrelated work. Bitwise saved data/forward, tolerant replay and a bitwise
subsequent optimizer trajectory are different claims. Original and follow-up
FP32/BF16 restored-update failures remain attached to their actual records;
an exact next forward or a later within-threshold observation does not erase them.

Performance compares FP32 with FP32 and BF16 with BF16 first, then reports the
speed/memory/numerical tradeoff of changing policy. BF16 AdamW memory includes
actual FP32 masters/slots where used; it is not assumed to halve total memory.

## Regional batching and concurrent execution observations

The [current rounding investigation](evidence/capacity-rounding-20261118.md)
separates a corrected implementation error from finite layout changes. Adding an
FP32 bias after a BF16 regional matrix result promoted the activation; this was
fixed in `231f56eb`. Its zero-weight witness checks that special case and dtype,
not every Linear decomposition. CPU noncontiguous Linear can use BF16 bmm then
BF16 add, while baddbmm rounds the combined operation. The NPU witnesses instead
observe neighbouring BF16 values from addmm/baddbmm and mm/bmm near a rounding
midpoint. Independent Decimal allocation reproduces selected support changes.
The original support, timestamp and numerical failures remain recorded.

The [native slice](evidence/native-regional-performance-20261118.md) has separate
small checks and larger performance-shape comparisons; four BF16 five-update
comparisons and the original four-worker restored gradient remain outside their
strict thresholds. Successful forward values or exact checkpoint decoding do not
certify an identical subsequent gradient/optimizer trajectory.

The [new installed-wheel checks](evidence/capacity-installed-20261118.md) observe
exact checkpoint data/next forwards in four cases, while placed BF16 retains
restored gradient/update deviations. Same-owner per-device backward can already
vary at that selected checkpoint; another restored owner reproduces the original
update exactly. Four scalar FP64 AdamW witnesses explain selected update deltas,
not every gradient difference. Regional multi-NPU qualification and its original
tolerances are recorded [separately](evidence/region-capacity-20261118.md).

## Training candidate and mode checks

The [fixed-start training-path evidence](evidence/training-paths-20261118.md)
compares ordinary regions, compatible batches, whole/Reception recomputation,
prefix summaries and placed serial/concurrent execution from a nonempty private
state and AdamW anchor. It records inference, grad-forward, repeated backward,
updates and a new-process restored update separately. CPU FP64 independent
recurrence checks pass their stated scope; FP32/BF16 cross-path deviations remain.
Timed BF16 prefix combinations retain support and integer-memory-time differences
and are not selected as a training default. An unbatched prefix witness still
changes support after updates; its separately warmed anchor cannot identify the
cause of every difference in the original combined trajectory.

A targeted FP32 Attention witness locates the fourth Attention call at the
regional site: identical Q/K/V/prior values use native additive log-prior PSE in
inference and a rank-one Q/K extension with gradients. Same-input replays
reproduce the small output difference; eight complete query rows agree with an
independent scalar FP64 formula at the original FP32 tolerance. This explains
that local difference, not all later model/state deviations or their gradients.
The production dispatch policy was not changed to manufacture an exact match.

The [twenty-update record](evidence/training-continuous-20261118.md) retains a
large-batch FC BF16 paired restored-gradient/moment deviation. The selected HB
points pass their paired restored-update thresholds, while all six new-process
updates complete finitely. Those single-owner new-process timing runs do not
compare the full original-owner update and are not additional equivalence passes.

## Reception and static indexing follow-up

The [three-profile FP32 evidence](evidence/reception-numerics-20261118.md) covers
inference, grad-forward, repeated VJPs, three updates and nine candidate-local
new-process restores. Eight of nine paths retain finite deviations under the
original thresholds, without observed support/integer-time/connectivity changes.
All nine local restored updates are within threshold; this does not establish
cross-candidate training equivalence. Bounded CPU formulas and Decimal AdamW
witnesses explain their selected coordinates only; the NPU capture limit did not
locate an earliest causal divergence.

The [final integration evidence](evidence/reception-validation-20261118.md)
separately records BF16 light64 output-support differences after updates,
FP32 two-NPU restored parameter deviations, installed placed BF16 deviations,
and optional static-view FP32 update deviations. Static views preserve recorded
support/work in that small comparison, but do not promise a bitwise optimizer
trajectory. The five [100-update FP32 runs](evidence/reception-training-20261118.md)
are finite with paired restored updates within their original threshold; they
are training-shape observations and do not erase any of these cross-path failures.

The [subsequent four-device inference observations](evidence/transformer-depth-batch-20261118.md)
retain a B4 Full/output-support difference against two-device placement. The B8 pair has
matching work, effective-parameter, support and time summaries, with different packing.
Neither finite inference nor matching aggregate summaries establishes value/gradient
identity or adds four-device training and restored-update qualification.

## Explicit memory-binding candidates

The [five-family record](evidence/memory-candidates-20261119.md) preserves the
original FP32/BF16 thresholds. Small CPU/NPU FP32 comparisons retain finite
cross-path deviations; BF16 X and half-width KV each retain an output-support
difference after the third update. Candidate-local restored updates, actual
d768 narrow-KV formula/VJPs and medium FP32 continuous runs have separate scopes.
Successful inference does not qualify the FFN-adjusted near-parameter candidates
for training, nor new Timed or multi-device combinations.

The original 9B FP32 memory-shared `region_batch_size=1/4` inference pair retains
its discrete failure: identical parameter/input hashes and physical NPU, but
different output-support hashes and one extra Full action with size1. The
[bounded diagnosis and repair](evidence/cleanup-numerics-20261119.md) reproduced
that observation and located a wrong sorted-quad face after prefix cancellation.
The original complete-shape replay lost a positive control despite a zero total
residual. Commit `97c52960` checks the final face's KKT bounds and sends rejected
faces through the existing same-device, same-dtype fallback; no support epsilon
or production FP64 substitution was added.

The retained NPU value/support and independent score-VJP witness passes after the
repair. The post-fix 9B pair has matching internal support, output support and
original times, with Full 180209 on both paths; 298 internal scalar element
comparisons still exceed the original value threshold. Small FP32 training retains
1/1/8 failed cross-path fields over three updates, while each path's actual
fresh-process restored update is within threshold. These source-scoped checks do
not relabel old failures, certify every boundary, or transfer old throughput to
the repaired production source. BF16's selected third-update witness instead
crosses a true support boundary in the upstream rounded scores; its full upstream
cause remains unresolved.
