# Precision policy and recorded qualification

The current implementation has FP32 and explicit BF16 arithmetic paths, exercised
on CPU and NPU in [fixed numerical qualification](evidence/roadmap-numerics-followup-20261118.md).
The [initial BF16 investigation](evidence/roadmap-numerics-initial-20261118.md) and
[original FP32 report](evidence/instances-fp32-20261118.md) retain their failures.
Implementation, completed execution and numerical equivalence are separate facts;
[STATUS](STATUS.md) owns the current handoff. CPU FP64 is available for bounded
accuracy diagnosis. FP16 is outside this cycle.

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
