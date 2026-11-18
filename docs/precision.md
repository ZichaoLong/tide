# Precision: requested policy and measured capability

Original model qualification is FP32 on CPU/single NPU, with finite numerical
deviations recorded in [the report](evidence/instances-fp32-20261118.md).
The current implementation adds the explicit BF16 arithmetic policy below.
New device qualification and finite deviations belong to [STATUS](STATUS.md);
the implementation alone does not establish numerical equivalence.
CPU FP64 is available for bounded accuracy diagnosis. FP16 is outside this cycle.

## Implemented arithmetic policy

Keep model parameters and floating inputs in FP32 and use
`with tide.precision.precision_context(device, "bf16"):` around forward/loss;
run backward and AdamW outside that context. `"fp32"` explicitly disables outer
autocast. Bare `model.to(bfloat16)` is not this policy. Matrix projections use
actual BF16 operations, while mass, allocation, sensitive scalar score contraction,
KV summaries and SSD recurrence/state use FP32. KV keys/values use BF16; original
times remain int64. Parameter gradients and AdamW moments remain FP32, with the
parameters themselves serving as master weights. Autocast's cast cache is disabled.

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
subsequent optimizer trajectory are different claims. The present FP32 report
retains small restored AdamW parameter deviations; governance does not resolve them.

Performance compares FP32 with FP32 and BF16 with BF16 first, then reports the
speed/memory/numerical tradeoff of changing policy. BF16 AdamW memory includes
actual FP32 masters/slots where used; it is not assumed to halve total memory.
