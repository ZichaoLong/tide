# Precision: requested policy and measured capability

Current model qualification is FP32 on CPU/single NPU, with finite numerical
deviations recorded in [the report](evidence/instances-fp32-20261118.md).
The user has added BF16 to the next [work programme](ROADMAP.md). It is a target,
not a claim that current `.to(bfloat16)` or autocast use is supported/qualified.
CPU FP64 is available for bounded accuracy diagnosis. FP16 is outside this cycle.

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
