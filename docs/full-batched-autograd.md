# Optional batched Full autograd

`Options.full_autograd = "replay" | "batched"` selects a native execution
implementation, outside graph/model/checkpoint identity. Default: `replay`.
Python's independent reference remains unchanged. The Python native adapter
accepts `full_autograd="batched"`; the scale executable and exported PDG runner
accept `--full-autograd batched`. Packed execution is required. Unknown choices
and Full programs without a declared implementation fail before execution.

## Scope and mechanism

The original packed training path computes Full numerically as a batch, then
recomputes each event with autograd enabled. RowEmit repeated the expensive dense
row projection for every selected event. The optional implementation performs
one batched affine operation and records a custom first-order VJP. It does not
skip Aggregate, Read, comparison, Next, Full values, output slots or delivery.
Aggregate/state/Read/Next semantic replay remains where previously applicable.

`isolated_linear.cpp` takes separate row tensors and one shared `[out,in]`
weight. It returns separate outputs with `set_materialize_grads(false)`.
Backward packs only rows with defined cotangents, computes `dX = dY W` and
`dW = dYᵀ X`, and returns undefined gradients for untouched row inputs.
A defined all-zero cotangent remains defined. Weight views and aliases pass
gradients to their existing owners. A caller-owned stack retains its own dense
tensor VJP. Frozen rows with a frozen weight keep `requires_grad=False` even
when another row is trainable. SavedVariable retains version checks on weights.

Small pointwise activation, normalization, residual, bias and Emit graphs stay
separate per event. This preserves independent Full-value/slot/control roots;
rooting a fresh Full value never connects a row-projection parameter used only
by emitted slots. ProjectionEmit batches the FFN/SwiGLU matrices and each
present slot projection; RowEmit batches its whole dense row projection and
then selects existing logical/phase slices. HARD/SOFTP/HST formulas stay the
same. The identity boundaries used by encoded SettleGraph remain identities.

The implementation is shared by packed streaming and legal frontier/block
calls, including independent native schedules and encoded/native SettleGraph.
This does not increase scheduler time batches or independent node dispatch.
In `no_grad`/`inference_mode` the original numeric batch remains in use.
`batched_full_events` counts selected events on the new grad path;
`semantic_full_replays` counts only actual old Full replay, and is zero there.

## VJP boundary

The contract is first-order differentiation from public roots to model owners,
external inputs and initial-state leaves, including isolated roots, shared and
unused owners, None/connected-zero, repeated retained-graph queries and HST's
declared surrogate. Higher-order requests through the new linear fail explicitly;
forward AD, compiled autograd and arbitrary intermediate adjoints are unqualified.
Only CPU FP64/FP32 is supported.

An unused row can cause an upstream backward node to receive an undefined
cotangent. Built-in SemanticValue and HST now propagate that undefined value,
instead of materializing zeros. Custom autograd ancestors must obey the same
undefined-cotangent contract to use this opt-in path. Arbitrary user custom
programs/hook behavior are not certified by built-in profile tests; retain
`replay` when this condition is not established. No claim of arbitrary model
or arbitrary custom-program support follows from this optimization.

## Verification and measurement

- `test_isolated_linear.py`: scalar formula, finite differences, independent and
  aliased rows, noncontiguous weights, frozen lanes, undefined/zero gradients,
  repeated roots, version checks, SGD/AdamW skip/update behavior.
- `test_row_emit_grad.py`: independent Python formula, scalar and packed native,
  fresh Full versus each phased slot, absent slots and shared row weight.
- `test_full_autograd.py`: complete trace/state/pending and isolated first-order
  roots for FFN/SwiGLU/LH profiles, slot/broadcast, HARD/SOFTP/HST and both schedulers.
- `test_full_autograd_training.py`: native PDG/DAG/Settle, independent schedules,
  native optimizers, owners, detach and continuation, cross-policy chunk cuts.
- Small scale `--check 1 --grad 1 --full-autograd batched` additionally checks
  complete results and isolated gradients against scalar RowEmit, including
  independent input leaves. Gradient checks require D<=64, B<=8 and steps<=12.

Performance must use the same graph, parameters, dtype, window and thread budget.
Grad-forward timing contains no backward or optimizer; backward correctness
does not establish backward throughput. Preserve earlier repeated Add evidence.
The bounded follow-up uses small smoke and D256/B32 cost probes before wide
D2048/B512, three independent optimized processes, and one fresh same-binary
replay control alongside the retained three-repeat baseline. Source/build,
resources, raw metrics, terminal status and remaining limits belong in the
reviewed report; STATUS holds its current completion state.

## Forward diagnostics

The native scale executable and PDG comparison runner permit `--work-count 1`
and `--operator-profile 1` with grad-forward. Counts include numeric execution
and any semantic replay; they are not unique logical work or backward FLOPs.
Replay timers sum calling-thread elapsed time and must not be added to wall
latency. Use a separate uninstrumented process for performance comparisons.
LH's grad-forward diagnostic restriction remains unchanged.
