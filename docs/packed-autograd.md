# Packed values and semantic autograd connectivity

Packing independent events can change observable autograd behavior. For two
separately differentiable initial memories, a loss reading only sample 0 should
return `(gradient, None)`. A dense stack followed by a batched kernel and slice
can instead return `(gradient, zero)`. Parameters upstream of unused samples can
then acquire gradients and optimizer state, including AdamW weight decay.
Causal masks have the same issue for future observations. Numerical forward
agreement and objectives rooting all outputs do not detect this defect.

## Current correctness path

Python blocks and native streaming/blocks retain the requested packed forward
kernels. They evaluate packed state/read and Full without autograd recording.
When grad mode is enabled, they also construct independent scalar/event graphs
from the original per-event inputs and per-owner state, before stacking.

`autograd.value` / `semantic_value` returns an exact clone of the packed value.
Its backward routes the cotangent only into the independently constructed
reference graph. A nondifferentiable reference returns a detached packed tensor.
Shape, dtype, device and state metadata must agree. A sequence binds each state
before using it as the previous state of the next semantic step; subsequent
derivatives therefore use the actual packed forward state. Reads use these bound
proposals, and selected Full replays use the bound comparison and controls.

This preserves local dependency structure without a global liveness pass or
numerical-zero heuristics. True ancestors still receive connected zeros when the
loss is multiplied by zero. Caller-owned packed tensors keep ordinary tensor
VJPs: an unused row is zero, and any caller-side stacking retains its own edges.
Repeated `autograd.grad(..., retain_graph=True)` queries are supported.

## Cost and limits

- `semantic_state_replays` counts extra per-event step/read evaluations;
  `semantic_full_replays` counts extra selected Full evaluations. Existing
  batch/sequence/Full counters count the packed forward work separately.
  Native counters are accumulated by the coordinating thread.
- `no_grad` and `inference_mode` do no semantic replay. Their packing and node
  parallelism remain active. A missing replay counter is equivalent to zero.
- Training currently pays both packed numerical work and the scalar semantic
  graph's compute/storage cost. Packed call counts alone are not training-speed
  evidence. An optimized packed backward preserving the same isolated-root
  contract remains a performance task.
- The qualified profiles are deterministic and functional. Custom programs must
  satisfy the same step/batch contract. Random/stateful side effects need an
  explicit replay contract before they can use this path.
- Qualification concerns first-order VJPs, including the declared HST VJP.
  Roots include public trace/state/output tensors; differentiation inputs are
  model parameters, external inputs and initial-state leaves. Arbitrary internal
  intermediate-to-intermediate adjoints are not an executor equivalence claim.
  Higher-order derivatives, forward-mode AD and compiler transformations are
  not certified by these tests.

## Navigation and validation

Binding: `python/tidegraph/autograd.py`, `cpp/src/autograd.cpp`.
Preparation: `packing.py`, `block_prepare.cpp`, `stream.cpp`.
Full: `blocks.py`, `block.cpp`, `stream.cpp`.

`tests/isolated_cases.py` supplies separate input/initial-state leaves and
upstream parameters. `test_isolated_gradients.py` separately roots outputs,
pending messages, state slots and trace/control tensors across all five memory
families, Emit modes, FP64/FP32 and serial/parallel/packed/step paths.
`test_isolated_continuation.py` checks cuts, cursor snapshots, pending delivery
and optimizer updates. `test_isolated_schedules.py` checks independent chain and
self-loop schedules plus direct/encoded SettleGraph and caller-owned stacks.
Current qualification status is in `STATUS.md`; historical all-root evidence
does not certify this isolated-root contract.
