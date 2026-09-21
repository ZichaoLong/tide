# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **1296 tests passed** at
`df2a851d759da9d6e5ccd7cef9ba931e416fdf07`.
See `evidence/local-ports.md`; prior evidence is linked from `ROADMAP.md`.
No active background jobs. Unit `tide-foundation-ports-20260921-1047`
completed with exit 0, no worker remains. Artifacts:
`artifacts/ports-20260921-1047/{status.json,task.log,verification/}`.

Stable local input/output mappings, native flat inverse indexes, SettleGraph
remapping and graph/checkpoint identity guards are qualified. A mixed-input
SettleGraph case also now restores canonical fiber order after projecting source
tags. See `local-ports.md`. Native graph format is v5; checkpoint payload remains
v3, with a changed graph fingerprint. No implicit old-checkpoint migration.

Next code increment: per-slot Full/Emit. The concrete API, example profiles,
integration sites and acceptance gates are in `full-emit-plan.md`; it is still
a plan and no FullKernel implementation has started.

The packed isolated-gradient defect is fixed for tested first-order public-root
VJPs to parameters, external inputs and initial-state leaves. Packed numerical
kernels retain their forward values; grad-enabled execution additionally builds
independent scalar/event graphs. Counters and performance limits are explicit in
`packed-autograd.md`. No numerical-zero-to-None conversion or global liveness
traversal is used. Inference has no replay.

Implemented: CPU FP64/FP32 PositiveDelayGraph sparse streaming, native
serial/node-parallel and batch packing, TimedDAG frontier, SettleGraph
encoding/direct Python execution, independent self-loop/chain anchors, complete
trace/VJP comparisons, sharing, explicit detach and checkpoint v3. Native owned
cursors preserve queues across windows. State kernels own preparation and accept
custom native implementations. EMA, identity, diagonal selective SSM,
Linear/Delta, event attention/GQA/window and tanh/SwiGLU profiles are present.
Refer to evidence for the exact executor/profile cells tested.

Environment: aarch64; Python 3.11.15; Torch/LibTorch 2.10.0+cpu. Local Python:
`/home/zlong/anaconda3/bin/python`. CPU commands need
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` here to avoid unrelated NPU plugin auto-loading.
No packages were changed. CMake derives LibTorch from the selected Python.

## Next action

1. Generalize Full/Emit to return per-slot values or absence using the qualified
   layouts. Follow `full-emit-plan.md` and preserve all independent schedules.
2. Extend source-aware Aggregate, full Next/Read, region history/selector, then
   loss statistics and original LH C++ inference comparison (`lh-compatibility.md`).
3. Performance qualification must address replay cost/optimized backward, cache
   allocation, structured Delta chunks and observed sparse work. No speed claim
   follows from kernel counts.

## Boundaries

- The LH worktree contains user changes; do not modify or clean it. LH supplies
  inference compatibility only, never the training authority.
- Original-LH numerical parity and large workload performance remain pending.
  Event attention is not LH same-fiber attention or pretrained-model compatibility.
- Native SettleGraph runs the exact TimedDAG encoding; its graph compiler is a
  Python frontend. Native execution itself does not call Python.
- Evidence is tied to immutable source revisions. Build artifacts are ignored;
  verify source/binary fingerprints before fresh qualification.
- Cleanup dry run found no eligible obsolete artifacts; nothing was deleted.
