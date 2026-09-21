# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **1267 tests passed** at
`c04b89eb5e2ab414a3367ae4ae8b65f24cc08768`.
See `evidence/isolated-autograd.md`; prior evidence is linked from `ROADMAP.md`.
No active background jobs. Unit `tide-foundation-autograd-20260921-1034`
completed with exit 0, no worker remains. Artifacts:
`artifacts/autograd-20260921-1034/{status.json,task.log,verification/}`.

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

1. Generalize local programs. First establish validated per-node input/output
   slots outside shared parameter modules, with compact native inverse indexes,
   SettleGraph boundary remapping and graph/checkpoint identity guards. Then
   Full/Emit can return per-slot values or absence. See `module-extension-plan.md`.
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
