# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **1727 tests passed** at
`4fa29b6d2291da006781aa456a15647e874d4fb1`.
See `evidence/aggregate-programs.md`; prior evidence is linked from `ROADMAP.md`.
Previous qualification unit `tide-foundation-aggregate-20260921-1203`
completed with exit 0, no worker remains. Artifacts:
`artifacts/aggregate-20260921-1203/{status.json,task.log,verification/}`.

Stable local input/output mappings, native flat inverse indexes, SettleGraph
remapping and graph/checkpoint identity guards are qualified. A mixed-input
SettleGraph case also now restores canonical fiber order after projecting source
tags. See `local-ports.md`. Native graph format is v7; checkpoint payload remains
v3, with a changed graph fingerprint. No implicit old-checkpoint migration.

Per-slot Full/Emit is qualified: native/Python program extension interfaces,
slot-affine parameters and sparse phase-based emissions across all schedules.
The concrete API and boundaries are in `full-programs.md`. No active job remains.
Aggregate's five profiles and extension seam are qualified, with a newly found
embedding limit: a custom program reading atom kind/origin position sees changed
tags after SettleGraph converts inputs to edges. Built-in local-slot programs
pass the tested embedding. Immediate next fix: graph-owned source-origin views,
preserving physical routing/scales but restoring program-visible tags and
canonical order. Add a minimal custom-program regression in Python and native.
Active implementation: origin views now map source kind/ID/position and restore
canonical program order before Aggregate, while resolving local slots and scales
from physical identities. Python custom-program regression and malformed-origin
checks passed 11 cases in FP64/FP32; native custom check adds mixed boundary/
internal fibers and hand-computed forward/VJP. Development build
`artifacts/origin-build-20260921-1215` exited 0 and targeted origin/Aggregate/
SettleGraph/port/CLI suites passed **340 tests**. A subsequent small refinement
avoids allocating the origin index and re-sorting sources when no view exists.
Rebuild `artifacts/origin-build-20260921-1219` exited 0, no worker remains;
source-origin/CLI checks passed 21 tests after the refinement. Next: commit this
fix, then qualify clean source with `python scripts/qualify.py --output-dir
artifacts/origin-20260921-1222`, unit `tide-foundation-origin-20260921-1222`.
Freeze source while active and archive evidence separately after terminal success.
Continue with `next-read-plan.md` after qualification.

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

1. Fix tag-sensitive custom Aggregate under SettleGraph embedding as above.
2. Extend full Next/Read, region history/selector, then
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
