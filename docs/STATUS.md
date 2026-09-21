# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **1918 tests passed** at
`4b5e14f6412c587d3ccec4ce75f1b6c88dbc3820`.
See `evidence/read-programs.md`; prior evidence is linked from `ROADMAP.md`.
Unit `tide-foundation-read-20260921-1311` completed with exit 0, is inactive
and has MainPID 0. Artifacts:
`artifacts/read-20260921-1311/{status.json,task.log,verification/}`.

Stable local input/output mappings, native flat inverse indexes, SettleGraph
remapping and graph/checkpoint identity guards are qualified. A mixed-input
SettleGraph case also now restores canonical fiber order after projecting source
tags. See `local-ports.md`. Native graph format is v9; checkpoint payload remains
v3, with a changed graph fingerprint. No implicit old-checkpoint migration.

Per-slot Full/Emit is qualified: native/Python program extension interfaces,
slot-affine parameters and sparse phase-based emissions across all schedules.
The concrete API and boundaries are in `full-programs.md`. No active job remains.
Aggregate's five profiles and extension seam are qualified. Graph-owned origin
views fix tag-sensitive custom programs under SettleGraph boundary encoding and
restore canonical program order. Raw routing/scales still use physical identity;
graphs without views allocate no origin index or additional source sort. The
native custom example includes mixed boundary/internal fibers and independent
forward/VJP formulas. See `aggregate-programs.md` and source-origin evidence.

No active job. Complete-content propagation, registered Python state programs,
independent Read programs and all three region modes are qualified. Read's
program requests expose only the selected state and preserve complete metadata;
see `content-programs.md`, `read-programs.md` and their evidence.

Next implementation is ready for clean qualification: complete requests,
registered custom programs, adopt/control-blend profiles, graph-owned clear,
state validation, native independent-node execution and prefill capability gates.
See `next-programs.md`. Development build `tide-foundation-next-build-20260921-1330`
passed, is inactive and has MainPID 0. Related CPU FP64/FP32 checks passed:
**122 tests in 17.52s**, including standalone custom Next and existing sharing.

Next qualification unit: `tide-foundation-next-20260921-1335`.
Command: `python scripts/qualify.py --output-dir artifacts/next-20260921-1335 --jobs 2`.
Artifacts: `artifacts/next-20260921-1335/{status.json,task.log,verification/}`.
Run after the implementation commit via scripts/job.py in background.slice.
Freeze source while active. Inspect MainPID/exit, status.json and verification
result.json; archive evidence separately only after successful termination.
Native graph format advances to v10; checkpoint payload stays v3. After this
qualification, replace the finished Next plan with the remaining region/history/
control plan, then implement that seam before LH profiles and numerical parity.

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

1. Inspect the clean Next qualification above and archive its evidence. Then
   implement region histories/controls and selector programs.
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
