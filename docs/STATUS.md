# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **1740 tests passed** at
`dd351c9524f8530929ee66e5d151f3176b85f890`.
See `evidence/source-origins.md`; prior evidence is linked from `ROADMAP.md`.
Previous qualification unit `tide-foundation-origin-20260921-1222`
completed with exit 0, no worker remains. Artifacts:
`artifacts/origin-20260921-1222/{status.json,task.log,verification/}`.

Stable local input/output mappings, native flat inverse indexes, SettleGraph
remapping and graph/checkpoint identity guards are qualified. A mixed-input
SettleGraph case also now restores canonical fiber order after projecting source
tags. See `local-ports.md`. Native graph format is v8; checkpoint payload remains
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

Complete-content implementation is ready for clean qualification: source tags,
summary and contributions now reach state/Read/Full, packed metadata and replay.
Python custom state programs register parameters and validate sharing; the native
adapter rejects unmatched Python overrides. See `content-programs.md`.
Development build `tide-foundation-content-build-20260921-1248` passed, is inactive
and has MainPID 0. Relevant FP64/FP32 checks: **619 passed in 54.70s**.
The preceding build `content-build-20260921-1240` failed because one test fixture
still used legacy fiber pointers; it is fixed. Both build artifacts are retained.

Next job: `tide-foundation-content-20260921-1250`, command
`python scripts/qualify.py --output-dir artifacts/content-20260921-1250 --jobs 2`.
It will run immediately after the implementation commit, via `scripts/job.py` in
`background.slice`, with two build workers and one ATen/BLAS thread. Artifacts:
`artifacts/content-20260921-1250/{status.json,task.log,verification/}`.
Freeze source until it terminates. Inspect service MainPID/exit and both JSON
records; a live job is not passing evidence. On success, archive an evidence
report in a separate commit; on failure, preserve logs and fix the reproducer.

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

1. Inspect complete-content qualification above and archive its result. Then
   implement region Read modes and independent readout programs, followed by
   complete Next requests and prefill capability guards (`next-read-plan.md`).
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
