# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **903 tests passed** at
`55abdd5badc31255e574c237be4ea7d624b839eb`.
See `evidence/m5d-memory-packing.md`; prior evidence is linked from `ROADMAP.md`.
No active background jobs. Unit `tide-foundation-memory-pack-20260921-1006`
completed with exit 0; no worker remains. EMA/SSM now have joint
`[time,batch,width]` scans, grouping nonempty equal-length samples and compacting
final persistent tensors. The native owned cursor remains qualified.

Implemented: CPU FP64/FP32 PositiveDelayGraph sparse streaming, native
serial/node-parallel and batch packing, TimedDAG frontier, SettleGraph
encoding/direct Python execution, independent self-loop/chain anchors, complete
trace/VJP comparisons, sharing, explicit detach and checkpoint v3. State kernels
now own preparation; native clients can supply their own StateKernel. EMA,
identity, diagonal selective SSM, Linear/Delta, event attention and tanh/SwiGLU
profiles are present.
Refer to each report for the exact executor/profile cells tested.

Environment: aarch64; Python 3.11.15; Torch/LibTorch 2.10.0+cpu. Local Python:
`/home/zlong/anaconda3/bin/python`. CPU commands need
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` here to avoid unrelated NPU plugin auto-loading.
No packages were changed. CMake derives LibTorch from the selected Python.

## Next action

1. Generalize local programs, starting with Full/Emit per-edge values and absent
   coordinates. See `module-extension-plan.md` for ordered gates. Preserve all
   existing independent schedules and exact SettleGraph boundary mapping.
2. Extend region history/selector and Aggregate/Next contracts, then loss
   statistics and original LH C++ inference comparison (`lh-compatibility.md`).
3. Performance qualification must also address cache allocation, structured Delta
   chunks and observed sparse work; no speed claim follows from kernel counts.

## Boundaries

- The LH worktree contains user changes; do not modify or clean it. LH supplies
  inference compatibility only, never the training authority.
- Original-LH numerical parity and large workload performance remain pending.
  Event attention is not LH same-fiber attention or pretrained-model compatibility.
- Native SettleGraph runs the exact TimedDAG encoding; its graph compiler is a
  Python frontend. Native execution itself does not call Python.
- Evidence is tied to immutable source revisions. Build artifacts are ignored;
  verify source/binary fingerprints before fresh qualification.
