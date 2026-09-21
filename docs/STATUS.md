# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **799 tests passed** at
`652f2e7a7a09f4dcf6b220cbc058c580cc10c41d`.
See `evidence/m5c-attention.md`; prior evidence is linked from `ROADMAP.md`.
M5C's unit completed with exit 0. The next candidate adds an owned native
streaming cursor with persistent queues, incremental input validation, explicit
snapshot/detach, and failure recovery. See `streaming-cursor.md`. Python reference
regressions: 84 passed. Cursor/native/full regression qualification is pending.

Planned unit: `tide-foundation-cursor-20260921-0954`.
Command: `/home/zlong/anaconda3/bin/python scripts/job.py --output-dir artifacts/cursor-20260921-0954 -- /home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir artifacts/cursor-20260921-0954`.
Freeze this checkout while active. Inspect that directory's `status.json`,
`task.log`, and `verification/result.json`; stop with
`systemctl --user stop tide-foundation-cursor-20260921-0954` if necessary.

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

1. Finish cursor build/full qualification; fix failures and commit evidence.
   The functional streaming/Python paths remain equivalence anchors.
2. Generalize region/Agg/Full interfaces and loss statistics; review original LH
   C++ inference mapping (`lh-compatibility.md`) before choosing its exact profiles.
3. Performance qualification must also address cache allocation, structured Delta
   chunks and joint batch/sequence SSM (currently one prefill scan per sample).

## Boundaries

- The LH worktree contains user changes; do not modify or clean it. LH supplies
  inference compatibility only, never the training authority.
- Original-LH numerical parity and large workload performance remain pending.
  Event attention is not LH same-fiber attention or pretrained-model compatibility.
- Native SettleGraph runs the exact TimedDAG encoding; its graph compiler is a
  Python frontend. Native execution itself does not call Python.
- Evidence is tied to immutable source revisions. Build artifacts are ignored;
  verify source/binary fingerprints before fresh qualification.
