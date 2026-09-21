# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **619 tests passed** at
`ca0a12764ab0f766ba86f46c84a6d60f2e35e303`.
See `evidence/m5b-matrix-memory.md`; prior evidence is linked from `ROADMAP.md`.
M5B's unit completed with exit 0. M5C candidate adds aggregated-event GQA/window,
ragged K/V slots, checked packed-sequence metadata, and actual batch/sequence
attention groups. Targeted Python attention/schedule/metadata checks: 60 passed.
Native/full regression qualification is pending; see `attention.md`.

Planned durable unit: `tide-foundation-m5c-20260921-0939`.
Command: `/home/zlong/anaconda3/bin/python scripts/job.py --output-dir artifacts/m5c-20260921-0939 -- /home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir artifacts/m5c-20260921-0939`.
Freeze this checkout while the unit is active. Inspect its `status.json`,
`task.log`, and `verification/result.json`; stop if necessary with
`systemctl --user stop tide-foundation-m5c-20260921-0939`.

Implemented: CPU FP64/FP32 PositiveDelayGraph sparse streaming, native
serial/node-parallel and batch packing, TimedDAG frontier, SettleGraph
encoding/direct Python execution, independent self-loop/chain anchors, complete
trace/VJP comparisons, sharing, explicit detach and checkpoint v3. State kernels
now own preparation; native clients can supply their own StateKernel. EMA,
identity, diagonal selective SSM, Linear/Delta and tanh/SwiGLU profiles are present.
Refer to each report for the exact executor/profile cells tested.

Environment: aarch64; Python 3.11.15; Torch/LibTorch 2.10.0+cpu. Local Python:
`/home/zlong/anaconda3/bin/python`. CPU commands need
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` here to avoid unrelated NPU plugin auto-loading.
No packages were changed. CMake derives LibTorch from the selected Python.

## Next action

1. Finish M5C build/full qualification; fix failures and commit evidence separately.
2. Generalize region/Agg/Full interfaces and loss statistics; review original LH
   C++ inference mapping (`lh-compatibility.md`) before choosing its exact profiles.
3. Native owned streaming cursor/advance API is needed for very sparse, large
   workloads: current functional windows copy/validate all cached state each cut.
   Keep the functional executor as an oracle. Performance qualification must also
   address cache allocation, structured Delta chunks and observed batching.

## Boundaries

- The LH worktree contains user changes; do not modify or clean it. LH supplies
  inference compatibility only, never the training authority.
- Attention, original-LH numerical parity and large
  workload performance remain pending. Do not expand claims from SSM or EMA tests.
- Native SettleGraph runs the exact TimedDAG encoding; its graph compiler is a
  Python frontend. Native execution itself does not call Python.
- Evidence is tied to immutable source revisions. Build artifacts are ignored;
  verify source/binary fingerprints before fresh qualification.
