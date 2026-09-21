# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **481 tests passed** at
`50b5e943b4ba183d66d78a30a4e94468c6ed8349`.
See `evidence/m5a-state-programs.md`; prior evidence is linked from `ROADMAP.md`.
No active background jobs. Unit `tide-foundation-m5a-20260921-0900` completed
with exit 0; no worker remains.

M5B candidate now adds Linear Attention and gated DeltaRule matrix-state kernels,
including packed independent-sample steps and sequence scans. New targeted Python
checks: 48 passed. Native and full regression qualification is pending; see
`matrix-memory.md` for the dense Delta scan performance boundary.
Planned unit: `tide-foundation-m5b-20260921-0920`.
Command: `python scripts/job.py --output-dir artifacts/m5b-20260921-0920 -- python scripts/qualify.py --output-dir artifacts/m5b-20260921-0920`.
Freeze this checkout until terminal status. Inspect that directory's status,
task log and verification manifest. Stop with
`systemctl --user stop tide-foundation-m5b-20260921-0920`.

Implemented: CPU FP64/FP32 PositiveDelayGraph sparse streaming, native
serial/node-parallel and batch packing, TimedDAG frontier, SettleGraph
encoding/direct Python execution, independent self-loop/chain anchors, complete
trace/VJP comparisons, sharing, explicit detach and checkpoint v3. State kernels
now own preparation; native clients can supply their own StateKernel. EMA,
identity, diagonal selective SSM and tanh/SwiGLU Full profiles are present.
Refer to each report for the exact executor/profile cells tested.

Environment: aarch64; Python 3.11.15; Torch/LibTorch 2.10.0+cpu. Local Python:
`/home/zlong/anaconda3/bin/python`. CPU commands need
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` here to avoid unrelated NPU plugin auto-loading.
No packages were changed. CMake derives LibTorch from the selected Python.

## Next action

1. Finish M5B build/full qualification; fix any failures and commit evidence.
2. Add GQA/window attention with ragged KV representation, actual packed sample
   and sequence batches, explicit position/same-fiber policies and cache VJPs.
   See `module-extension-plan.md` and `state-programs.md` for interfaces.
3. Remaining broader scope: general region/Agg/Full programs, loss statistics,
   original LH C++ inference adapter (`lh-compatibility.md`), scale/performance.

## Boundaries

- The LH worktree contains user changes; do not modify or clean it. LH supplies
  inference compatibility only, never the training authority.
- Attention, Linear Attention, DeltaRule, original-LH numerical parity and large
  workload performance remain pending. Do not expand claims from SSM or EMA tests.
- Native SettleGraph runs the exact TimedDAG encoding; its graph compiler is a
  Python frontend. Native execution itself does not call Python.
- Evidence is tied to immutable source revisions. Build artifacts are ignored;
  verify source/binary fingerprints before fresh qualification.
