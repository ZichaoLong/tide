# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **619 tests passed** at
`ca0a12764ab0f766ba86f46c84a6d60f2e35e303`.
See `evidence/m5b-matrix-memory.md`; prior evidence is linked from `ROADMAP.md`.
No active background jobs. Unit `tide-foundation-m5b-20260921-0920` completed
with exit 0; no worker remains.

M5B adds qualified Linear Attention and gated DeltaRule matrix-state kernels,
including packed independent-sample steps and sequence scans. Mixed SSM/Linear/
Delta SettleGraph embeddings are covered. See `matrix-memory.md` for the dense
Delta scan performance boundary. M5C attention is the next implementation.

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

1. Add GQA/window attention with ragged KV representation, actual packed sample
   and sequence batches, explicit position/same-fiber policies and cache VJPs.
   See `module-extension-plan.md` and `state-programs.md` for interfaces.
2. Remaining broader scope: general region/Agg/Full programs, loss statistics,
   original LH C++ inference adapter (`lh-compatibility.md`), scale/performance.

## Boundaries

- The LH worktree contains user changes; do not modify or clean it. LH supplies
  inference compatibility only, never the training authority.
- Attention, original-LH numerical parity and large
  workload performance remain pending. Do not expand claims from SSM or EMA tests.
- Native SettleGraph runs the exact TimedDAG encoding; its graph compiler is a
  Python frontend. Native execution itself does not call Python.
- Evidence is tied to immutable source revisions. Build artifacts are ignored;
  verify source/binary fingerprints before fresh qualification.
