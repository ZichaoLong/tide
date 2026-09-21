# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **843 tests passed** at
`900f19577f5bf6cead893ea53cf73a475f5dc1c1`.
See `evidence/native-cursor.md`; prior evidence is linked from `ROADMAP.md`.
No active background jobs. Unit `tide-foundation-cursor-20260921-0954` completed
with exit 0; no worker remains. The owned native streaming cursor keeps queues
and state native across cuts, with explicit snapshot/detach and error recovery.
See `streaming-cursor.md` for the ownership and finite-valued equivalence contract.

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

1. Implement joint batch/sequence SSM (currently one prefill scan per sample),
   using the checked packed-sequence interface. Compare with per-sample scans
   and independent stepping, including ragged samples and initial-state VJPs.
2. Generalize region/Agg/Full interfaces and loss statistics; review original LH
   C++ inference mapping (`lh-compatibility.md`) before choosing its exact profiles.
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
