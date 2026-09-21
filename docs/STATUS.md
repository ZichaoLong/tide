# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **843 tests passed** at
`900f19577f5bf6cead893ea53cf73a475f5dc1c1`.
See `evidence/native-cursor.md`; prior evidence is linked from `ROADMAP.md`.
Cursor qualification completed with exit 0. A new candidate adds joint
`[time,batch,width]` EMA/SSM scans, grouping nonempty equal-length samples and
compacting final persistent tensors. Targeted Python checks: 16 passed.
Native/full regression qualification is pending.

Planned unit: `tide-foundation-memory-pack-20260921-1006`.
Command: `/home/zlong/anaconda3/bin/python scripts/job.py --output-dir artifacts/memory-pack-20260921-1006 -- /home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir artifacts/memory-pack-20260921-1006`.
Freeze this checkout while active. Inspect that directory's `status.json`,
`task.log`, and `verification/result.json`; stop with
`systemctl --user stop tide-foundation-memory-pack-20260921-1006` if necessary.

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

1. Finish joint EMA/SSM batch/sequence qualification and commit evidence.
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
