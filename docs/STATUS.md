# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest clean qualification: **903 tests passed** at
`55abdd5badc31255e574c237be4ea7d624b839eb`.
See `evidence/m5d-memory-packing.md`; prior evidence is linked from `ROADMAP.md`.
Unit `tide-foundation-memory-pack-20260921-1006` completed with exit 0.
EMA/SSM now have joint
`[time,batch,width]` scans, grouping nonempty equal-length samples and compacting
final persistent tensors. The native owned cursor remains qualified.

Active correctness work: isolated output roots exposed a
packing defect not covered by the 903-test all-root suite. Reading only sample 0
returns connected-zero gradients for independent sample-1 initial state through
packed state/Full operations; the scalar reference returns None. This can also
reach upstream parameters. Do not claim arbitrary isolated-root VJP equivalence
until clean qualification. Local semantic replay is implemented in Python and
native source; counters and limits are in `packed-autograd.md`. Targeted Python
and native checks: **364 passed**, FP64/FP32, including isolated public roots,
connected numerical zeros, independent parameters, optimizer updates,
serial/parallel/packed paths, cuts/cursor and SettleGraph/specializations.
Development native build `artifacts/autograd-build-20260921-1030` exited 0;
the preceding `1028` build failed on an include, now fixed. No worker remains.

Next: commit this implementation, then launch clean qualification under unit
`tide-foundation-autograd-20260921-1034`, artifacts
`artifacts/autograd-20260921-1034/{status.json,task.log,verification/}`.
Command: `python scripts/qualify.py --output-dir artifacts/autograd-20260921-1034`.
Freeze source while the job runs; inspect terminal status and archive evidence
in a separate commit. No original-LH numerical or performance run is included.

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

1. Finish the active isolated-gradient repair and clean qualification above.
2. Generalize local programs, starting with Full/Emit per-edge values and absent
   coordinates. See `module-extension-plan.md` for ordered gates. Preserve all
   existing independent schedules and exact SettleGraph boundary mapping.
3. Extend region history/selector and Aggregate/Next contracts, then loss
   statistics and original LH C++ inference comparison (`lh-compatibility.md`).
4. Performance qualification must also address cache allocation, structured Delta
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
