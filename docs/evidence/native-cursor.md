# Owned native streaming cursor, 2026-09-21

Clean source: `900f19577f5bf6cead893ea53cf73a475f5dc1c1`.
Command: `python scripts/qualify.py --output-dir artifacts/cursor-20260921-0954`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **843 passed**, pytest 69.42 seconds.
Retain `artifacts/cursor-20260921-0954/{status.json,task.log,verification/}`.

New FP64/FP32 coverage: mixed EMA/SSM/Linear/Delta/attention cyclic graphs,
parallel/unpacked versus serial/packed combinations, all Emit modes, selected
clear and observe-all/selected-only adoption. Multi-window traces, state/history,
pending messages and parameter/input/initial-slot VJPs match the independent
Python reference. Each-cut exports also match functional native streaming.

Import/export storage isolation, explicit state/pending detach, checkpoint
reimport, trace-disabled execution, shared-pool calls from two Python threads,
and inference-mode propagation pass. Invalid inputs leave the cursor retryable;
a delayed-edge time overflow marks it failed and requires snapshot recovery.
Standalone C++ validation callbacks prove 512 imported states are checked once,
with no repeat state validation during advances touching only two samples.
Existing 799 tests remain passing.

See `../streaming-cursor.md` for ownership and failure boundaries. Work-boundary
checks do not establish wall-time or scale performance. The cursor reuses the
qualified native event scheduler; independence comes from the Python scheduler,
while the functional native API checks materialization/ownership equivalence.
