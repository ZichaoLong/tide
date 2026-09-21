# Joint batch/sequence EMA and SSM, 2026-09-21

Clean source: `55abdd5badc31255e574c237be4ea7d624b839eb`.
Command: `python scripts/qualify.py --output-dir artifacts/memory-pack-20260921-1006`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **903 passed**, pytest 73.38 seconds.
Retain `artifacts/memory-pack-20260921-1006/{status.json,task.log,verification/}`.

New FP64/FP32 cases compare EMA/SSM Python grouped prefill, native grouped
prefill, native per-sample prefill and causal steps with the independent Python
time-major reference. All Emit modes, ragged inputs and independent initial-state
VJPs are covered. Existing 843 tests remain passing, including SettleGraph and
native cursor regressions.

Execution-shape tests establish that lengths 7/4/7 use two scan calls, with a
maximum [time,batch,width] shape [7,2,4]; the native ungrouped path uses three
calls with B=1. An idle fourth sample creates no candidate/state. Final persistent
read and SSM-memory tensors have compact storage under inference mode.

This profile's scan is an affine prefix algorithm over accepted observations.
No padding or recurrence crosses sample boundaries. Joint tensor execution and
storage checks do not establish wall-time speedup; optimized/chunked SSM kernels
and full-model compatibility require separate evidence.
