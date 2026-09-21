# Original LH selector and FP64 Read qualification

2026-09-21; clean implementation `805c74eab1fc8041dc83791e64700dcbc68e26af`.
Full foundation suite: **2318 passed in 178.55 seconds**. The outer job,
verification and oracle records all report passed, exit 0. Unit
`tide-foundation-lh-selector-20260921-1442` is inactive, MainPID 0.
Artifacts: `artifacts/lh-selector-20260921-1442/` (`status.json`, `task.log`,
`verification/{result.json,tests.log}`, `oracle/{result.json,build.log,float64.log,float32.log}`).

Qualification command: `python scripts/qualify.py --output-dir
artifacts/lh-selector-20260921-1442 --jobs 2 --lh-snapshot
artifacts/lh-source-20260921-1428`, via `scripts/job.py` in `background.slice`.
CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, C++11 ABI, two build jobs,
single-thread ATen/BLAS; `TORCH_DEVICE_BACKEND_AUTOLOAD=0`.

## Original-source inference checks

The unchanged snapshot has 69 C++/header files, identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.
Its manifest captures actual dirty LH source at HEAD
`5fd237d40c9880ccb6e511e4bf20799c7022fd1e`; the reference worktree was only read.
Original Selector, Adjacency, GraphConfig and BaseConfig are compiled with CPU
runtime assertions and `-UNDEBUG`. The runner checks snapshot inventory/hashes,
Tide source/library identities and oracle CMake identity before and after runs.

Both FP64 and FP32 logs report: **12 cases, 288 ticks, 7368 candidate
occurrences**, original heap/tensor vs Tide serial, node-parallel packed streaming
and frontier. Selected IDs/payloads and selected/affected history maps match at
every tick. Cases cover sparse IDs, ties, lead-point inclusion, budgets 1/2,
ragged/missing samples, whole idle ticks and forced activity.

## Tide training and precision checks

The full suite includes direct FP64 norm accumulation for FP32 payloads, explicit
descriptor promotion and payload control/history conversion, independent norm
VJPs including zero, selected/affected route formulas, checked int64 counters,
cuts and cross-schedule public-root gradients. A rounded FP32 norm cast to FP64
fails the close-score anchor. LH's detached selector scores do not define training.

This certifies a **selector component**, not whole-LH inference. Original int32
counters and composite double ranks are exercised only in a safe small-count
domain; overflow and composite precision limits are excluded. Add idle decay,
same-fiber attention, source signaling and Pronounce remain separate gates.
These are correctness results, not performance measurements.
