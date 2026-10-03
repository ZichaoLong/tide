# CUDA-linked host and exported-entry qualification

Clean implementation `2c04005255b3d0b674cef302894bb9a5d0f0378e`, 2026-10-03.
[Machine audit](eager-cuda-host-20261003.json) records exact source, binaries,
loader, terminal jobs and log hashes. This qualifies the aarch64 CUDA build and
CPU execution under that stack; NVIDIA device execution remains unverified.

Four jobs passed with exit0, inactive/dead services and empty cgroups:

| Job | Accepted scope |
| --- | --- |
| build-eager-cuda-clean01 | Fresh core, bindings and standalone executables, Release/C++17/GCC10.3.1; Torch2.10.0+cu128, CUDA Toolkit12.8.1, Python3.11.15, CXX11 ABI |
| build-eager-cuda-consumer-clean01 | Fresh external CMake consumer/static capacity probe using the installed core; loader contains CUDA libraries, no missing/stub/Python/NPU dependencies |
| eager-cuda-host-clean01 | 187 CPU FP64/FP32 owner/copy/capacity/consumer/CLI/checkpoint checks passed;47 accelerator cases explicitly deselected;0 skipped |
| eager-relocated-entry-clean01 | Seven Python/native/standalone/offline entry checks passed from a no-Git source export;25 cases explicitly deselected;0 skipped |

The headless CUDA configure used explicit `TORCH_CUDA_ARCH_LIST=8.0`. This is a
build setting, not an SM80 hardware test or a claim about every CUDA architecture.
The affected gate preserves independent CPU comparisons and explicit unavailable
CUDA rejection. It does not replace the complete historical regression suite.

All1522 source inventory hashes match. The exported tree has no `.git`, uses its
own Python sources, and has matching source-export manifest
`e69df87cdbf741fc62d2d6ef81a1c8ffc7d137f589cea95b216ef85cb46f58d8`.
The relocated gate explicitly reused source-matching CPU core/client binaries;
it did not rebuild the core from the relocated directory.

Raw records are `TASK/runs/NAME/{status.json,task.log}`; units are
`tide-execution-flows-NAME`. Frozen source is
`TASK/sources/eager-capacity-clean01`; builds are
`TASK/builds/eager-cuda-clean01` and `TASK/builds/eager-cuda-consumer-clean01`.
The export is `TASK/exports/eager-execution-2c04005`. Re-audit with
`python TASK/launchers/eager_cuda_host_evidence.py 2c04005255b3d0b674cef302894bb9a5d0f0378e`.
The audit checks retained raw manifests and artifacts; public paths are symbolic.

[Target commands](../eager-target-validation.md) provide fresh build, independent
CPU and actual two-device copy/owner/capacity gates. Real CUDA forward/backward,
optimizer/continuation, GPU dtype/driver/architecture combinations, GPU profiling
and throughput, and x86_64 execution remain target-pending. NPU capacity
calibration and full-size performance acceptance are separate. F1–F7 is open.
