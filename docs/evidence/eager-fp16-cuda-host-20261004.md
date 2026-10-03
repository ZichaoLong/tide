# Updated eager CUDA-linked consumer, host qualification

Clean source `c6ef22474f658ecb12dd11310c710781023932ab`, 2026-10-04.
The fresh external consumer includes FP16 payload gradients with FP32 masters
and the corrected CPU training allocation allowance. Its existing CUDA core
has exactly matching C++ sources. [Audit](eager-fp16-cuda-host-20261004.json).

Both build and affected gate passed with exit0, inactive services and empty
cgroups. **108 CPU checks passed in 80.50 seconds**, no skips or deselections:
capacity, traffic bounds, eager precision and nonboolean automatic-chunk rejection.
The prior 187 checks and relocated-entry qualification were not repeated.

Torch 2.10.0+cu128, CUDA Toolkit 12.8.1, aarch64, Python 3.11.15, GCC 10.3.1,
C++17 and CXX11 ABI. The headless build uses architecture 8.0. Loader closure
contains real CUDA libraries and no missing/stub/Python/NPU dependencies.
All frozen source, core binaries, installed libraries and consumer hashes match.

This is build and CPU evidence. NVIDIA execution, GPU FP16/FP32/FP64 parity,
CUDA profiling/performance and x86_64 qualification remain target-pending.
[Target commands](../eager-target-validation.md) retain those separate gates.

Raw: `TASK/runs/{build-eager-half-cuda-clean01,eager-half-cuda-host-clean01}`;
consumer: `TASK/builds/eager-half-cuda-clean01`; exact commands in the audit.
Re-audit with `python TASK/launchers/eager_half_followup_evidence.py`.
