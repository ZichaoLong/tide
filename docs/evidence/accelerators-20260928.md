# Accelerator library qualification — 2026-09-28

Implementation: `10cd630` (after Python gate foundation `adeb819`). Verification-only
nested-environment correction: `bf4ac1c`; graph/runtime/C++ code is unchanged by
that correction. Structured environments, case inventories, artifact hashes and
command templates are in [the evidence manifest](accelerators-20260928.json).
Build products and full per-case reports remain under `artifacts/accelerator-*`.

The complete clean CPU gate passed **8636 tests** in 1618.89 seconds, all **22**
complex topology cells and **10** installed Python/native/C++ checks. The
verification-only follow-up also passed its ten CPU installed-consumer checks.
All 22 final qualification jobs are terminal with zero exit status; the evidence
manifest records their exact source identities and report hashes.

## Verified device matrix

Host: Linux aarch64, Python3.11.15, GCC10.3.1, CXX11 ABI=1. Device:
`Ascend910_9392` (A3), public driver25.3.rc1. Each run received one dynamically
queued physical chip mapped to `npu:0`. Physical placement and exact launch
commands remain in its queue/status records; this is not a performance comparison.

| CANN | Torch / TorchNPU | Python complete cases | Native adapter complete cases |
| --- | --- | --- | --- |
| 8.5.0 | 2.9.0+cpu / 2.9.0 | 39 passed | 43 passed; CSR rejection passed |
| 8.5.1 | 2.9.0+cpu / 2.9.0 | 39 passed | 43 passed; CSR rejection passed |
| 8.5.2 | 2.9.0+cpu / 2.9.0 | 39 passed | 43 passed; CSR rejection passed |
| 9.0.0 | 2.10.0+cpu / 2.10.0 | 39 passed | 43 passed; CSR rejection passed |

All 328 supported configuration gates passed from clean `10cd630` using default
FP32 `atol=1e-6, rtol=1e-5`. Case names and full requested/effective configurations
are retained. The 8.5.1/8.5.2 rows are observations of these installed combinations;
they do not broaden the vendor's published compatibility matrix. No new CANN
installation was needed. NPU FP64, the explicitly FP64 Read profile and optional
CSR fiber pooling remain unsupported. The adapter fails explicitly; it never
converts CSR, lowers precision or silently selects another execution policy.

Every supported case compares an independent CPU schedule: full values and exact
routes, state/history/pending, isolated VJPs including None connectivity, chunks,
trace-disabled execution, three AdamW updates, fresh-process checkpoint restore
and NPU-to-CPU handoff. Coverage includes three graph families, four specializations,
mixed state modules, same-fiber memories, LH activation/norm Full, Aggregate
profiles, selectors, sparse emission/reset, periodic clocks and delayed parallel
edges. Native adds worker/packing/cache/layout policies. See
[the reproducible commands and exact scope](../accelerators.md). These finite
checks do not certify arbitrary modules, graph sizes, training convergence or
performance. Full-scale experiment validation remains the caller's responsibility.

## Independent C++ and installed consumption

The standalone NPU SDK is built from TorchNPU
`94f8a8e6b523d7ba553e1b80d5b5248478391526`, against Torch2.10/CANN9.0, separately
from the Python wheel runtime. Its SDK library/config hashes and matching Torch
build identity are recorded. C++ initialization uses the SDK public interface;
seeding uses its exported NPU generator because this release does not implement
the generic PrivateUse1 default-generator hook.

`sdk-a2` passed independent CPU Streaming versus NPU ring/diamond schedules,
shared parameter owners, full observables, backward, three AdamW steps,
checkpoint values/optimizer slots restored onto NPU and CPU, and subsequent
updates. It asserts worker inheritance of a nondefault device stream. The
standalone process has no Python runtime dependency. The native owner gate
separately passed SGD momentum and AdamW with aliases, disconnected/zero gradients,
optimizer slot placement, same-device/CPU restore and subsequent updates.

Installed Python/native NPU use passed nine checks: three families through the
copied external application and the installed qualifier/fresh-process resume.
The independently installed CMake NPU consumer also passed allocation, chunking
and backward. Its loader has neither unresolved/stub libraries nor
`libtorch_python`/`libpython` dependencies. Standalone SDK qualification is only
for 2.10/CANN9.0; the 2.9 rows qualify native execution through Python registration.

Current checkpoint versions are unchanged. Python sessions retain graph
continuation. Native `TIDENCK1` remains named weights and optimizer state, without
adding graph-continuation serialization. Portable numerical handoff does not
promise cross-vendor RNG or bitwise optimizer trajectories.

## Placement evidence

Candidate-only warm forward/backward/AdamW profiles at `10cd630` recorded:

| Process | Accelerator kernel events | Scalar / transfer events |
| --- | --- | --- |
| Python mixed TimedDAG | 1295 | 193 `_local_scalar_dense`, 96 `MEMCPY_ASYNC` |
| Native worker configuration | 1013 | 157 `_local_scalar_dense`, 60 `MEMCPY_ASYNC` |

Both run on a nondefault stream. Traces include matmul, activation backward,
reductions and optimizer updates on Ascend hardware. Standalone `msprof` recorded
7015 operators: 6143 AI_VECTOR_CORE, 788 MIX_AIV and 84 AI_CORE. No CPU-fallback
events or corresponding launch diagnostics were observed in these reviewed gates.
Host metadata, scalar scheduling/validation, comparison and serialization copies
are expected. Kernel presence and these traces do not prove absence of every
possible fallback in untested configurations. Profiling here is placement evidence,
not a speedup measurement.

## CUDA and migration boundary

The local aarch64 stack is Torch2.10.0+cu128 with CUDA Toolkit12.8.1. CPU and CUDA
native libraries built in separate directories. The CUDA-linked stack passed
38 directed CPU/CLI/qualification tests, including explicit unavailable-device
failure. Its installed Python/native/C++ consumption passed ten checks after the
nested-venv dependency fix. A separate CMake/loader check passed with real CUDA
libraries and no Python/stub dependencies. CUDA host tests cover FP32/FP64 on CPU.
**No NVIDIA GPU is present: CUDA kernel execution, GPU FP32/FP64 equivalence and
all x86_64 builds remain target-machine work.**

A migration fixture contains independent CPU mixed-TimedDAG inputs, complete
three-step trajectory and a CPU continuation/optimizer checkpoint. It passed CPU
comparison and CPU-to-native-NPU restore. The source export and migration packet
retain hashes and explicit device commands; see `artifacts/accelerator-migration`.
Run the one-fixture check first, then the complete configuration suite and installed
consumers on each selected host/framework/driver combination.

## Provenance and retained failures

Final gates use clean source. Builds originate from frozen `native-dev03` development
source; all five C++ content hashes exactly equal `10cd630`. Binary hashes are
retained and the full CPU gate validates reused artifacts. Python/native CANN suites
record their runtime stack and adapter build identity. `bf4ac1c` changes only the
consumer verifier to retain the selected Torch distribution inside nested venvs;
its CPU and CUDA installed-consumer checks passed on clean source.

Earlier failures remain failed: incomplete self-loop fixture, invalid periodic
phases, standalone generic RNG hook, profiler list-format parsing and scratch
output outside the result directory, unsupported NPU CSR multiply, missing pytest
in the private CUDA environment, and nested-venv selection of base CPU Torch.
Corrected runs have new source/run identities. No tolerance was enlarged to mask
a discrepancy. No experiment data/head/loss was moved into the reusable library.
