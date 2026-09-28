# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch: graph-execution-foundation.

## Delivery and source boundary

Python/LibTorch × CUDA/Ascend NPU implementation and local acceptance are complete.
The reusable GraphConfig → GraphRuntime → Session contract, independent reference
schedules, CPU FP32/FP64, graph semantics and checkpoint versions are preserved.
Experiments continue to own data, heads, losses and training orchestration.
Accelerator scope is single-device eager FP32. No subagents or pushes.

Implementation commits: adeb819 (Python qualification), 10cd630 (native runtimes,
configuration suites, profiling and consumption), bf4ac1c (consumer verifier keeps
the selected Torch in a nested venv). That last correction changes no runtime,
Python package or C++ execution code. The subsequent evidence commit changes only
documentation and the support matrix; see Git HEAD for its exact identity.

Reviewed results: [accelerator report](evidence/accelerators-20260928.md),
[structured manifests/hashes](evidence/accelerators-20260928.json),
[build and target-machine commands](accelerators.md).

## Terminal acceptance

All 22 final qualification jobs passed with exit 0 and inactive systemd units.
No qualification jobs remain active. Exact names, commands, source identities,
start/finish times and report hashes are in the structured evidence.

- cpu-a2 at clean 10cd630: 8636 CPU tests (1618.89 s), all 22 complex topology
  cells and ten installed Python/native/C++ checks.
- python/native-cann850/851/852/900-a2: 39 Python and 43 native supported cases
  per stack, totaling 328 positive gates; four explicit unsupported CSR gates.
  Full observables/VJPs, chunks, three AdamW updates, fresh-process resume and
  NPU-to-CPU checkpoint handoff pass at unchanged FP32 tolerances.
- sdk-a2: independent standalone C++ NPU ring/diamond, backward, owner sharing,
  three AdamW steps, nondefault worker streams and NPU/CPU checkpoint loading.
  owners-a2 separately covers SGD momentum/AdamW alias/None/zero-gradient slots.
- installed-python-npu-a2: nine checks. installed-cpp-npu-a2: independent CMake
  consumer and loader closure without Python/stub dependencies.
- python/native-profile-a2: 1295/1013 accelerator kernels; standalone msprof:
  7015 operators. Reviewed traces/logs show no CPU-fallback event or diagnostic;
  this is finite placement evidence, without performance or exhaustive claims.
- cuda-host-a2: 38 directed tests on CPU in the CUDA-linked stack.
  installed-cpp-cuda-host-a2 passed loader/CPU checks. Clean bf4ac1c installed
  consumers passed ten checks each on CPU and the CUDA-linked CPU stack.
- migration-golden-a2 and migration-check-cpu/npu-a3: independent CPU mixed
  TimedDAG three-step trajectory and CPU-to-native-NPU checkpoint restore pass.

Task root: /mi/data2T/zlong/tide-accelerator. Each run is linked as
artifacts/accelerator-NAME, with status.json, task.log and gate/result.json
(or profile/result.json); NPU queue.json records physical placement. Units are
tide-accelerator-NAME.service in background.slice, with two build workers and
single-thread CPU pools. All five native builds retain their development-origin
identity and binary hashes; their C++ content exactly equals committed 10cd630.

Earlier failures remain failed and retained: self-loop/periodic fixtures,
standalone generic RNG hook, profiler list parsing/scratch output, unsupported NPU
CSR, missing CUDA pytest and nested-venv selection of CPU Torch. Corrected runs
have separate source/run identities. No tolerance was enlarged.

## Environments and limits

Host aarch64, Ascend910_9392 (A3), public driver25.3.rc1; NVIDIA GPU absent.
Public modules ascend/dev-workspace-8.5.0, -8.5.1 and -8.5.2 use Torch/TorchNPU2.9;
ascend/dev-workspace-9.0.0 uses 2.10. CANN8.5.1/8.5.2 are exact-site empirical
results, not an expansion of official compatibility claims. No new CANN install
was needed. Runtime driver libraries remain under the existing public setup.

Standalone module libtorch-npu/2.10.0-cann9.0.0 is qualified only for that tuple.
Never load its library into Python's TorchNPU wheel process. Private CUDA module
is torch-cuda/2.10.0-cu128 via module use ~/privatemodules; the toolkit is 12.8.1.
CUDA toolchains remain under /mi/data2T/zlong, public NPU stacks under /opt.

NPU FP64, norm-fp64-v1 Read and optional CSR fiber pooling are unsupported;
public adapters reject explicitly. Standalone C++ callers must avoid CSR.
Native TIDENCK1 stores named weights/optimizer slots, not graph continuation.
CUDA hardware execution/FP32/FP64 parity and every x86_64 build remain unverified.
AMP/low precision, distributed jobs, compilation/fused kernels and performance
qualification remain separate extensions.

## Migration handoff and next target-machine action

Delivery artifact location: artifacts/accelerator-migration (outside Git, on
/mi/data2T). Its README.md contains explicit CPU/CUDA/NPU commands;
packet-manifest.json records the clean export commit and fixture hashes;
SHA256SUMS covers the transferable source, fixtures and helper. The adjacent
archive/checksum is for copying to another host. No toolchains or binaries are
part of the source packet. Delivery verification is recorded in the artifact's
status.json; it checks relocation against the independent CPU golden fixture.

For a new machine activate a matching toolchain, check the archive/content hashes,
then use Python3.11 and distinct build/output directories. From the exported source:

```sh
python scripts/build.py --backend cpu --build-dir build/cpu --jobs 2
python scripts/qualify_library.py --reuse-build --build-dir build/cpu --output-dir artifacts/cpu-001
python scripts/build.py --backend cuda --build-dir build/cuda --jobs 2
python scripts/qualify_accelerator.py --device cuda:0 --implementation python --output-dir artifacts/cuda-python-001
python scripts/qualify_accelerator.py --device cuda:0 --implementation native --native-library build/cuda --output-dir artifacts/cuda-native-001
```

Run the packet's golden check, standalone C++/owner/profile/installed-consumer
commands as well. Qualify CUDA FP64 separately using --dtype float64. For NPU,
follow the two-runtime recipe in accelerators.md. Record each exact host/device,
Torch/vendor/driver/ABI tuple and keep target failures; update support claims
only after its required gates pass. No additional local full regression is
needed for this documentation-only evidence update.
