# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch: graph-execution-foundation.

## Authorized work and implementation boundary

Python/LibTorch × CUDA/Ascend NPU for the reusable Tide library. Single-device
eager FP32 accelerator baseline; CPU FP32/FP64 remains required. Preserve
independent schedules, graph semantics, public Session and standalone C++ use.
CUDA live execution belongs to the target machine. Matched local CANN testing
and installations are authorized. No subagents or pushes.

A1 is adeb819. A2 is ready for its implementation commit: isolated native
backends, separate standalone/Python NPU lifecycle, checkpoint/optimizer device
placement, worker streams, CMake exports, finite suites, profiling and consumers.
NPU FP64/FP64 Read and optional CSR pooling are explicit unsupported capabilities.
No semantic/checkpoint version changes or task heads/data enter the library.

Directed evidence (development snapshots, not final release acceptance):
- native-dev03: CPU/CUDA/NPU Python/NPU standalone and Torch2.9 NPU builds passed.
  Current C++ hash matches all five build manifests.
- native-cpu-tests-dev02: 65 passed; follow-up 12 CPU boundary/standalone tests passed.
- native-dev02/dev04/dev05: every supported expanded NPU case passed across
  directed runs (39 Python, 43 native); CSR policy rejection verified separately.
- native-sdk-test-dev04: independent C++ NPU ring/diamond, VJPs, three AdamW
  updates, nondefault worker streams and same-device/CPU checkpoint passed.
- native-owners-dev04: native SGD/AdamW aliases, None/zero gradients and restored
  device slots passed. Installed Python/native (9 checks) and independent CMake
  NPU consumers passed; loader closure has no Python or stubs.
- python/native-profile-dev04: 1295/1013 hardware kernels observed; default
  profiler scratch location polluted the snapshot directory. Fixed to output cwd.
  Host copies/scalar decisions remain observable; no exhaustive no-fallback claim.
- Clean adeb819 Python initial 10 cases passed CANN8.5.0/8.5.1/8.5.2/9.0.0.

Task root: /mi/data2T/zlong/tide-accelerator, with sources/, builds/, runs/.
Each run is exposed by artifacts/accelerator-NAME; status.json, task.log,
queue.json and gate/result.json retain exact source/environment/device evidence.
Historical failures remain: missing self-loop size, invalid periodic fixture,
standalone generic RNG hook, profiler JSON/scratch handling, NPU sparse CSR,
and cuda-host-dev05 missing pytest. Reproducers retained. GPU environment now
has pytest9.1.1; no framework upgrades were performed.

## Next commands and active submissions

Commit A2, freeze exact clean commit as accelerator-a2, then submit:
- cpu-a2: scripts/qualify_library.py --reuse-build using builds/native-cpu-dev03.
- python/native-cann900-a2, python/native-cann850-a2, python/native-cann851-a2,
  python/native-cann852-a2: complete named device suites, one queued NPU each.
- sdk-a2, owners-a2, profiles and installed consumers: exact-commit NPU gates.
- cuda-host-a2: CPU tests in CUDA-linked build and explicit unavailable failures;
  installed CMake/Python consumer CPU checks. No GPU execution claim.
Units are tide-accelerator-NAME, in background.slice; runs/NAME contains exact
commands/log/status, NPU queue assignments and results. Submit only from frozen
source; two build workers/one CPU thread pool. All listed submissions are still
pending, not passing evidence. Inspect terminal results and commit reviewed
support/evidence separately. Keep working until the qualification is terminal.

## Environments

Public CANN8.5.0/8.5.1/8.5.2 + Torch/TorchNPU2.9.0; CANN9.0.0 +2.10.0.
Standalone SDK libtorch-npu/2.10.0-cann9.0.0; private CUDA
 torch-cuda/2.10.0-cu128. Host aarch64, Ascend910_9392; NVIDIA GPU absent.
Previous CPU release: 8621 tests +22 complex cells at ff708a1; follow-up at
aa03a03. Historical results do not qualify A2. See evidence/library-foundation.md.
