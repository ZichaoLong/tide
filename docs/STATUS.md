# Current handoff

Updated 2026-09-30T08:51:25.997731+00:00. **ACTIVE — user resumed execution; tested commits may be pushed.**
Repository `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`, branch `graph-execution-foundation`.
No subagents. Reference repositories and ObsidianVault remain read-only.
Preserve unrelated dirty work. Latest alignment explicitly takes priority over
run-ml-experiments: retain minimal useful source/input/config/environment identity,
raw results/failures, declared synchronized timing and bounded jobs. Reuse existing
records; Trackio and extra experiment infrastructure must not block implementation.
Training acceptance is infrastructure correctness/performance, not convergence.

## Contract and priorities

[execution-flows.md](execution-flows.md) is authoritative; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Each candidate independently consumes inputs and initial state;
no numerical CPU route prepass, whole-window potential expansion or fixture shortcut.
General online node-time greedy prefill accepts each family's legal topology/input,
including positive-delay PDG feedback. Device residence includes decisions/progression.

1. Close finite-profile correctness/failure gates, then remove mandatory diagnostic
   exports and measure bounded small/medium behavior to guide packed computation.
2. Expand real module contracts/attention, vectorized computation, FP16 and complete
   model/KV/activation/communication memory planning with safe chunking.
3. Complete public five presets and graph/language matrix, peer progression and
   actual backward/VJP/optimizer; then representative/full-size comparisons.

PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. CPU, mixed A/B/C, resident; small/medium all, full-size
CPU/screened mixed/resident. FP32 main, FP16 separate, CPU FP64 oracle; three fresh
processes before recommendations. CUDA actual execution remains target-machine pending.
F1–F7 are NOT complete. The current raw device flow is single-device FP32 inference.

## Latest completed device increment

Periodic state-clock implementation **49ff108** is committed/pushed. Clean
build-device-clock-clean01, device-clock-components-clean01 and
device-clock-profile-clean01 all PASSED exit0. Source clock-clean01/build
 device-clock-clean01, matching core runtime-guard-npu-clean01. Four CPU CTests,
standalone loader and all19 cells passed:480 mapped windows,18 multi-phase windows,
12 phase refusals, plus560 Add/640 content/384 window regressions. Trace95457 AIV
and746 AI Core tasks, no AiCPU/fallback. This is FP32 inference placement/correctness,
not throughput or resident training. [Evidence](evidence/device-clock-20260930.md).

Identity/EMA/Add converts int64 global timestamps to local ticks on device; persistent
clocks and message/history/Read coordinates stay global. Invalid event phase refuses
transaction9; malformed persistent phase rejects restore. Add repeats multiplication
per local tick with explicit max_repeat_ticks65536/work refusal8. Sum and state have
vector/scalar options; current Read is scalar, linear-v1 only.

Earlier immutable evidence remains scoped to the tested source:
- [online greedy](evidence/online-greedy-20260930.md),2038d88:8849 CPU tests,
  four CTests,six native/Python NPU FP32 fixtures incl.VJP/update/continuation;
  host scheduling with NPU tensors, not resident training.
- [control](evidence/device-control-20260930.md),5c5b582;
  [packing](evidence/device-packing-20260930.md),eff5945;
  [selection](evidence/device-selection-20260930.md),6220011.
- [content](evidence/content-loop-20260930.md),4d2f09e;
  [selected Full](evidence/selected-full-20260930.md),5bf61e3.
- [runtime/Read](evidence/runtime-read-20260930.md),8244f13/06db0c2:
  clean failure ownership/lifecycle and old/proposal Read gates.
- [window](evidence/device-window-20260930.md),4e45072:advance separate from exports.
- [sum](evidence/device-sum-20260930.md),fbc6652:packed vector sum.
- [Add](evidence/device-add-20260930.md),bbe66e2:literal recurrence/vector state.

## Current implementation and next commands

Uncommitted Norm32 core: cpp/src/read.cpp, python/tidegraph/readout.py,
 tests/test_norm_precision.py, scripts/accelerator_cases.py, docs/read-programs.md.
Explicit norm-fp32-v1 converts before reduction/returns FP32, including FP16/FP64
payloads; norm-fp64-v1 stays unchanged. Normal conversion/norm VJP, connected-zero
at zero, unused linear Read weights disconnected; profile changes graph identity.

Development gates PASSED exit0:
- norm32-cpu-dev01:304 Python/native CPU norm/Read tests.
- norm32-python-dev01:30 Python tests,22 native cases deselected.
- norm32-cpu-ctests-dev01:all6 CPU CTests.
- norm32-python-npu-dev01:18 cases,three families × three Read modes × two schedules,
  including independent CPU comparisons,VJPs,three optimizer updates and continuation.
- build-norm32-npu-dev01:standalone NPU core; build-norm32-native-npu-dev01:bindings.

Snapshots norm32-dev01 (base d9ff8fd,CPU) and norm32-dev02 (base49ff108,NPU) have
identical cpp hashes; build dirs norm32-cpu-dev01/norm32-npu-dev01/
norm32-native-npu-dev01. They include recorded dirty overlays. These are development
gates; commit implementation then qualify a clean exact commit separately.

NEXT: commit clock evidence/this refreshed status, preserve Norm32/older consumer WIP.
Launch norm32-native-npu-dev01 from frozen norm32-dev02 with scripts/qualify_accelerator.py
 --device npu:0 --implementation native --dtype float32
 --native-library TASK/builds/norm32-native-npu-dev01 --output-dir OUT/verified
 and all18 --case norm32-{pdg,timed-dag,settle}-{content,old,proposal}-{streaming,greedy}.
Queue120s/work900s. Candidate-only profile via scripts/profile_accelerator.py,
 case norm32-timed-dag-proposal-greedy, same native build, work480s.
Add independent standalone C++ Norm32 gates. After core gates commit, perform clean
CPU regression/NPU validation, then integrate norm-fp32-v1 into the raw ContentFlow.
Raw-device Norm32 is NOT implemented yet; old runtime-guard core is incompatible
with changed cpp sources: use matching new norm32 standalone core for later raw work.
Continue LH Full/attention/KV, packed Read/transport,FP16,memory/public matrix,
peer progression and resident backward/VJP/optimizer. No pause is active.

## Retained failures and older work

Original sources/logs remain; never relabel failures from later successes:
- device-clock-gates-dev01 rejected fixture identity boundaries with non-global clocks
  before device execution. Fixture corrected; original source/build/log retained.
- build-device-sum-dev01 failed Muls scalar deduction; local scalar load fixed it.
- device-broadcast-gates-dev01 SIGSEGV at static NPU finalizer; gdb debug01 identified
  main-thread TLS teardown. Fixed07fcae4; later clean runtime qualification above.
- device-queue-profile-dev01 lacked marker, msprof Resource temporarily unavailable.
  A later diagnostic did not erase that failure; require exit+marker+termination.
- control-dev05 nested captured NPUGraph RIExecuteAsync on a bound stream107000;
  unsupported bridge removed. Raw control uses a global label target list.
- build-device-content-dev01 failed generated kernel headers on local alias I;
  explicit int64_t fixed it. Ascend C uses Unix Makefiles/PIC/underscore targets.
- Tensor scatter_reduce closure host-falls-back on this stack and rejects NPU.
  Earlier eager exact-int64 sort uses AiCPU; raw Ascend C traces are separate.

Preserve uncommitted tools/accelerator_scale/flow_*, bounded/resident/peer files,
client CMake/build script, benchmark_execution_flow.py, verify_execution_flows.py,
tests/test_flow_semantics.py. These are limited DAG/rank-aligned consumers, not
revised general-online delivery. flow-dev05 CPU24 passed; NPU first18 FP32 passed
then first FP16 resident Add gradient failed. dev06 builds passed; no dev06 gates.
Reusable peer CPU8/NPU16 gates passed separately with TASK_QUEUE_ENABLE=0.

Historical `tide-execution-flows-historical-cpu-attention-01.service` stays SIGSTOP.
`TASK/runs/historical-cpu-attention-01/pause.json` is authoritative despite running
status.json. Holds host memory and TASK/timing.lock. Do not blindly resume it.
Before formal timing, resolve its invalid interrupted timing and lock deliberately,
preserving records. Do not stop other users' or unidentified tasks.

Wide packet:480 body nodes,2208 body edges,two identity boundaries,D2048/B512/T12/V50304.
Add9,468,053,696; Attention17,521,117,376 parameters. Historical Attention17,269,426,339
is different. Historical mixed FP32 ms/token: Add inference CPU7.387522/NPU2 17.858098;
Attention CPU19.531847/NPU4 56.012343; Add complete training CPU78.793172/NPU4 47.932888
(NPU throughput1.6438× faster); Attention NPU9 128.275286, no valid CPU training ratio.
These do not qualify the new resident path.

## Environment and durable operation

TASK=`/mi/data2T/zlong/tide-execution-flows`; each job logs/status in
TASK/runs/NAME/{task.log,status.json}, repo symlink artifacts/execution-flows-NAME.
Units `tide-execution-flows-NAME.service`, detached background.slice. Frozen sources
must never be modified. Last disk check:311GB data,27GB root free.

Module `libtorch-npu/2.10.0-cann9.0.0`, Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized public /opt stack overrides dated personal-stack guide defaults.
TASK_QUEUE_ENABLE=0, TORCH_DEVICE_BACKEND_AUTOLOAD=0, ATen/BLAS threads1, build jobs2.
Retain module PYTHONPATH and prepend source/python. SoC Ascend910_9392,16 chips64GiB;
cooperative leases remap physical devices to logical npu:0. No hardcoded placement.
Trackio interpreter `/home/zlong/venvs/trackio/bin/python`, project tide-execution-flows.

```
python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT [--commit SHA] [--npu --max-wait 120] -- timeout --signal=TERM --kill-after=10s 900s '{python}' scripts/COMMAND ...
```

Placeholders {python}/{base}/{source}/{out}; exact source identity/hash stored beside
snapshot. Existing snapshot reused read-only. Qualification requires --commit.
Handoff writes use scripts/durable_records.py atomic fsynced replacement/read-back.
Re-entry: git status --short --branch; python scripts/status.py; inspect actual terminal
records, then continue the authorized contract. Submitted/running never means passed.
