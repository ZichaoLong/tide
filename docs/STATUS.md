# Current handoff

Updated 2026-09-30T09:48:03.980178+00:00. **ACTIVE — user resumed execution; tested commits may be pushed.**
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
per local tick with explicit max_repeat_ticks65536/work refusal8. Sum, state and Read have vector/scalar options; linear-v1 and norm-fp32-v1 Read
are now implemented and clean-qualified below.

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

## Current public Norm32 increment

Implementation **0e66d89** committed/pushed. norm-fp32-v1 explicitly converts the
visible value before reduction and returns FP32 for FP16/32/64 payloads. Normal
conversion/norm VJP, connected-zero at zero, unused linear Read weights disconnected;
profile belongs to graph identity. norm-fp64-v1 stays unchanged and rejected on NPU.
Development304 CPU checks,8 CTests,18 Python/18 native NPU fixtures and standalone
6 combinations/18 updates passed; candidate-only profile604 kernels passed.

Clean source norm32-clean01 at0e66d89:
- norm32-cpu-clean01 PASSED full verify.py CPU regression:8901 tests +8 CTests.
  4200s total/jobs2; build norm32-cpu-clean01, output runs/.../verified.
- build-norm32-npu-clean01 and build-norm32-native-npu-clean01 PASSED.
- norm32-python-npu-clean01 and norm32-native-npu-clean01 PASSED18 cases each.
- norm32-standalone-npu-clean01 PASSED6 schedule/Read combinations,18 updates,
  FP16/FP32 analytic norm tests. Standalone loader resolves in the authorized module.
- norm32-native-profile-clean01 PASSED604 kernels,12 Norm tasks,97 host scalar events,
 72 memcpy events; no AiCPU/fallback. Host-scheduled training, not resident training.
[Immutable public Norm32 evidence](evidence/norm32-20260930.md).
No pending permission or pause.

## Current device Norm32/vector Read increment

Implementation **8a735ef** committed/pushed; clean qualification running. Same explicit FP32 norm profile inside
the online device loop. ContentLimits.vectorized_read defaults true; scalar Read
remains selectable. Metadata preflight checks exact int64 clocks/work limits,
independent owner/width tiles perform vector state preparation and dot/norm partials,
then device reduction/sqrt completes scores. Actual time order is retained per tile.
The existing selector checks finite scores before transaction commit. FP32 inference only.

Development PASSED on frozen read-vector-dev02, matching core norm32-npu-dev02:
- build-device-read-vector-dev02:four CPU CTests/loader, build device-read-vector-dev02.
- device-read-vector-gates-dev02:all20 cells,530 norm windows including width2048,
  analytic zero/stable ties,overflow/fp64 refusals; clock/Add/content/window regressions.
- device-read-vector-profile-dev02:bounded norm32 trace PASSED, raw placement retained.
Earlier read-vector-dev01 regression passed clock/Add/content/window; its Norm fixture
was not run because it shared the already-diagnosed long-tick test mistake below.

Clean source read-vector-clean01 at8a735ef: build-device-read-vector-clean01 PASSED
(four CPU CTests/loader; matching norm32-npu-clean01 core),
device-read-vector-components-clean01 PASSED all20 cells,
device-read-vector-profile-clean01 PASSED separate norm32 placement profile.
[Immutable Read evidence](evidence/device-read-20260930.md): 96173 AIV +1083 AI Core;
410 vector Read/410 reductions, no AiCPU/fallback.
This does not qualify throughput,attention,FP16,peer progression or resident training.

Uncommitted next LH increment must be excluded from the Read commit:
packed_lh_full.{h,cpp},lh_full_check.cpp, activation/normalization methods in
cann_program.{h,cpp}, int-array lifetime in cann_api.{h,cpp}, target-kind filtering
in ascendc/tide_full_plan.cpp and packed_full.cpp. Wired into ContentFlow now, with local LH reservation deducted from tanh budget.
Grouped selected-only vendor activation/norm, safe sentinel padding and bounded
chunks;32 component cases/nine profiles plus96 complete windows planned.
build-device-lh-full-dev01 PASSED four CPU CTests/loader from frozen lh-full-dev01,
matching norm32-npu-clean01 core. NEXT device-lh-full-gates-dev01: lh-full/full/norm32/failure,
FAILED first lh-full component numerical equality (exit2, not timeout).
Original logs retained. diagnostic01 isolated width7 LayerNorm differences about1e-5.
Bounded mode probe completed: modes0/1 same mismatch; mode2 rejects161002.
Independent FP64 shows CPU FP32 itself differs above fixed tolerance for low variance.
Component-only conditioning budget/independent FP64 check added; complete graph
comparison stays strict. device-lh-full-precision-dev02 RUNNING diagnostic-only relink,
300s/queue120s; after success rebuild frozen source and run full gates/profile.
This is unverified FP32 broadcast inference; no slot-affine or training claim.

## Retained failures and older work

Original sources/logs remain; never relabel failures from later successes:
- device-norm-gates-dev01 FAILED120s and device-norm-profile-dev01 FAILED180s.
  Gdb found CPU mul: fixture Add owners at cut2^55 lacked near-cut clocks, causing
  ~2^55 literal ticks. Fixed fixture and added CPU-work guard; no runtime shortcut
  or timeout relaxation. diagnosis.json retained; diagnostic profile not accepted.
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
