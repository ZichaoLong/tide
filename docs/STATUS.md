# Current handoff

Updated 2026-09-30T11:01:24.452225+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository /home/zlong/llm/graph-execution-foundation, real path
/var/tmp/zlong-graph-execution-foundation/repository; branch graph-execution-foundation.
No subagents. Reference repositories and ObsidianVault are read-only. Preserve the
uncommitted older accelerator_scale/flow consumer work listed below.

## Authoritative scope

[execution-flows.md](execution-flows.md) is the execution contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Latest user alignment takes priority over run-ml-experiments:
minimal source/input/config/environment identity, raw results/failures, synchronized
timing and bounded resources/stops. Reuse records; Trackio must not block implementation.
Training acceptance means forward/backward/VJP/optimizer/continuation and complete
throughput, not convergence. No pending permission or pause.

Every candidate independently consumes input/state/parameters; no numerical CPU
route prepass, whole-window potential expansion or fixture-specific schedule.
General online node-time greedy prefill accepts each family's legal topology/input,
including PDG positive-delay feedback. Device residence includes online decisions.

Priorities: finite-profile correctness/failure gates and bounded profiling; actual
modules/attention/KV and complete safe chunking; public five presets and matrix,
peer progression and resident backward/optimizer; then representative/full-size
comparisons. PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill
× inference/complete training. CPU, mixed A/B/C, resident; FP32 main, FP16 separate,
CPU FP64 oracle. Three fresh processes before performance recommendations.
CUDA real execution remains target-machine pending. **F1–F7 are not complete.**
The raw device flow is still single-device FP32 inference.

## Latest qualified increments (committed and pushed)

- Device periodic clocks, implementation49ff108: four CPU CTests, all19 device
  cells,480 mapped windows/18 multi-phase windows/12 refusals;95457 AIV+746 AI Core.
  [Evidence](evidence/device-clock-20260930.md).
- Public norm-fp32-v1, implementation0e66d89:8901 full CPU tests/eight CTests,
  Python18/native18 NPU fixtures; standalone6 combinations/18 updates. Profile604
  kernels,97 host scalar/72 memcpy events. Host-scheduled tensor training, not
  resident training. [Evidence](evidence/norm32-20260930.md), evidence commit543de2b.
- Device vector Read, implementation8a735ef: all20 cells,530 norm windows;
  scalar/vector,content/old/proposal,width2048,int64 clocks,overflow/FP64 refusals.
  Trace96173 AIV+1083 AI Core,no AiCPU/fallback.
  [Evidence](evidence/device-read-20260930.md), evidence commita117944.
- Selected LH Full, implementationcd03ca8: nine activation/norm profiles,40
  component cases/96 strict complete windows,all21 cells. Trace27763 AIV,no AiCPU.
  34/536 low-variance component rows miss ordinary CPU/NPU tolerance; both CPU and
  NPU pass an independent FP64 conditioning budget. CPU maxabs1.31656e-5,NPU1.51344e-5.
  Full graph comparisons and well-conditioned rows retain original tolerances.
  [Evidence](evidence/device-lh-full-20260930.md), evidence commitd336c37.

All LH clean jobs PASSED: build-device-lh-full-clean01,
device-lh-full-components-clean01,device-lh-full-profile-clean01. Frozen source
lh-full-clean01 atcd03ca8; build device-lh-full-clean01/core norm32-npu-clean01.
These finite qualifications do not certify throughput,FP16 flow,slot-affine,
attention/KV,the full public matrix,peer progression or resident training.

## Current native origin-order correction

Native Aggregate used unstable std::sort for equal InputOrigin keys. It now uses
stable_sort, matching Python and retaining distinct logical sources. A32-edge
regression in cpp/test/custom_aggregate.cpp fails against old0e66d89 core (exit2,
exact expected message) and passes against corrected core, FP32/FP64.
origins-oldcore-regression01 preserves old-library/test-object hashes and failure.

Development core builds build-origins-cpu-dev01 and build-origins-npu-dev01 PASSED,
frozen origins-core-dev01. origins-cpu-gates-dev01 PASSED eight CTests, two standalone
Aggregate gates and362 focused source-origin/domain/ports/Settle Python checks.
Correction832a881 committed/pushed. Clean CPU/NPU builds from origins-core-clean01
at832a881 PASSED. origins-cpu-gates-clean01 PASSED eight CTests, two standalone
Aggregate gates and362 focused tests; origins-standalone-npu-clean01 PASSED.
[Narrow stable-order evidence](evidence/native-origin-order-20260930.md) records
these gates separately from full regression.

## Current InputOrigin device increment committed/pushed

Implementation1be3619: static origin table; int64 clock preflight/refusal10;
stable metadata order for scalar/vector Aggregate. Physical identities,scales,
contribution rows,pending and emitted messages stay physical. Exported source views
apply projection only at observation. Logical alias collision remains2.
Nonempty emit_phases is explicitly refused until phase routing is implemented.

Development build PASSED. device-origins-gates-dev01 PASSED all22 cells including
24 ordering/tie/cancellation cases,128 continuation windows,8 refusals.
device-origins-profile-dev01 PASSED27256 AIV tasks,no AiCPU/fallback.
Frozen origins-dev01 includes the recorded patch; these are development results.

Clean build-device-origins-clean01,device-origins-gates-clean01 and
device-origins-profile-clean01 PASSED from frozen origins-clean01 at1be3619,
matching clean core origins-npu-clean01. All22 cells passed; origin gate24 ordering
cases,128 windows,8 refusals. Profile27256 AIV tasks,no AiCPU/fallback.
[Device-origin evidence](evidence/device-origins-20260930.md) records the clean qualification.

## Packed slot emission qualified on6445121

Committed/pushed implementation6445121 supports device-planned time-phase presence,
selected affine projection chunks and batched physical delivery. Unscaled slot
journals preserve zero-scale diagnostics; Full auxiliary values remain separate.
Scope: single-device FP32 HARD inference. Old broadcast component remains available.

Clean build-device-emission-clean01,device-emission-gates-clean01 and
profile-clean01 PASSED from frozen emission-clean01 at6445121,matching core
origins-npu-clean01. Four CPU CTests,all23 cells;16 emission components,66 complete
windows,6 refusals. Profile14375 AIV+176 AI Core,no AiCPU/fallback. Source/raw hashes
and limits are in [emission evidence](evidence/device-emission-20260930.md).

Development build-dev01 failed on test helper ADL ambiguity; two dependent jobs
failed without acquiring a device. gates/profile-dev04 failed at bool tensor
initializer construction. Later tests use ordinary bool ones/fill. Pre-run review
corrected illegal identity phases and placed poison below public finite validation.
Original snapshots/logs remain; no failure relabeled and no tolerance/formula change.
Incremental development builds recorded byte-identical production source/relinked
tests. The final independent clean build above supersedes that provenance for qualification.

## SwiGLU Full in development

Uncommitted packed_swiglu_full.{h,cpp},swiglu_check.cpp and flow/profile/build
integration implement selected-only batched SwiGLU with the original residual.
Static parameter table packs actual SwiGLU owners; device planner chooses chunks.
HARD FP32 inference only; no new backward or throughput claim.
build-device-swiglu-dev01 RUNNING from frozen swiglu-dev01 using core origins-npu-clean01,
jobs2/bound1800s. Then all24 device cells900s and separate swiglu profile480s,
one NPU/queue120s; retain failures. This job is independent of immutable emission
qualification. Never modify either active snapshot/build.

## Failures retained and prior work

- LH dev01 fixed-tolerance component failed. diagnostic01 isolated LayerNorm;
  modes0/1 unchanged,mode2 refused161002. precision-dev02 used independent FP64
  formulas/declared component budget; complete graph equality unchanged. Original
  logs/snapshots and diagnosis.json remain; no failure is relabeled.
- device-norm-gates-dev01/profile-dev01 timed out120/180s: malformed test initial
  clocks caused CPU oracle ~2^55 literal Add ticks (gdb confirmed CPU mul). Fixed
  fixture and added work-bound guard; no power shortcut or timeout increase.
- Older closure/build/runtime failures remain in durable records and immutable
  evidence. In particular raw control does not use the unsupported nested captured
  NPUGraph bridge; Tensor scatter_reduce host fallback remains rejected.

Preserve dirty tools/accelerator_scale/flow_*,bounded/resident/peer files,their CMake
and build script,scripts/benchmark_execution_flow.py,verify_execution_flows.py,
tests/test_flow_semantics.py. Limited DAG/rank-aligned consumers, not revised general
online delivery. flow-dev05 CPU24 passed; NPU18 FP32 passed before FP16 resident Add
gradient failure. dev06 builds passed, no gates. Peer CPU8/NPU16 passed separately.

Historical tide-execution-flows-historical-cpu-attention-01.service remains SIGSTOP.
TASK/runs/historical-cpu-attention-01/pause.json overrides its running status.
It holds host memory and TASK/timing.lock. Do not blindly resume/stop it. Resolve
its interrupted timing and lock deliberately before formal timing, preserving logs.
Historical mixed Add complete training CPU78.793172/NPU4 47.932888 ms/token means
NPU throughput1.6438× faster. This does not qualify the new resident path.
Historical Attention has no valid CPU complete-training ratio.

## Environment and durable operation

TASK=/mi/data2T/zlong/tide-execution-flows. Each job: TASK/runs/NAME/{status.json,task.log},
repository symlink artifacts/execution-flows-NAME. Unit tide-execution-flows-NAME.service,
background.slice. Frozen snapshots/builds must not be overwritten or modified.
Last disk check307GB data/27GB root free; recheck before large writes.

Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized public /opt stack overrides dated guide defaults. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0,CPU/BLAS threads1,build jobs2. Retain module PYTHONPATH
and prepend snapshot/python. SoC Ascend910_9392;16 chips64GiB. Cooperative leases
choose physical devices and remap to logical npu:0; no hardcoded placement.

```
python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT [--commit SHA] [--npu --max-wait 120] -- timeout --signal=TERM --kill-after=10s 900s '{python}' scripts/COMMAND ...
```

Placeholders {python},{base},{source},{out}. Existing snapshot reused read-only;
qualification requires --commit. norm32_after_core.py is a reusable bounded600s
build-dependency wrapper; do not reserve an NPU while waiting for a build.
origins_cpu_gates.py takes a build path,checks source hash,then runs the focused CPU
matrix. Handoff writes use scripts/durable_records.py atomic fsync/read-back.
Re-entry: git status --short --branch;python scripts/status.py;inspect real terminal
records before continuing. Submitted/running never means passed.
