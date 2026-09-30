# Current handoff

Updated 2026-09-30T07:31:21.158919+00:00. **ACTIVE — user resumed execution; tested commits may be pushed.**
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

## Current completed window increment

Committed/pushed4e45072 separates device advance from optional CPU exports/journals.
Clean build-device-window-clean01, device-window-components-clean01 and
 device-window-profile-clean01 all PASSED exit0. Source window-clean01/build
 device-window-clean01, core runtime-guard-npu-clean01. Four CPU CTests/loader,
all16 component cells,640 prior content windows plus384 new window cases.
Lean trace:192 windows +one expected refusal,26378 AIV/360 AI Core,193 model
submits/boundary waits, no journal/AiCPU/fallback. Not throughput/training.
Evidence [device-window-20260930](evidence/device-window-20260930.md).
Earlier window-dev01 directed results remain retained, without replacing clean evidence.

## Active packed-sum development

Uncommitted new PackedSum and Ascend C metadata/vector kernels. Scalar device path
remains selectable; ContentLimits.vectorized_aggregate defaults true in this draft.
Payload width tiles256, stable message-order reduction, preflight before parallel
writes, no arithmetic on absent buffer rows. sum_check.cpp covers72 scalar/vector
cases,144 input-changing replays, nine malformed/sticky refusals; widths1..2048.
build-device-sum-dev01 FAILED: Muls template inferred __gm__ float for scale.
Preserved source/log; fix loads scale into a local float before the vector API.
Added explicit inference/autograd rejection. RUNNING build-device-sum-dev02 on frozen sum-dev02, matching
runtime-guard-npu-clean01 core; jobs2/900s. After build run sum/content/window directed
gates (900s) and sum placement (480s), queue120s. On failure inspect source/log first.
Minimal source/config/raw results only; no tracking infrastructure or new full-size job.

## Newly completed immutable qualification

Runtime guard **8244f13** and Read **06db0c2**, both pushed; evidence
[runtime-read-20260930](evidence/runtime-read-20260930.md).
All nine qualification jobs terminal PASSED exit0:

- build-runtime-guard-cpu-clean01: six CPU CTests.
- build-runtime-guard-npu-clean01: standalone core build.
- build-device-failure-clean01: four CPU CTests and loader.
- runtime-guard-lifecycle-clean01: eight fresh NPU processes.
- device-failure-gates-clean01: five recoverable injected faults and quarantined
  finite worker's expected exit86; no physical hung-device/reset claim.
- device-runtime-components-clean01: control FP32 and numerical FP32/FP16.
- build-device-read-clean01: four CPU CTests/standalone loader.
- device-read-components-clean01: all15 cells;640 content windows,3252 events,
  2012 emissions,66 multi-time windows including38 state-Read batches.
- device-read-profile-clean01:101630 AIV +716 AI Core tasks,914 state-Read tasks,
  644 model submits +644 boundary waits;30236 other stream sync API calls from
  setup/boundary/diagnostic work. No AiCPU/fallback. Not throughput.

Sources runtime-guard-clean01/read-clean01, builds with matching names above.
Read can prepare observe-all/no-clear state sequences; device-ready restricts
state-causal regions to one complete frame while other regions keep prefixes.
Runtime raw owners retain leases until resources drain/destroy. Failed programs
reject replay; quarantine refuses finalization and new work until worker exit.

Earlier immutable evidence remains scoped to its exact source:
- [online-greedy](evidence/online-greedy-20260930.md),2038d88:8849 CPU tests,
  four CTests and six native/Python NPU FP32 fixtures with VJP/optimizer/continuation.
  Host scheduling + NPU tensors, not resident training.
- [device-control](evidence/device-control-20260930.md),5c5b582;
  [packing](evidence/device-packing-20260930.md),eff5945;
  [selection](evidence/device-selection-20260930.md),6220011.
- [content-loop](evidence/content-loop-20260930.md),4d2f09e:identity-Full FP32.
- [selected-Full](evidence/selected-full-20260930.md),5bf61e3:all14 cells,
  24 Full cases,160 windows/788 events/488 emissions,20 multi-time windows;
  410 AIV +16 AI Core tasks. Local Full scratch budget is not whole-model planning.

## Retained failures and older work

Original sources/logs remain; never relabel failures from later successes:
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
must never be modified. Last disk check:321GB data,29GB root free.

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
