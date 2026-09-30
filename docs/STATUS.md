# Current handoff

Updated 2026-09-30T06:43:42.869668+00:00. **ACTIVE — user resumed the execution contract; push each tested commit.**
Repo `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`, branch `graph-execution-foundation`.
No subagents. Reference repos and ObsidianVault remain read-only. The user has
explicitly resumed work toward the agreed contract and authorized pushing future
commits. The previous next-commit pause is revoked. Preserve unrelated dirty work.

## Contract and next action

[execution-flows.md](execution-flows.md) is authoritative; [ROADMAP F1–F7](ROADMAP.md)
is the single backlog. General online node-time greedy prefill accepts every
family's legal topology/input, including positive-delay PDG feedback. No numerical
route prepass, whole-window potential expansion or fixture shortcuts. Residency
includes online decisions and progression, not just tensor storage or fixed capture.

Clean `5bf61e3` qualification PASSED: build-device-full-clean01,
device-components-clean04 (all14 cells), device-full-profile-clean01.
Evidence [selected-full-20260930](evidence/selected-full-20260930.md). All three
jobs are terminal exit0. Source full-clean01/build device-full-clean01 retained.
Runtime-resource guard development PASSED all six jobs on frozen
runtime-guard-dev01 (base5bf61e3 dirty overlay):
- build-runtime-guard-cpu-dev01: CPU core/bindings + six CTests.
- build-runtime-guard-npu-dev01: standalone core build.
- build-device-failure-dev01: component build + four CPU CTests/loader.
- runtime-guard-lifecycle-dev01: eight fresh NPU processes.
- device-failure-gates-dev01: five recoverable fault cases; separate quarantined
  worker retains owners/refuses finalization and exits86 as required.
- device-runtime-components-dev01: control FP32 + numerical FP32/FP16.
Reports/logs/status under TASK/runs/NAME, failure/component results in verified/.
The runtime last-session guard and close poisoning are ready for a separate
implementation commit/push; then clean CPU/NPU core and failure qualification.
No physical hung-kernel/driver-reset recovery claim is made.

Read-mode increment remains UNCOMMITTED and unqualified. build-device-read-dev01
is RUNNING on frozen read-dev01 (basebc9150b overlay) with core runtime-guard-npu-dev01,
build device-read-dev01, SoC Ascend910_9392, jobs2/900s. New content/old/proposal/mixed
Read uses device per-region earliest-frame fallback for active-only adoption or
selected clear, and multi-time preparation otherwise. Intended content gate640
windows, including exact causal batch assertions. After build run --checks content
ready with queue120s/task900s, then bounded content placement if parity passes.
No full-size timing is scheduled; historical CPU Attention stays suspended.

## Current increment: selected matrix Full

`PackedFull` adds actual selected identity/tanh Full, raw ACLNN FP32 KEEP_DTYPE
batch matmul/tanh/index-copy and device-controlled bounded physical chunks.
Comparison snapshots are saved before selected clear. Inactive owners never
enter Full arithmetic; padding uses independent zero sentinels and distinct
scratch destinations. ContentFlow records actual Full results and per-call
`full_chunks` / effective `full_chunk_rows`. The scratch estimate may reduce
requested rows; one-row impossibility explicitly fails. This is local Full
parameter/scratch budgeting, not full model/KV/training memory planning.

Development source: HEAD base `61ee767` plus frozen dirty overlay
`TASK/sources/full-dev01`; per-file hashes `TASK/sources/full-dev01.snapshot.json`.
Build `TASK/builds/device-full-dev01`, matching `runtime-npu-clean01` core.
All services use prefix `tide-execution-flows-` and suffix `.service`:

| Job | Terminal result and retained output |
| --- | --- |
| build-device-full-dev01 | PASSED exit0; four CPU CTests and standalone loader checks; jobs2,900s bound; TASK/runs/build-device-full-dev01/{status.json,task.log} |
| device-full-gates-dev01 | PASSED exit0; `--checks full content`; 24 Full cases,160 window/continuation comparisons,788 events,488 actual emissions,20 windows with multi-time node batches; TASK/runs/device-full-gates-dev01/verified/result.json and per-cell logs |
| device-full-profile-dev01 | PASSED exit0; `--check full`; 426 task records:410 AIV +16 AI Core, including48 Full planners,16 BatchMatMulV2,24 Tanh,24 ScatterUpdate; TASK/runs/device-full-profile-dev01/profile/result.json and hashed raw CSVs |

NPU jobs leased physical1 as logical npu:0; module and runtime are below.
Gates cover widths1/7/33, chunks1/4, empty/identity/tanh/tail masks and inactive
NaN poison; content continuation uses chunks1/3 against independent CPU Streaming
and Greedy. Content cases preserve feedback, parallel edges, unequal delays,
missing/present-zero inputs, exact large clocks/counts, active-only adoption,
selected clear and restoring the candidate's own cut under the other schedule.
These directed development cells were followed by the clean all14-cell qualification above.
Profiling found no AiCPU task or host-fallback diagnostic. It includes construction
and CPU assertions; no throughput or complete-flow placement claim follows.

The table above preserves the earlier development results. Clean implementation
qualification is now recorded in the selected-full evidence linked above. The existing identity-Full qualification remains on clean4d2f09e:
[content-loop-20260930](evidence/content-loop-20260930.md), all13 component cells;
content80 windows/394 events/244 emissions/10 multi-time windows and11606 AIV
placement records. Selector/group evidence on clean6220011 is
[device-selection-20260930](evidence/device-selection-20260930.md); earlier clean
eff5945 packing/lifecycle evidence is
[device-packing-20260930](evidence/device-packing-20260930.md).

Confirmed execution order:

1. Close the injected-failure/runtime-lifetime gate before broad promotion;
   selected-Full clean qualification is complete.
2. Expand real module contracts/attention and packed numerical kernels, full
   memory planning and FP16; retain arbitrary legal topology/input handling.
3. Complete public graph/language/preset coverage, peer progression and actual
   backward/VJP/optimizer training; then representative/full-size comparisons.

Retained build-device-content-dev01 failed because CANN generated launch headers
could not resolve local alias I. Explicit int64_t fixed it; dev02 and clean4d2f09e
passed. Original failing source/logs are unchanged.

Required performance matrix: PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch;
CPU/NPU × streaming/prefill × inference/complete training. Five presets CPU,
mixed A/B/C, resident. Small/medium covers all; full-size compares CPU, screened
mixed and resident, both schedules. FP32 main, NPU FP16 separate, CPU FP64 oracle;
three fresh processes for recommendations. CUDA device verification remains remote.
Safe model byte chunking, complete VJP/optimizer and peer progression remain.

## Latest development and runtime repair

Runtime repair committed **07fcae4**: remove the unsafe static NPU exit finalizer;
`RuntimeSession` finalizes at the client scope before main-thread TLS teardown.
Nested owners, idempotent close and refused reopen are tested. CPU/CUDA do not
perform vendor teardown. Embedders may own teardown themselves.

- `build-runtime-npu-dev01` PASSED standalone core build.
- `build-runtime-cpu-dev01` PASSED CPU core/bindings and six real CTests:
  lifecycle, Greedy and Settle × FP32/FP64.
- `runtime-lifecycle-gates-dev01` PASSED eight fresh NPU processes.
- Frozen source `runtime-dev01` was a development overlay based on73b1c1f;
  core content is the runtime-fix implementation. These are development results.

`build-device-ready-dev01` PASSED build/four CPU CTests/standalone loader;
source `TASK/sources/ready-dev01`, binary `TASK/builds/device-ready-dev01`,
matching `runtime-npu-dev01` core. `device-components-dev01` PASSED all11 cells:
control8; numerical4 each FP32/16; eager queue; closure60; transaction46 each;
broadcast37 each; ready packing/device no-emission drain54 each. Results and
per-cell logs: `TASK/runs/device-components-dev01/verified/result.json`.
The raw control check now also owns RuntimeSession; clean qualification checks it.

Three separate development placement probes PASSED after the repair:
- `device-queue-profile-dev02`:51 queue-proposal tasks; all218 tasks AIV.
- `device-broadcast-profile-dev01`:37 route +37 queue tasks; all394 tasks AIV.
- `device-ready-profile-dev01`:65 closure +65 ready-pack +11 queue tasks AIV;
  two AiCPU OnesLike setup tasks,538 AIV tasks. No host fallback warning.
Reports under each job's `profile/result.json`. Includes setup and CPU assertions;
**not throughput evidence**. Original failures are retained unchanged.

New stages in `tools/device_online`, described in [device-control.md](device-control.md):
- QueueTransaction: device proposal/commit/refuse, bulk payload gather, stable
  actual messages, sticky capacity/coordinate errors and live-capacity reuse.
- BroadcastRouter: static CSR only; selected action outputs generate actual
  physical-edge messages on device, with exact arrival-overflow refusal.
- DeviceReady: closure, exact stable atom/fiber packing and complete region-time
  candidate frames. Device-controlled no-emission consumption continues windows
  and explicitly refuses exhausted iteration budgets.

These are mutable inference components. The new content-driven loop closes only
its explicit FP32 module profile; sparse slot delivery, other module contracts,
VJP, optimizer and peer progression still require implementation.
Scalar AIV metadata ordering needs throughput optimization; byte budgets for
model/activation/KV/communication are not covered by queue capacity.
Runtime-resource guard development now verifies deterministic API failure and
quarantine handling; its clean qualification is next. Physical hardware hangs
and driver resets remain outside that gate, requiring failed-worker termination.

## Retained failures and operational lessons

- `device-broadcast-gates-dev01` FAILED SIGSEGV(-11/245). Gdb job
  `device-broadcast-debug01` located getCurrentNPUStream -> DestroyUsedStreams ->
  finalize_npu -> static NpuShutdown at process exit. Fixed in07fcae4.
- `device-queue-profile-dev01` FAILED before marker; msprof reported Resource
  temporarily unavailable. A gdb diagnostic later completed but did not erase
  that failure. msprof may exit0 after child failure; driver rejects its warning.
- CANN control requires one global label target list, created before task emission.
- Nested captured NPUGraph RIExecuteAsync on a bound stream failed107000; the
  unsupported bridge was removed. control-dev05 retains its reproducer.
- Tensor scatter_reduce closure has a host fallback on CANN9/TorchNPU2.10 and
  explicitly rejects NPU. Exact int64 eager sorting uses device-local AiCPU.
- Optional Ascend C uses Unix Makefiles (Ninja /./ object extraction mismatch),
  underscore kernel targets and PIC. Generated kernel libraries register at load
  time; even their help belongs under a device lease, not CPU CTest.

## Immutable earlier qualification

[online-greedy-20260930](evidence/online-greedy-20260930.md), source2038d88,
evidence7aeb27c:8849 CPU FP64/FP32 tests in1713.58s, four independent C++
Greedy/Settle CTests, matching Python/standalone NPU builds and six Python/native
FP32 graph-family fixtures (observables, isolated VJPs, continuation, three AdamW
updates, fresh-process checkpoint and CPU handoff). Host scheduling + NPU tensors;
not resident execution, standalone NPU Greedy runtime or FP16 Greedy coverage.
Host max_events is live-fiber capacity, not model bytes; host atom bookkeeping remains.

[device-control-20260930](evidence/device-control-20260930.md), source5c5b582,
evidence73b1c1f: clean build/four CPU CTests/loader, five NPU component cells,
60 closure cases and60 AIV closure trace tasks. These earlier runs did not expose
the later-discovered shutdown defect. Qualification remains scoped to its source.

## Preserved older consumer work

Do not discard uncommitted `tools/accelerator_scale/flow_*`, modified bounded/
resident/peer files, client CMake/build script, `benchmark_execution_flow.py`,
`verify_execution_flows.py`, `tests/test_flow_semantics.py`. These are limited
DAG/rank-aligned finite-window consumers, not revised general-online delivery.

Prior flow-dev05 CPU24/24 passed. NPU dev05 first18 FP32 cells passed; whole job
failed at first FP16 resident Add gradient. dev06 CPU/NPU overlay builds passed;
no dev06 runtime gates. Its changes unscale gradients, compare optimizer slots,
and mask inactive proposals before Full; poison regression remains untested.
Reusable peer CPU8/NPU16 gates passed separately with TASK_QUEUE_ENABLE=0.

## Paused historical work and workload identity

`tide-execution-flows-historical-cpu-attention-01.service` is deliberately SIGSTOP
suspended. `TASK/runs/historical-cpu-attention-01/pause.json` is authoritative;
status writer still says running. Holds host memory and `TASK/timing.lock`.
Do not blindly resume. Resolve lock and invalid paused timing deliberately before
formal benchmarks, preserving records. No new full-size benchmark this increment.

New wide packet:480 body nodes,2208 body edges,2 identity boundaries;
D2048/B512/T12/V50304. Add9,468,053,696 and Attention17,521,117,376 parameters.
Historical Attention17,269,426,339 differs. Historical mixed FP32 ms/token:
Add inference CPU7.387522/NPU2 17.858098; Attention CPU19.531847/NPU4 56.012343;
Add complete training CPU78.793172/NPU4 47.932888 (NPU throughput1.6438x faster).
Attention NPU9 128.275286; no valid CPU-training ratio.

## Environment and operations

TASK=`/mi/data2T/zlong/tide-execution-flows`; runs/status/logs under `TASK/runs/NAME`,
linked through `artifacts/execution-flows-NAME`. Frozen snapshots and original
failure reproducers retained. Last free-space check323GB; refresh for large writes.

Module `libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
TASK_QUEUE_ENABLE=0; TORCH_DEVICE_BACKEND_AUTOLOAD=0; retain module PYTHONPATH and
prepend source/python. CPU ATen/BLAS threads1. TIDE_BUILD_DIR selects matching
binding. Actual SoC Ascend910_9392,16 chips64GiB; cooperative device leases.
Trackio `/home/zlong/venvs/trackio/bin/python`, project `tide-execution-flows`.

`TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT [--commit COMMIT]
[--npu --max-wait120] -- COMMAND` creates a frozen hashed snapshot and detached
background.slice service. Use --commit for qualification. Placeholders:
{python}, {base}, {source}, {out}. Long jobs retain exact commands/status/logs.
On re-entry run `git status --short --branch` and `python scripts/status.py`, read
this handoff and actual terminal records, then continue F1–F7 under the renewed
user authorization. Push tested commits; do not resume the historical slow job blindly. Submitted/running jobs are never passing evidence.
