# Current handoff

Updated 2026-09-30T05:26:17.982017+00:00. **ACTIVE: continue the user-confirmed execution contract.**
Repo `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`, branch `graph-execution-foundation`.
No subagents, no push. Reference repos and ObsidianVault remain read-only.

## Contract and next action

[execution-flows.md](execution-flows.md) is authoritative; [ROADMAP F1–F7](ROADMAP.md)
is the single backlog. General online node-time greedy prefill accepts every
family's legal topology/input, including positive-delay PDG feedback. No numerical
route prepass, whole-window potential expansion or fixture shortcuts. Residency
includes online decisions and progression, not just tensor storage or fixed capture.

NEXT: selector and grouped queue transactions passed development and are ready
for an implementation commit. `FrameSelector` covers count-v1/positive-v1 with
FP32 scores, exact int64 counts/ties, complete candidate softmax and proposed
history. `QueueTransaction` now preflights/snapshots all participating queues
before any shared-error commit, including cross-queue payload aliases.
`build-device-selector-dev02` and `device-components-dev02` PASSED all12 cells
on frozen selector-dev02 with matching clean runtime-npu-clean01 core. Queue
cells include46 ordinary +6 transaction-group cases per FP32/FP16; selector39.
`device-queue-profile-dev03` PASSED:63 proposal tasks, all475 device tasks AIV.
`device-selector-profile-dev01` PASSED:39 selector tasks, all428 tasks AIV.
These are development component results, not complete graph execution/training.

Commit only the device implementation/docs/build registry (preserve older dirty
consumer work), then run a clean build and all12 cells plus queue/selector
profiles at that exact commit. Core has not changed since eff5945; the build
driver checks source and binary fingerprints before reusing runtime-npu-clean01.
Use snapshot selector-clean01/build device-selector-clean01, jobs2,900s build;
queue waits120s, gates900s, profiles480s. If a job fails, preserve it and fix from
the reproducer before qualifying. Then separately commit evidence. Continue
integration of real module work/Read, selection, state/continuation and actual
emission feedback; the no-emission drain is not a graph executor.

Clean eff5945 qualification is committed as f604bfc:
[device-packing-20260930](evidence/device-packing-20260930.md). All eight jobs
passed, including CPU six CTests, NPU core/component builds, eight lifecycle
processes, all11 component cells and three placement profiles. Do not reuse
an older binary under a new source claim.

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

These are mutable inference components. No complete Tide numerical module/state
loop, sparse slot delivery, VJP, optimizer or peer progression yet.
Scalar AIV metadata ordering needs throughput optimization; byte budgets for
model/activation/KV/communication are not covered by queue capacity.
Runtime timeout handling quarantines uncertain live program owners; injected
failure/lifetime testing, including runtime-scope finalization of such workers,
is still required before promoting the complete backend.

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
this handoff and actual terminal records, then continue F1–F7. Do not ask again
for authorization. Submitted/running jobs are never passing evidence.
