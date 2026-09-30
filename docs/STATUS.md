# Current handoff

Updated 2026-09-30T04:27:01.860062+00:00. **ACTIVE: continue the user-confirmed execution contract.**
Repo `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`, branch `graph-execution-foundation`.
No subagents, no push. Reference repos and ObsidianVault remain read-only.

## Contract and next action

[execution-flows.md](execution-flows.md) is authoritative; [ROADMAP F1–F7](ROADMAP.md)
is the single backlog. General online node-time greedy prefill accepts each
family's arbitrary legal topology/input, including positive-delay PDG feedback.
No numerical route prepass, whole-window potential expansion or fixture shortcuts.
Natural streaming degeneration is legal. Residency includes online decisions and
progression, not just NPU tensor storage or fixed capture.

NEXT: fix standalone NPU shutdown order, then rerun broadcast/queue/profile gates.
Gdb `device-broadcast-debug01` located SIGSEGV in getCurrentNPUStream called by
DestroyUsedStreams -> finalize_npu -> static NpuShutdown destructor at process exit.
New RuntimeSession owns teardown before main-thread TLS destruction; nested/idempotent
lifecycle checks added. `build-runtime-npu-dev01` PASSED (1800s/two workers),
frozen runtime-dev01. `build-runtime-cpu-dev01` PASSED CPU build and all six CTests.
`runtime-lifecycle-gates-dev01` PASSED eight fresh NPU processes. Component build follows
with the matching new NPU core. Launch `runtime-lifecycle-gates-dev01`, eight fresh
NPU processes to check the exit-order defect,120s queue/400s task.
`build-device-ready-dev01` PASSED (four CPU CTests/loader), snapshot ready-dev01,
using core runtime-npu-dev01. Launch `device-components-dev01`, full11-cell FP32/FP16
component gate via scripts/verify_device_control.py,120s queue/900s task.
Includes closure/fiber/frame packing and device no-emission drain loop. Inspect
its verified/result.json and per-cell logs; a printed marker without process exit0 is insufficient.
Develop ready-fiber/region packing while the core builds; no formal heavy timing.

Clean5c5b582 qualification remains the historical result in
[evidence](evidence/device-control-20260930.md); it did not expose the exit bug.
QueueTransaction dev01 passed46 FP32 and46 FP16 cases. Broadcast dev01 built but
its first gate ended SIGSEGV; it is NOT passed. Queue profile dev01 also failed
before marker; separate gdb diagnostic completed normally but does not erase it.
All reproducer snapshots/logs are retained. Runtime/profile repeat requires the fix.

Device component implementation is `5c5b582`; `7aeb27c` is
immutable host-greedy evidence. Implementation `2038d88`
is qualified as below. The device component/build/profile scripts are now committed. Only older
consumer development edits and the active handoff remain uncommitted.

Required performance matrix: PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch;
CPU/NPU × streaming/prefill × inference/complete training. Five presets CPU,
mixed A/B/C, resident. Small/medium covers all; full-size compares CPU, screened
mixed and resident, both schedules. FP32 main, NPU FP16 separate, CPU FP64 oracle;
three fresh processes for recommendations. CUDA device verification remains remote.
Conservative/aggressive-safe model byte chunking, continuation, VJP and optimizer
boundaries remain requirements. Queue/closure scratch capacity is not model budgeting.

## Terminal host-greedy qualification

Evidence: [online-greedy-20260930](evidence/online-greedy-20260930.md), committed
as `7aeb27c`, source `2038d8867ad0133af9b95589379435c95a326f03`, clean snapshot
`TASK/sources/greedy-clean01`.

- `greedy-clean-cpu01`: clean CPU build; four actual standalone Greedy/Settle
  FP64/FP32 CTests; **8849 tests passed in1713.58s**. Entire job passed.
- `build-greedy-npu-python01` and `build-greedy-npu-sdk01`: passed matching
  Python-owned and standalone SDK builds.
- `greedy-npu-semantic01`: six FP32 Python/native × PDG/TimedDAG/Settle fixtures
  passed independent CPU observables/VJPs, chunk continuation, three AdamW updates,
  fresh-process checkpoint and CPU handoff. `semantic.json` and six reports.
  This is host scheduling + NPU tensors, not resident execution or exhaustive
  NPU coverage. Independent C++ NPU Greedy runtime qualification remains separate.
- Earlier development611-test gate,16 cyclic attention cases and four C++ client
  runs passed; overlap is not added to the8849 count. Earlier mistaken tests,
  missing CTests and retained failures are not relabelled.

Host Greedy uses actual pending fibers, positive-delay region closure and safe
region-time prefixes with existing local block kernels. `max_events` limits live
fibers, not cumulative work or byte usage. Host per-atom bookkeeping remains.

## Device component: clean qualification and development

Prepared `build-device-control-clean01`: clean snapshot `control-clean01` at
5c5b582; output `builds/device-control-clean01`, matching immutable core
`greedy-npu-sdk01`;600s build, two workers. Build PASSED at04:15:57Z; four CPU
CTests and standalone loader passed. Clean device and profile qualification PASSED;60 closure tasks on AIV.
See immutable evidence above. New queue transaction development is not yet qualified.

`tools/device_online` contains CANN runtime control, raw ACLNN packed arithmetic,
fixed-capacity tensor queue and an optional Ascend C metadata-closure kernel.
Details and limits: [device-control.md](device-control.md).

`build-device-control-dev14` PASSED build/four CPU CTests/standalone loader.
Snapshot `TASK/sources/control-dev14`; binaries `TASK/builds/device-control-dev14`.
Component source hash in `control-build.json`; exact SoC `Ascend910_9392`.
Build uses two workers and a600s bound, Unix Makefiles for optional Ascend C.

`device-control-gates-dev13` PASSED all five component cells, physical1/logical0:
- eight raw control cases: changed limits, zero work, explicit exhaustion and
  continuation, exact int64 above2^55, normal lifecycle;
- packed numerical loop: four FP32 and four FP16 scalar-CPU-recurrence comparisons;
- tensor queue: stable physical-edge order, missing/zero, atomic overflow refusal,
  reusable live capacity; int64 sorting uses device-local AiCPU;
-40 Ascend C closure/branch cases, feedback/unaligned/parallel edges, changed
  input, exact int64 near maximum and invalid live coordinates.

`device-closure-gate-dev14` PASSED all60 cases, including empty queue and
right-boundary checks, physical1/logical0. Prepared `device-closure-profile-dev14`:
msprof, one NPU120s queue/480s task, with separate profile/result.json/log/raw
operator records. Profile PASSED:60 tide_closure tasks on AI_VECTOR_CORE;
390 total operators including20 AI_CPU OnesLike construction initializers.
No host CPU fallback warning. No performance claim. Other unchanged components retain dev13 results.

The kernel is one AIV scalar-pipeline program with preallocated metadata scratch.
It computes readiness and a branch flag from actual pending work, no host scalar
choices. It does NOT yet consume/emit the complete queue, dispatch Tide modules,
carry graph state, run backward/optimizer, or coordinate peers. Fixed tensor queue
packing is separately checked under eager ATen, not yet integrated in that loop.
The mutable low-level primitives do not provide autograd. Timeout cleanup refuses
to free resources without confirmed completion; an injected-failure lifetime gate
is still pending before promoting the complete backend.

Observed constraints/failures retained under TASK/runs and frozen sources:
- raw loop probes01/02: capture/ordinary streams cannot be used as explicitly
  bound persistent streams; probe03 prematurely unbound; probe04 passed.
- control dev01 needed list creation before label marking; dev02 eight cases passed.
- nested captured numerical submodel (`device-stage-gate-dev05`) rejected107000:
  RIExecuteAsync cannot execute on a model-bound stream. Failed bridge removed;
  dev05 snapshot preserves reproducer. Direct ACLNN numerical stages pass instead.
- dev07 tensor closure returned correct values but emitted a real host CPU fallback
  for `scatter_reduce`; NOT native closure evidence. It now explicitly refuses NPU.
  AiCPU int64 sort remains distinct from that host fallback.
- CANN build integration required SDK-consistent headers, Makefiles to avoid Ninja
  /./ object-path keys, underscore kernel target names and PIC objects for PIE.
  Original failed build records remain terminal failures.
- closure dev12 stopped before kernel execution because the same label appeared
  in two CANN lists. One global list plus device index mapping fixed this in dev13.

## Preserved older consumer work

Do not discard uncommitted `tools/accelerator_scale/flow_*`, modified bounded/
resident/peer files, client CMake/build script, `benchmark_execution_flow.py`,
`verify_execution_flows.py`, `tests/test_flow_semantics.py`. They are limited
DAG/rank-aligned, finite-window consumers, not the revised general-online delivery.

Prior flow-dev05 CPU24/24 passed. NPU dev05 first18 FP32 cells passed, whole job
failed at first FP16 resident Add gradient. dev06 CPU/NPU overlay builds passed;
no dev06 runtime gates. Its changes unscale gradients, compare optimizer slots,
and mask inactive proposals before Full; poison regression remains untested.
Reusable peer CPU8/NPU16 gates passed separately with TASK_QUEUE_ENABLE=0.
These development results do not replace clean general-online qualification.

## Paused historical work and workload identity

`tide-execution-flows-historical-cpu-attention-01.service` is deliberately SIGSTOP
suspended, with `TASK/runs/historical-cpu-attention-01/pause.json` authoritative.
Its status writer still says running. It holds host memory and `TASK/timing.lock`.
Do not blindly resume. Resolve the held lock and invalid paused wall-clock timing
before formal timing; stop/restart deliberately with preserved records if needed.
The paused baseline must not block implementation. No new full-size benchmark was
launched in this increment.

New wide packet:480 reachable body nodes,2208 body edges,2 identity boundaries;
D2048/B512/T12/V50304. Add9,468,053,696 and Attention17,521,117,376 parameters.
Historical Attention17,269,426,339 is different. Historical mixed FP32 ms/token:
Add inference CPU7.387522/NPU2 17.858098; Attention inference CPU19.531847/NPU4
56.012343; Add complete training CPU78.793172/NPU4 47.932888 (NPU throughput
1.6438x faster). Attention NPU9 128.275286; no valid CPU-training ratio.

## Environment and operations

TASK=`/mi/data2T/zlong/tide-execution-flows`; runs/status/logs under `TASK/runs/NAME`,
linked through `artifacts/execution-flows-NAME`. Snapshot manifests and all failed
reproducers retained. Last space check324GB free; refresh before large writes.

Module `libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
TASK_QUEUE_ENABLE=0; TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH and
prepend source/python. CPU ATen/BLAS threads1. TIDE_BUILD_DIR selects the matching
binding. Device leases use the cooperative queue; never interrupt unrelated users.
Trackio `/home/zlong/venvs/trackio/bin/python`, project `tide-execution-flows`.

`TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT [--commit COMMIT]
[--npu --max-wait120] -- COMMAND` creates a frozen hashed snapshot and detached
background.slice service. Use --commit for immutable qualification. Important
placeholders: {python}, {base}, {source}, {out}. Build/gate drivers and commands
are recorded in the task statuses. Current directed gate executable:
`TASK/builds/device-control-dev14/tide-device-closure-check --device=npu:0 --dtype=float32`.

On re-entry run `git status --short --branch` and `python scripts/status.py`, read
this handoff, inspect actual terminal records, then continue F1–F7 without asking
again for authorization. Never call submitted/running jobs passed.
