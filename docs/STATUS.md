# Current handoff

Updated 2026-10-01T01:43:40.665537+00:00. **ACTIVE: user confirmed alignment and explicitly resumed implementation.**
No subagents. Reference repositories and ObsidianVault remain read-only.
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
F1–F7 remain incomplete. Cleanup e0afef4 and evidence e53abf5 are pushed.
Event-training development checks are complete; placement profiling is running.
Event/fiber fixed-commit qualification will be grouped at the attention milestone.
Historical CPU Attention is deliberately SIGSTOP; do not resume or stop blindly.

## Contract and iteration policy

[execution-flows.md](execution-flows.md) is the execution contract;
[ROADMAP F1–F7](ROADMAP.md) is the only backlog. Current user alignment outranks
run-ml-experiments: retain source/input/config/environment identity, original
results/failures, synchronized complete timing, finite resources and stop conditions.
Do not expand tracking infrastructure or block implementation on Trackio.
Training means forward/backward/VJP/optimizer/continuation/throughput; model
convergence belongs in later consumer experiments.

Candidates independently execute actual input: no reference events, routes,
values or gradients may enter their execution. General online greedy accepts
legal topology/input including positive-delay PDG feedback; natural streaming
fallback is valid. Preserve int64, stable order, parallel edges, missing/zero,
None/zero. Required performance: PDG LibTorch; TimedDAG/Settle LibTorch and
PyTorch; CPU/NPU × streaming/prefill × inference/complete training. Five presets
retain finer placement switches; FP32 main, FP16 separate. CUDA is target-pending.

Use directed affected-path checks during development. Group clean immutable
qualification at semantic milestones and reuse byte-verified unchanged build
objects for development. Do not rerun the unchanged portable core's 8,954 CPU
checks / 23 optional skips. Development evidence does not certify a fixed commit.

## Qualified state and current results

ROADMAP links exact commits and evidence for packed resident forward, public
CPU/mixed placement, inference/checkpoints, state/Full/graph VJP, alias-owner
reduction, device optimizers, retained windows, public C++/Python training,
LH/SwiGLU and normalized Aggregate. Python resident is a C++/CANN client, not an
independent pure-PyTorch resident schedule.

Fixed **4b8ced41f422301a4ced2bf2999c5d1d3cb63e3d** control/Read training qualification
is complete; evidence is published in
[resident-control-training-20261001](evidence/resident-control-training-20261001.md):

- build-control-training-clean01 / build-control-training-python-clean01 PASSED;
  separate standalone and Python-owned builds; standalone five CTests.
- control-training-clean01 PASSED all 48 component cells. Control VJP: 36
  configurations / 108 replays / CPU FP32+FP64. Public control training: 98
  trajectories / 1,568 retained windows / 392 updates, strict comparisons.
- control-training-python-clean01 PASSED 154 NPU cases, no skips;
  control-training-host-clean01 PASSED 76 / 151 optional NPU skips.
- control-training-profile-clean01 PASSED; separate 1,024 MB correctness trace,
  no observed AiCPU or host fallback. Includes construction/CPU assertions;
  not throughput. Physical 9 for gate, 13 for profile, each logical npu:0.
- TASK/launchers/control_training_evidence.py audits the 40-character revision,
  builds, all results and retained failures; PASSED and evidence written.

Numerical limits remain: LH/SwiGLU, Aggregate and control trajectories use
AdamW eps1e-5; public eps1e-8 and normalization epsilons are unchanged. RMSNorm
eps1e-8 near-zero strict end-to-end trajectory still fails. Only the existing
Full regression uses explicit conditioned controls (13 frames, max4.470348e-6);
new control training remains strict, with exact routes. Width257 retained
SwiGLU declares 2 GiB reverse budget. Prior strict failures remain failed.

Retained control failures: control-training-dev01 SIGSEGV in a fixture's absent
edge-scale indexing; control-training-python-new-dev02 unsupported Python Full
identity fixture; control-training-profile-dev02 application passed but 200 MB
aging/CANN task association prevented profiler export. Their later successful
runs do not relabel these records. The evidence audit includes all three.

## Old-work cleanup — committed and pushed e0afef4

All 24 old dirty files were SHA256-verified and pushed in
`archive/restricted-flow-20260930` at **964bf628c67270200dabe55b1bca026bd403cd37**;
TASK/restricted-flow-archive.json records exact file hashes. The archive is an
unqualified restricted consumer, not the general online delivery. No content lost.

Extracted fixes: inactive Full input masking before nonlinear arithmetic;
FP64 resident control precision; peer bridge release after completed VJP.
New tests cover overflowing unused Full, None/zero, two-window FP64/FP32,
independent Python/native packet semantics and three optimizer updates.

- cleanup-python-dev01 PASSED 56 cases.
- cleanup-relink-dev02 PASSED dtype CTest in CPU and standalone-NPU-linked CPU
  processes. Reused production source/object identity recorded in relink.json.
- cleanup-guard-relink-dev03 PASSED isolated overflow-fixture rebuild.
- cleanup-cpu-dev03 PASSED four affected Add/Attention forward/VJP/training cases.
- cleanup-npu-dev03 PASSED those four plus actual two-device peer replay/VJP;
  leased physical 1,3 as npu:0,1; queue120s/run1200s, now terminal.

Original dev01 builds FAILED a new fixture's pre-compilation graph identity;
dev02 gates FAILED a new overflow probe's unconfigured Full kind. Both fixture
errors were fixed without production changes; original failure logs remain.
Cleanup dev03 uses sources/cleanup-dev03 and builds/cleanup-{cpu,npu}-dev03;
this is directed development verification with byte-checked reused objects.

## Event attention increment — development verified, qualification pending

The candidate now trains aggregated-event GQA/window attention through actual
KV tape associations, device cache-chain reverse, Q/K/V/O alias reduction and
optimizer publication. Public cache roots/initial gradients and retained KV
bridges preserve None/zero, clear/eviction, empty caches and parameter-epoch
continuation. [Contract](resident-event-vjp.md). This is single-NPU FP32;
fiber adjoints, resident FP16 and peer progression/training remain pending.

All of these development jobs PASSED with exit0:
- build-event-training-dev01 / build-event-training-python-dev01: complete
  separate runtime-owner builds; standalone five host CTests also passed.
- build-event-training-dev02 / build-event-training-python-dev02: directed
  relinks, reusing byte-checked terminal objects; budget/checker-only fixes.
  development.json records reused hashes and loader closure. All current backend
  bytes match frozen sources/event-training-dev02.
- event-training-standalone-dev02:66 isolated/combined root cases and8 complete
  trajectories (32 updates,128 retained windows), CPU FP32/FP64, two schedules,
  feedback/DAG, initial and final independent KV, empty/clear/evicted caches,
  HST/SOFTP, cross-role aliases, widths1/4/257 and checkpoint continuation.
- event-training-python-new-dev02:17 cases passed, three families/both
  schedules/optimizers, device loss cotangents, None/zero, fresh-process resume.
- event-training-python-regression-dev02:154 old public cases passed.
- event-training-components-dev02:all8 affected attention/state/graph/parameter/
  retained/public/control/Aggregate component cells passed.

`event-training-profile-dev02` is RUNNING (queue120s/run360s,1024MB collection).
Inspect runs/NAME/status.json and profile/result.json before claiming placement.
Correctness and profile are separate; neither is complete throughput evidence.
NPU tests leased physical9 or13, each remapped npu:0. No long queue remains.

Next: commit/push this coherent implementation after these directed checks;
inspect the ongoing profile. Continue same-fiber attention training, including
complete current-fiber visibility, sum/mean/learned pooling, bias/decay slots and
source/scale adjoints, with independent CPU FP32/FP64 references. Reuse current
local attention VJP, never event aggregation as a substitute for a real fiber.
Group clean immutable builds/gates/profile for event+fiber at the attention
milestone instead of repeating all CANN compilation per intermediate module.
Development evidence does not certify a fixed commit. Then continue resident
FP16/peer progression, complete matrix, representative and full-size performance.
All new attention sources/tests/docs and shared-fixture CMake changes belong
to this increment; no unrelated dirty work or discarded files remain.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service. Status/log: RUN/status.json, RUN/task.log;
gate/profile under corresponding subdirectories. Module
libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized public /opt stack supersedes dated personal guide.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Queue chooses physical devices;
programs use logical npu:0. Last disk check:244 GiB data,24 GiB root free.

Cores: builds/placement-cpu-clean01, placement-npu-clean01 (standalone),
placement-npu-python-clean01 (Python-owned). Never load standalone SDK in
TorchNPU Python. Launch immutable/frozen tasks with
`python TASK/launchers/freeze_run.py --name NAME --snapshot NEW [--commit REV]
[--npu --npu-count N --max-wait 120] -- timeout --signal=TERM --kill-after=10s
900s '{python}' ...`. New edits require a new snapshot name. Builds use two
workers,1800s upper bound. All jobs use background.slice,Nice10. Never mutate
terminal evidence builds or snapshots. Use durable_records.py for atomic handoffs.

Historical historical-cpu-attention-01 is SIGSTOP; pause.json overrides running
status. It retains host memory and TASK/timing.lock. Resolve interrupted timing
before formal performance. Historical Add CPU78.793172/NPU4 47.932888 ms/token
means throughput1.6438x faster; it does not certify this resident backend.
No valid complete CPU Attention training ratio exists.
