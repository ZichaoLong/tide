# Current handoff

Updated 2026-10-01T03:30:54.724200+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
Commit/push authorization remains active; no requested pause. No subagents.
Reference repositories and ObsidianVault are read-only. Repository
`/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
[execution-flows.md](execution-flows.md) owns the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. F1–F7 remain incomplete.

## Contract and scope

Candidates independently consume common inputs/initial state/parameters. No CPU
reference event, route, numerical result or gradient becomes a candidate input.
Online greedy accepts legal topology/input, including positive-delay PDG feedback;
it may naturally degenerate to streaming. Keep int64, stable order, parallel-edge
identity, missing/zero messages and None/zero gradients. Required performance:
PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Python resident is a C++/CANN client, not an independent
PyTorch device scheduler. Five placement presets retain finer switches. FP32 main;
FP16 separate. CUDA execution is target-machine-pending.

Current alignment outranks run-ml-experiments. Keep minimal source/input/config/
environment identity, raw failures/results, synchronized complete timing, bounded
resources/stops. No duplicate tracking or Trackio blocker. Training means
forward/loss interface/backward/VJP/optimizer/continuation/throughput; downstream
model convergence is outside this foundation's delivery scope.
Use affected-path development checks and byte-verified terminal build reuse.
Do not rerun the unchanged portable core's8,954 CPU checks/23 optional skips.
Commit implementation, qualify clean immutable source, commit evidence separately.

## Qualified attention milestone

Pushed implementation **66a6ca5702d3618ca72ad57ac87f01a7190f9166** contains event and
same-fiber attention complete training. Clean qualification and audit passed:
[report](evidence/resident-attention-training-20261001.md),
[audit](evidence/resident-attention-training-20261001.json).
Independent event/fiber cache roots, retained KV/log-bias links, five fiber pooling
profiles, physical source/scale adjoints, shared owners and optimizer publication
into actual live banks are covered. Single-NPU FP32; no new throughput claim.

All qualification jobs are terminal PASSED/exit0 at that exact clean revision:
- build-attention-training-clean01: normal standalone build,5 CTests/loader.
- build-attention-training-python-clean01: separate Python-owned build/loader.
- attention-training-clean01:all52 registered component cells (physical9->logical0).
  Event66 root cases/8 trajectories; fiber172 roots/20 trajectories; independent
  CPU FP32/FP64. Local event4 configurations/12 replays,fiber37/74.
- attention-training-python-clean01:196 NPU cases,zero skips,including42 new
  event/fiber cases,real loss cotangents,three families/two schedules/optimizers,
  new-process checkpoint suffix (physical9).
- attention-training-host-clean01:76 CPU interface tests,193 optional NPU skips.
- build-attention-consumer-clean01 / attention-consumer-clean01: installed public
  C++ inference/retained training/optimizer restore,passed (physical13).
- attention-training-profile-clean01: separate bounded mixed event/fiber roots
  and one complete AdamW trajectory,512MB. 13267 tasks,275 fiber-reverse,
  25 event-reverse,126 cache-bias merge,16 optimizer; AI_VECTOR_CORE/AI_CORE/MIX_AIV.
  No observed AiCPU or logged CPU fallback. Includes CPU assertions,not throughput.

Audit script TASK/launchers/attention_training_evidence.py verifies source/core/
binary/loader/log/CSV hashes,complete cell inventory and retained failures.
The five failed fiber development jobs remain failed: GM scalar Muls operand,
launcher include omission,host include omission,ambiguous checker Tensor assignment,
and illegal sparse SourceDomain fixture. See audit for exact job identities.

Existing numerical boundaries remain: AdamW trajectories explicitly use eps1e-5;
public eps1e-8 and normalization epsilons unchanged. Only the existing Full
regression uses conditioned controls (13 frames,max4.470348e-6); new attention
checks remain strict. RMSNorm eps1e-8 near-zero strict trajectory failure remains
in the earlier evidence. Width257 SwiGLU declares2GiB reverse budget.

## FP16 component increment — development passed, commit next

Working tree contains explicit CANN FP32/FP16 conversion, real half PackedFull /
PackedLhFull buffers and byte budgets, FP16 scalar/vector packed sum with FP32
products/ordered accumulation, and shared vector tile conversion overloads.
Full/LH selection/chunk logic remains shared with FP32; inactive NaN and exact
int64 metadata checks remain. Entire resident session FP16 is still refused.
The normal build script now supports explicit standalone --checks and records
only the requested binaries; Full/LH have separate CMake targets. Omitted
--checks retains the complete build. Docs describe the component-only boundary.

All development jobs below are terminal PASSED/exit0:
- build-low-precision-components-dev01 / low-precision-components-dev01:
  recompiled host units with byte-verified qualified Full planner reuse;
  numerical/full FP32/FP16 passed. Separate low-precision-full-profile-dev01:
  544 AI_VECTOR_CORE/24 AI_CORE tasks,no AiCPU/logged fallback.
- build-low-precision-components-dev02 / low-precision-components-dev02:
  normal directed build,no reused CANN archive,one CTest,four dtype cells passed.
- build-low-precision-components-dev03 / low-precision-components-dev03:
  normal numerical/full/packed-lh build,one CTest,six dtype cells passed.
  Full32 cases per dtype; LH40 cases/9 profiles/536 norm rows per dtype.
  FP16 LH max CPU/device-vs-FP64 abs0.0276378 in conditioned fixtures;
  separate half budget,FP32 thresholds unchanged. low-precision-lh-profile-dev03:
  5364 AI_VECTOR_CORE tasks,no AiCPU/logged fallback.
- build-low-precision-sum-dev04 / low-precision-sum-dev04:
  normal --checks sum build,both dtypes passed (72 configurations/144 replays/
  18 refusals per dtype,widths1..2048,tails,empty/zero/NaN isolation).
  low-precision-sum-profile-dev04 passed separately:1205 AI_VECTOR_CORE tasks,
  no AiCPU/logged fallback. All three profiles include CPU assertions,not throughput.

Sources/builds use the same suffix under TASK/sources and TASK/builds.
Earlier Full/LH production bytes match dev03; subsequent changes affect sum,
shared vector helper and registry/build selection only. Current sum bytes match
low-precision-sum-dev04. These are dirty-source development checks,not a clean
complete-resident FP16 qualification. No new job remains live.

Attention evidence commit6cb3e2d is pushed.
Next: commit/push the tested FP16 component increment,then use a clean exact source
with scripts/build_device_control.py --checks numerical full packed-lh sum
(standalone core below,Ascend910_9392,jobs2,new build dir). Qualify that explicit
subset; do not call it full resident support. Continue FP16 state/attention/Read/
VJP/master publication and complete sessions,peer progression/communication/
training,representative five-preset screening and full-size CPU/mixed/resident
comparisons,then migration/version/CUDA-pending evidence per F1–F7.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service; RUN/status.json and task.log own lifecycle.
Gate/profile result.json and per-case logs live below their named subdirectories.
Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes the dated personal guide under user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical indices only.
Last checked space:240GiB data,25GiB root; recheck before large writes.

Core builds: placement-cpu-clean01,placement-npu-clean01 (standalone),
placement-npu-python-clean01 (Python-owned),all under TASK/builds.
Never load the standalone SDK into TorchNPU Python. Freeze with
`python TASK/launchers/freeze_run.py --name NAME --snapshot NEW [--commit REV]
[--npu --npu-count N --max-wait 120] -- timeout --signal=TERM --kill-after=10s
900s '{python}' ...`. New edits require a new snapshot. Builds use2 workers;
long jobs use background.slice,Nice10,finite bounds. Never mutate active
snapshots or terminal evidence. Use durable_records.py for atomic handoffs.

## Preserved historical boundaries

The24 earlier dirty files remain SHA256-verified on pushed
`archive/restricted-flow-20260930` at964bf628c67270200dabe55b1bca026bd403cd37;
TASK/restricted-flow-archive.json owns hashes. This is not the general backend.
Historical historical-cpu-attention-01 remains deliberately SIGSTOP; pause.json
overrides running status. Do not blindly resume or stop. It retains host memory
and TASK/timing.lock. Resolve interrupted timing before formal throughput.
Historical Add CPU78.793172/NPU4 47.932888ms/token is throughput1.6438x faster;
it does not certify resident execution. No complete CPU Attention training ratio.
