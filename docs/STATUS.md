# Current handoff

Updated 2026-10-01T06:47:11.941016+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
Commit/push authorization remains active; no requested pause. No subagents.
Reference repositories and ObsidianVault are read-only. Repository
`/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
[execution-flows.md](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the only backlog. F1–F7 remain incomplete.

## Contract and scope

Candidates independently consume common inputs/initial state/parameters. No CPU
reference event, route, numerical result or gradient becomes a candidate input.
Online greedy accepts legal topology/input, including positive-delay PDG feedback;
it may naturally degenerate to streaming. Preserve int64, stable order, parallel
edge identity, missing/zero messages and None/zero gradients. Performance matrix:
PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Python resident is a C++/CANN client, not an independent
PyTorch device scheduler. Five placement presets retain finer switches. FP32 main;
FP16 separate. CUDA execution is target-machine-pending.
Current alignment outranks run-ml-experiments: minimal source/input/config/environment
identity, raw failures/results, synchronized complete timing, bounded resources/stops.
No duplicate tracking or Trackio blocker. Training means forward/loss interface/
backward/VJP/optimizer/continuation/throughput; downstream convergence is out of scope.
Use affected-path checks and byte-verified terminal build reuse. Do not rerun the
unchanged portable core's 8,954 CPU checks/23 optional skips. Commit implementation,
qualify clean immutable source, commit evidence separately.

## Latest clean qualification

Implementation **5ce5346208fe73b33c9a5a85f1fef56441d76073** committed/pushed.
[FP16 normalized Aggregate/LH/SwiGLU adjoints](evidence/resident-fp16-extended-vjp-20261001.md)
and [audit](evidence/resident-fp16-extended-vjp-20261001.json).
All seven clean-source jobs PASSED/exit0:
- build-low-precision-extended-vjp-clean01:two affected host objects, shared
  comparison fixture and two checkers rebuilt; byte-matched terminal kernels reused.
- build-low-precision-extended-vjp-python-clean01:three affected Python-owned
  host objects rebuilt; client relinked, matching CANN archives reused.
- low-precision-extended-vjp-components-clean01:four dtype cells,physical9→logical0.
- low-precision-extended-vjp-regression-clean01:FP32 event/fiber training,physical13.
- low-precision-extended-vjp-python-clean01:215 passed,zero skips,physical1.
- low-precision-aggregate-vjp-profile-clean01:physical3,2277 AI_VECTOR_CORE/182 MIX_AIV.
- low-precision-extra-full-vjp-profile-clean01:physical9,8966 AI_VECTOR_CORE/
  88 AI_CORE/195 MIX_AIV. No observed AiCPU/logged CPU fallback. Not throughput.
All selected physical devices map to logical0.

Per dtype:Aggregate39 cases/117 replays; LH/SwiGLU90 cases/270 replays. Half
adds three strict rounding/cancellation fixtures, each with19/7/0 replay.
Aggregate tolerances remain1e-5/1e-6; Full half2e-3/2e-5; FP32/FP64 and anchors
1e-5/1e-6. CPU independent quantized-forward FP32/FP64 adjoints; actual candidate
operands are half, adjoint accumulation FP32. Preserve all None/zero/poison checks.
FP32 event/fiber regression has66/172 roots,8/20 trajectories,strict controls.

Preserved failures:Aggregate dev01 build name collision; extended dev01 shared
test helper still static; extended dev02/dev03 half LayerNorm numerical mismatch.
CANN half LayerNorm returned half-rounded mean/rstd inside FP32 buffers,
input-gradient error8.75e-4. Fix computes only Jacobian statistics in FP32 on
actual half activation and retains half normalized output for weight VJP.
Original failure records remain; no tolerance change. Audit passed all source,
core,binary,loader,archive members,raw logs and profiler CSV checks.

Prior qualifications: d2a1afc state/basic Full adjoints,
[report](evidence/resident-fp16-basic-vjp-20261001.md);9a84432 master/publication,
[report](evidence/resident-fp16-master-publication-20261001.md);8b05c04 HARD half
inference;66a6ca5 FP32 event/fiber training. These scopes do not enable complete
FP16 graph reverse or public training. No new full-size speed ratio.

## Active work and next action

Uncommitted next increment: local attention half VJP. Q/K/V and QK products use
payload precision; softmax/global key-tile normalization and adjoints use FP32.
Returned output rounds to actual payload after global derivative correction.
Whole-graph, event/fiber cache reverse and public FP16 training remain guarded.

Build-low-precision-attention-vjp-dev01 PASSED. Its component gate passed all
ordinary FP32/FP16 geometries, then FAILED its strict fixture's own required
sensitivity check. CPU analysis found nearly equal score-rounding errors; one
key changed from-2.71875 to-2.703125 to distinguish omitted QK rounding. Candidate
math/tolerances unchanged. Raw failure retained. FP32 event/fiber regression
low-precision-attention-vjp-regression-dev01 PASSED/exit0.
Build-low-precision-attention-vjp-dev02 PASSED; checker-only rebuild with
byte-verified dev01 host/kernel reuse. Active gate: low-precision-attention-vjp-dev02,
physical assignment in RUN/queue.json. Frozen source/build:
TASK/{sources,builds}/low-precision-attention-vjp-dev02. Inspect RUN/status.json
and gate/result.json before claiming success. Lease max120s; run600s.

First commit the reviewed extended-VJP evidence separately, preserving uncommitted
attention work. Once attention gate passes, commit its implementation, qualify
fixed clean standalone/Python builds, affected attention clients and profile.
Build launcher:build_precision_attention_vjp.py (full affected kernel/host/checker)
or build_precision_attention_recheck.py (checker-only, terminal dev01 dependencies).
Then event/fiber/control/graph half adjoints, retained windows, master checkpoint/
public FP16 training; peer progression/communication/training,five-preset screening,
representative/full-size CPU/mixed/resident performance and version/migration/CUDA
records. F1–F7 incomplete. Authorization remains active. No subagents or requested pause.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service; RUN/status.json and task.log own lifecycle.
Gate/profile result.json and per-case logs are under their named subdirectories.
Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes dated personal guide under user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical device indices.
Last space check:232GiB data,25GiB root; recheck before large writes.
Core builds:placement-cpu-clean01,placement-npu-clean01 (standalone),
placement-npu-python-clean01 (Python-owned). Never load standalone SDK into Python.
Freeze with `python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT
[--commit REV] [--npu --npu-count N --max-wait 120] -- timeout --signal=TERM
--kill-after=10s 900s '{python}' ...`. Long jobs use background.slice/Nice10.
Never mutate active snapshots or terminal evidence. Atomic handoff:durable_records.py.

## Preserved historical boundaries

The24 earlier dirty files are SHA256-verified on pushed archive/restricted-flow-20260930
at964bf628c67270200dabe55b1bca026bd403cd37; TASK/restricted-flow-archive.json owns hashes.
Historical historical-cpu-attention-01 remains intentionally SIGSTOP; pause.json
outweighs running status. Do not blindly resume/stop. It retains host memory and
TASK/timing.lock. Resolve interrupted timing before formal throughput.
Historical Add CPU78.793172/NPU4 47.932888ms/token is throughput1.6438x faster;
it does not certify resident execution. No complete CPU Attention training ratio.
