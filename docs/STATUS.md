# Current handoff

Updated 2026-10-01T07:08:11.262591+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Implementation **f20c2cc227a400d08c200475be56976ffe4ccc07** committed/pushed.
[FP16 local attention adjoint](evidence/resident-fp16-attention-vjp-20261001.md)
and [audit](evidence/resident-fp16-attention-vjp-20261001.json).
All six fixed-clean-source jobs PASSED/exit0:
- build-low-precision-attention-vjp-clean01:checker rebuilt, byte-matched terminal
  dev01 host/kernel archive reused. Not a full vendor rebuild.
- build-low-precision-attention-vjp-python-clean01:Python-owned attention host
  rebuilt and client relinked; matching CANN archives reused.
- low-precision-attention-vjp-components-clean01:two dtype cells,physical9→logical0.
- low-precision-attention-vjp-regression-clean01:FP32 event/fiber training,physical13.
- low-precision-attention-vjp-python-clean01:61 passed,zero skips,physical9.
- low-precision-attention-vjp-profile-clean01:physical1,1178 AI_VECTOR_CORE/
  94 AI_CORE/24 MIX_AIV. No observed AiCPU/logged CPU fallback. Not throughput.
All selected physical devices map to logical0. Each dtype4 geometries/12 replays;
half adds2 strict QK-rounding fixtures, each with3 replays, physical key tiles1/2.
Ordinary half2e-3/2e-5; unchanged FP32 and anchors2e-5/2e-6. Independent CPU
FP32/FP64 quantized-forward references compare forward and four local gradients.
FP32 training66/172 roots,8/20 trajectories. Complete half graph/public training
remains guarded. Source/core/archive/loader/log/CSV audit passed.

Preserved attention dev01 failure: ordinary cases passed, then fixture sensitivity
check failed. Two QK rounding errors nearly canceled; CPU arithmetic justified
changing one public key from-2.71875 to-2.703125. No candidate/tolerance change.
Build/gate dev02 passed. First audit assumed fixture archive copy, but linker uses
a direct immutable dependency; audit corrected to verify that hash and link path.
No artifact or test result was altered to satisfy the audit.

Prior: **5ce5346**, evidence commit588ed1d, normalized Aggregate/LH/SwiGLU;
[report](evidence/resident-fp16-extended-vjp-20261001.md). All7 jobs passed,6
standalone cells,215 Python cases,2 traces. Aggregate39 cases/117 replays per
dtype; Full90/270 plus3 strict half anchors. Half LayerNorm returns half-rounded
mean/rstd inside FP32 buffers; compute Jacobian statistics separately in FP32
on actual half activation, keep half normalized output for weight gradient.
Raw extended dev02/dev03 failures retained. Earlier d2a1afc state/basic Full,
9a84432 master/publication,8b05c04 HARD half inference,66a6ca5 FP32 full attention
training remain separately qualified. No new full-size speed ratio.

## Active work and next action

Uncommitted FP16 event cache/projection adjoints accept half forward banks/cache,
keep roots/carry/parameter partials FP32 and recompute QKV in actual half before
local half attention. No complete-graph/public guard removed. New event-vjp
checker/CMake target/component map uses independent ContentFlow forward journals,
2 samples/3 events, GQA, widths1/4/7/257,4 adopt/selection/clear variants,
6 proposal/final-cache None/zero/root modes and streaming/greedy.

Development build-low-precision-event-vjp-dev01, event-vjp gate dev01 and
FP32 event/fiber regression dev01 all PASSED/exit0. Both dtypes51 cases/102
replays against CPU FP32/FP64; FP32 regression66/172 roots,8/20 trajectories.
Added isolated cache boundary cases: FP32 sums beyond half range, disconnected
and connected empty/zero roots, poisoned padding, replay reset, length mismatch,
invalid lengths/roots and budget refusal. This is not a retained half graph claim.

Build and gate low-precision-event-vjp-dev04 PASSED/exit0 from frozen
TASK/sources/low-precision-event-vjp-dev04; both dtype cells include6 cache
boundary cases/12 replays/12 refusals. Byte-matched terminal host/kernel reused;
only checker rebuilt. Preserved test-only failures: build dev02 ambiguous empty
Tensor assignment, gate dev03 ATen vector-to-bool factory. Fixed with explicit
Tensor{} and int64-then-bool conversion; no candidate/tolerance change.

This implementation is ready to commit. Next fixed-clean-source builds:
TASK/launchers/build_precision_event_recheck.py low-precision-event-vjp-clean01
and build_precision_event_vjp_python.py low-precision-event-vjp-python-clean01.
Use freeze_run.py --commit NEW_REV --snapshot low-precision-event-vjp-clean01,
900s build bounds; then independent event-vjp, FP32 event/fiber regression,
affected Python precision/event/fiber tests and separate half event profile.
Lease waits max120s; tests/profile600s. Audit using
TASK/launchers/precision_event_vjp_evidence.py NEW_REV, commit evidence separately.
No qualification result is claimed yet. Complete half public training guards
remain, and no new full-size speed ratio is established.

Remaining sequence: fiber cache/projection, control/graph half adjoints,
retained windows, master checkpoint/public FP16 training; then peer progression/
communication/training, five-preset screening, representative/full-size
CPU/mixed/resident performance, version/migration/CUDA records. F1–F7 incomplete.
Authorization remains active; continue after commits without asking to resume.
No subagents or requested pause.

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
