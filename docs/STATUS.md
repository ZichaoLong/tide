# Current handoff

Updated 2026-10-01T06:54:09.846907+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Commit/push reviewed attention evidence separately. No new active NPU/build jobs;
the historical CPU Attention job is intentionally paused as recorded below.
Continue implementing FP16 event/fiber cache and projection adjoints, control/
graph reverse, retained windows, FP32-master checkpoint/public training. Never
remove whole-graph/public guards merely because local components passed.

Investigation for the next increment:
- event_reverse.cpp rejects half banks/cache and currently allocates cache
  cotangents with payload dtype. Keep cache roots/carry/parameter partials FP32.
- tide_event_reverse_pack.cpp needs half parameter loads, followed by actual half
  QKV projection recomputation. Saved cache journal is already widened FP32.
- Local attention now accepts half Q/K/V and returns payload-rounded output
  widened to FP32 for output-projection gradients. Its internal softmax correction
  stays unrounded FP32. Do not feed it all-FP32 QK recomputation for a half graph.
- ContentFlow::parameter_banks().attention exposes actual forward EventAttentionTape
  even for half; state_tape/full_tape also work. This permits independent local
  cache/projection tests while reverse_tape() keeps its complete-graph guard.
- complete()/merge() cache helpers must preserve missing roots and zero gradients;
  arbitrary poisoned padding stays unread. Bridges/eviction/adoption/clear need
  explicit tests, not only a projection formula check.

Then peer progression/communication/training, five-preset screening,
representative/full-size CPU/mixed/resident performance and version/migration/CUDA
records. F1–F7 incomplete. Authorization remains active; continue after commits
without asking to resume. No subagents or requested pause.

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
