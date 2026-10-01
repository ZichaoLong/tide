# Current handoff

Updated 2026-10-01T07:18:06.338093+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Implementation **4f195d22961d66495b30ecc447ececf684eac969** committed/pushed.
[FP16 event cache/projection report](evidence/resident-fp16-event-vjp-20261001.md)
and [audit](evidence/resident-fp16-event-vjp-20261001.json).
All6 fixed-clean-source jobs PASSED/exit0: standalone/Python builds,event-vjp
both dtypes,FP32 event/fiber training,61 Python cases(no skips),half profile.
Each dtype51 cases/102 replays plus6 cache boundary cases/12 replays/12 refusals.
Actual independent forward journals; CPU FP32/FP64 quantized-forward reference.
None/zero,large int64,adopt/clear/window/GQA,poison padding,FP32 cache sums beyond
half range and mismatched lengths covered. Half rtol2e-3/atol2e-5,original FP32
1e-5/1e-6. Regression66/172 roots,8/20 trajectories. Profile52140 AI_VECTOR_CORE,
2088 AI_CORE,450 MIX_AIV; no observed AiCPU/logged CPU fallback,not throughput.
Runtime jobs physical1/3/9/13→logical0. Source/archive/loader/log/CSV audit passed.
Standalone checker rebuilt with byte-matched terminal host/kernel reuse;
Python-owned event host rebuilt/client relinked. No portable-core changes.

Preserved test-scaffold failures: build dev02 ambiguous empty Tensor assignment,
gate dev03 unsupported vector-to-bool factory; fixed without algorithm/tolerance
changes. Dev04 passed. Early audit refused live regression; terminal audit passed.
Previous local attention implementation f20c2cc/evidence63824ba remains qualified;
normalized Aggregate/LH/SwiGLU5ce5346/evidence588ed1d. See their evidence,not new
complete graph/public FP16 training claims. No full-size speed ratio changed.

## Active work and next action

Event evidence is ready for its separate commit. Uncommitted next increment:
FP16 local same-fiber VJP in fiber_vjp.cpp and four fiber_vjp AscendC kernels.
All adjoints remain FP32. QKV matmul/bias,Q scaling before QK,completed query
outputs and final pooling must reproduce half rounding. Mean pooling sums
before dividing; cache bias retains actual half tick rounding. Independent
CPU quantized-forward checks still need updating. No device result yet for fiber.
Do not claim integration or remove complete-graph/public guards.

Next: finish local fiber checker, freeze bounded development build and run
fiber-vjp two dtype cells plus affected FP32 event/fiber training. Use max120s
lease waits,900s build/600s gate bounds. Then implementation commit,clean immutable
standalone/Python checks,separate profile,evidence commit. Continue autonomously.

Remaining: fiber cache/projection integration,control/graph half adjoints,
retained windows,master checkpoint/public FP16 training; then peer progression/
communication/training,five-preset screening,representative/full-size
CPU/mixed/resident performance,version/migration/CUDA records. F1–F7 incomplete.
No subagents or requested pause. Authorization remains active after commits.

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
