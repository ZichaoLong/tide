# Current handoff

Updated 2026-10-01T08:47:25.498737+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Implementation **25e996c6a55b56ede9b689c94b8085ff45a56908** committed/pushed.
[FP16 graph/retained reverse report](evidence/resident-fp16-graph-reverse-20261001.md)
and [audit](evidence/resident-fp16-graph-reverse-20261001.json). All7 fixed-source
jobs PASSED/exit0:standalone/Python builds,4 graph/retained cells,3 regression
cells,97 Python tests(no skips),2 half profiles. Each dtype122 graph windows,
42 retained trajectories/168 windows,FP32/FP64 references,replay. Sum Aggregate,
identity/EMA/Add state,identity/tanh Full,HARD/HST/SOFTP,all Read coordinates and
mixed norm/linear,feedback,aliases,after-close/poisoned-live tapes,None/zero.
Half roots×256;VJP rtol2e-3/atol2e-5;FP32 unchanged. Graph profile124237
AI_VECTOR_CORE/1941 AI_CORE/1176 MIX_AIV;retained95158/1666/1656 respectively.
No observed AiCPU/logged CPU fallback;placement only,not throughput.
Fiber regression90 cases/180 replays per dtype;FP32 control training98 trajectories/
1568 windows/392 updates. All7 jobs terminal. Source/archive/object/kernel/loader/
log/CSV hashes authenticated by TASK/launchers/precision_graph_evidence.py REV.
Clean snapshot/build low-precision-graph-clean01;Python-owned build
low-precision-graph-python-clean01. Retain dev01 missing-template build failure;
dev02 half gate refused at CLI before execution due missing explicit
allow_npu_float16;fixed in checker only. Dev04 strengthened Read coverage and passed.
No production from failed build reused,no tolerance relaxation.
Prior control3e33973/evidencee77cd41 and actual fiber2d7cee1/evidence60f13da
remain separately qualified.

## Active work and next action

Commit/push current graph/retained evidence,then extend the same independent
quantized CPU reference and retained checker to normalized Aggregate,LH and
SwiGLU,then event/fiber cache graph integration. Existing graph_vjp accepts half;
local module adjoints are qualified but whole-graph coverage is currently only
the base profile above. Public training/master/checkpoint guards remain.
Next production/test code not yet edited. precision_graph_fixture.cpp contains
CPU-only HalfState/HalfSum/HalfFull and physical transport rounding;
retained_check.cpp already checks all named owners/aliases,initial and physical
boundary gradients. Reuse these checks;do not duplicate schedulers or CPU core.
Normalized Aggregate must round raw source product to half before multiplying
FP32 coefficients,then accumulate unrounded coefficient products in FP32;sum's
source products are not rounded before accumulation. LH uses standalone
normalization→half→weight multiply→half→optional bias→half. SwiGLU uses actual
half matmuls/SiLU/product/residual. Local independent references exist in
extra_full_check.cpp and aggregate_vjp_check.cpp. Internal graph roots/adjoints
always FP32,including retained roots;do not recompute whole forward in FP32.
Training future:training_backward.cpp must accept FP32 cotangents independent
of payload dtype;training_parameters.cpp and training_owner.cpp still FP32-only,
master/named half correspondence and checkpoint validation need explicit handling.
No portable core changes;do not repeat unchanged8,954 CPU checks.
Remaining F1–F7:all-module half graph/public training/master/checkpoint integrations,
device peer progression/communication/training,five-preset screening,representative/
full-size CPU/mixed/resident performance,version/migration/CUDA records.
Implementation commit→immutable qualification→separate evidence commit,push each.

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
