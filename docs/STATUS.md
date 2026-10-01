# Current handoff

Updated 2026-10-01; extended-graph qualification complete, attention integration next. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Extended graph qualification on db9e985e9d7163563ed8877be28b5f8148f490fd is
complete:all4 immutable jobs PASSED/exit0. [Report](evidence/resident-fp16-extended-graph-20261001.md)
and [audit](evidence/resident-fp16-extended-graph-20261001.json). Each dtype110
extended trajectories/440 windows,122 base graph windows,42 base retained
trajectories/168 windows. All6 cells passed. Half extended profile254571
AI_VECTOR_CORE/2391 AI_CORE/6057 MIX_AIV,no observed AiCPU/logged fallback.
Production unchanged from25e996c;only oracle/retained checker rebuilt. Audit
TASK/launchers/precision_extended_graph_evidence.py REV passed. All current
qualification jobs terminal. No new full-size throughput conclusion.
Retain dev01 insufficient128MiB wide SwiGLU reverse budget;explicit256MiB used
for width257. Retain dev02 CPU FP32 LayerNorm cancellation failure;half oracle
keeps FP32→half forward and native CPU FP64 normalization derivative,with
analytic-zero anchor. No production/tolerance change.

Attention cache integration implemented;dev02 build and event/fiber gates all
PASSED/exit0. Per dtype:event66 trajectories/264 windows,fiber152/608;independent
CPU FP32/FP64 references,shared owners,feedback,HARD/HST/SOFTP,both schedules,
GQA/eviction,all five pools,mixed modules,periodic clocks,widths1/4/257,None/zero,
poisoned live journals/padding,after-close reverse and exact cache replay.
New precision_cache_fixture.cpp and retained_cache_fixture.cpp;existing retained
checker shares graph/parameter/boundary checks. Production source unchanged;
fixture CMake extended. Dev01 failed only on test-helper matmul ADL ambiguity;
renamed half_matmul in dev02,no failed production artifact reused.

Next commit/push implementation,freeze low-precision-cache-graph-clean01.
Build TASK/launchers/build_precision_cache_graph_v1.py SNAPSHOT(900s),then separate
bounded event/fiber/regression checks(600s,lease120s). Regression checks graph-vjp,
retained,extended-retained. One half fiber-retained --profile-smoke placement
trace(2 trajectories:mixed HST/periodic SOFTP;512MB). Audit
TASK/launchers/precision_cache_graph_evidence.py REV expects5 terminal jobs,
10 standalone cells and the separate2-case profile. Commit/push evidence separately.
No repeat Python-owned build/client tests or portable core for test-only source.
All current development jobs terminal;historical CPU baseline remains paused.
Current public training remains guarded. training_backward.cpp must accept FP32
roots independently of payload dtype;freeze_model/restore and checkpoint
master↔named half correspondence need explicit handling before lifting guards.
No portable core changes;do not repeat unchanged8,954 CPU checks or97 Python
client tests for test-only source changes.
Remaining F1–F7:half cache/public training/master/checkpoint,device peer progression/
communication/training,five-preset screening,representative/full-size performance,
version/migration/CUDA records. Commit implementation→immutable qualification→
separate evidence commit;push each. No pause requested.

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
