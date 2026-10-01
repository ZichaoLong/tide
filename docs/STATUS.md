# Current handoff

Updated 2026-10-01T08:43:19.404042+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Implementation **3e3397378f8b6545230b3ba543a9b0c72296cc94** committed/pushed.
[FP16 Emit/control report](evidence/resident-fp16-control-20261001.md) and
[audit](evidence/resident-fp16-control-20261001.json). All7 jobs PASSED/exit0:
standalone/Python-owned builds,4 component cells,FP32 control-training regression,
97 Python tests(no skips),two half profiles. Per dtype36 VJP cases/108 replays,
3 forward cases/9 replays,36 complete-flow configurations/144 windows.
FP32 training98 trajectories/1568 windows/392 updates. Profile control2499
AI_VECTOR_CORE/144 MIX_AIV;flow75735 AI_VECTOR_CORE/3042 AI_CORE/36 MIX_AIV.
No observed AiCPU/logged CPU fallback;not throughput. All jobs terminal.
Audit: `python TASK/launchers/precision_control_evidence.py 3e3397378f8b6545230b3ba543a9b0c72296cc94`.
Source/archive/kernel/loader/log/CSV hashes authenticated. Clean snapshot/build
low-precision-control-clean01;Python-owned build low-precision-control-python-clean01.
Retain dev01 missing-checker-object build failure and dev03 FP32 regression failure.
Dev03 incorrectly bypassed ordinary identity Full Emit;restored passing dev02
production bytes and strengthened identity/connected-zero checks. Only boundary
identity bypasses Emit;public cpp/src/full_kernel.cpp owns this behavior.
Prior actual fiber reverse2d7cee1/evidence60f13da remains qualified separately.

## Active work and next action

FP16 control evidence committed/pushed as e77cd41. Current implementation:
actual half complete-graph reverse and retained-window sum/identity/EMA/Add/tanh
integration,including HARD/HST/SOFTP. CPU-only precision_graph_fixture adapts
independent Streaming with real half forward rounding and FP32/FP64 leaves;
never feeds candidate events/results. Actual source/delivery payloads stay half,
journals/cotangents/parameter alias accumulation stay FP32. Retained tests cover
four windows including empty continuation,after-close/overwritten-live journals,
aliases,connected-zero/None,physical message identities and byte accounting.
Public training/master/checkpoint guards remain until full integration.

Dev04 build and all4 graph/retained cells PASSED/exit0. Each dtype122 graph windows
and42 retained trajectories/168 windows;replay,all three Read coordinates and mixed
linear/norm,HARD/HST/SOFTP,FP32/FP64 independent references. Dev02 standalone/Python
builds,fiber reverse both dtypes,FP32 control training98 trajectories/1568 windows/
392 updates and97 Python tests all PASSED/exit0. Only checker sources changed
since dev02 production/oracle build. Dev03 intermediate component gate also passed.
Retain dev01 launcher failure:old CMake template lacked new fiber-reverse target;
production rebuilt in dev02. Dev02 half gate failed before execution because
checkers omitted allow_npu_float16;fixed explicitly,not a changed runtime default.

Next commit/push implementation;freeze low-precision-graph-clean01 at that revision.
Clean build launchers:build_precision_graph_v3.py(standalone,source-verified dev02
production/oracle reuse,two checker workers) and build_precision_graph_python_v1.py
(Python-owned graph reverse rebuild);900s. Then graph-vjp/retained all4 cells,
fiber-reverse both dtypes/control-training FP32,97 Python tests,two half profiles
(graph-vjp,retained;512MB each),600s/lease120s. Audit
`python TASK/launchers/precision_graph_evidence.py REV` expects7 terminal jobs,
7 standalone cells and97 Python tests. Raw logs: TASK/runs/NAME/status.json and
 task.log; unit tide-execution-flows-NAME.service. Commit evidence separately.
All current development jobs terminal;historical baseline remains intentionally paused.
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
