# Current handoff

Updated 2026-10-01T08:25:27.730825+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Commit/push current evidence,then implement actual half complete-graph reverse,
retained windows,and FP32 master/checkpoint/public FP16 training in tested increments.
Local half state/Full/normalized Aggregate/attention/event/fiber/control adjoints
are qualified;graph_vjp.cpp and public training still explicitly reject half.
Journals/roots/adjoints stay FP32;forward payload/parameters/cache stay real half.
Do not widen the entire half forward to FP32 to bypass integration. Inspect
normalized Aggregate gradient allocation and public roots/owner master handling.
For graph reference,use independent scheduling and declared per-operation rounding;
sum Aggregate accumulates FP32 source products and casts only the final content,
not each contribution before the sum. Preserve None/zero and physical identities.
Update fiber_reverse_check's old empty-root guard test when enabling graph VJP.
Next production code not yet edited. No task runtime jobs active beyond historical
intentionally paused baseline. Remaining F1–F7:half integrations,device peer
progression/communication/training,five-preset screening,representative/full-size
CPU/mixed/resident performance,version/migration/CUDA records.
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
