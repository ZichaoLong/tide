# Current handoff

Updated 2026-10-01T05:12:56.237804+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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
Current alignment outranks run-ml-experiments: minimal source/input/config/environment
identity, raw failures/results, synchronized complete timing, bounded resources/stops.
No duplicate tracking or Trackio blocker. Training means forward/loss interface/
backward/VJP/optimizer/continuation/throughput; downstream convergence is out of scope.
Use affected-path checks and byte-verified terminal build reuse. Do not rerun the
unchanged portable core's8,954 CPU checks/23 optional skips. Commit implementation,
qualify clean immutable source,commit evidence separately.

## Latest clean qualification

Implementation **8b05c0495fd5bc3cb874f6cb1548b25f3d5ac4b5** is pushed.
FP16 single-device HARD resident inference is qualified:
[report](evidence/resident-fp16-inference-20261001.md),
[audit](evidence/resident-fp16-inference-20261001.json).
Seven immutable-source jobs all PASSED/exit0:
- build-low-precision-inference-clean02:standalone,affected host objects rebuilt,
  six consumers relinked,byte-verified terminal dependencies reused.
- build-low-precision-inference-python-clean01:separate Python-owned runtime.
  Both manifests record reuse;neither is a from-scratch vendor rebuild.
- low-precision-inference-components-clean01:four FP32/FP16 cells,physical3.
- low-precision-inference-regression-clean01:four affected FP32 cells,physical13.
- low-precision-inference-python-clean01:215 passed,zero skips,physical9.
- low-precision-inference-host-clean01:76 passed,212 optional NPU skips.
- low-precision-inference-profile-clean01:two configurations/eight windows,physical9;
  2390 AI_VECTOR_CORE,76 AI_CORE,2 MIX_AIV,no observed AiCPU/logged CPU fallback.
Each selected physical device maps to logical0. Profiling is not throughput.

Complete flow:78 configurations/312 windows per dtype. Attention payload:
64 configurations/192 replays per dtype. Includes all9 LH profiles;public controls
export to payload dtype after FP32 softmax;Read descriptors stay FP32. State,
messages,KV and outputs retain FP16;bias ticks round individually;normalization/
weighted attention merge use FP32. Full/LH minima use actual dtype.
Source/core/binary/loader/raw logs/profiler CSVs and every reused content-archive
member audited by TASK/launchers/precision_inference_evidence.py8b05c04(full hash).

Retained failures:build-low-precision-attention-dev01 missing header;
low-precision-inference-flow-dev03 mismatched minimum dtype;
low-precision-inference-python-dev04 exported control dtype;
build-low-precision-inference-clean01 task-local loader assertion incorrectly
required shared linkage on static components. Each remains FAILED/exit1.
Successful corrected runs do not relabel them. No task job is currently live
except the deliberately paused historical CPU baseline below.

Prior milestones:9f010c9 forward components ([evidence](evidence/resident-fp16-forward-components-20261001.md));
66a6ca5 FP32 event/fiber training ([evidence](evidence/resident-attention-training-20261001.md)).
Only the older Full training trajectory uses conditioned-control comparison;
this increment's event/fiber training is strict. No new speed ratio exists.

## Next action

Commit/push this clean evidence separately. Continue implementation;no requested
pause. Next required integration is FP16 training:FP32 master optimizer,
finite/representability gate before commits,payload-aware parameter publication,
actual-half-forward VJPs and retained-window/checkpoint/control support. Do not
merely remove dtype guards or widen a whole half forward into FP32:recomputed
VJP intermediates must respect forward rounding. Begin with bounded optimizer/
publication components and independent CPU master-update comparisons,then
integrate the complete training owner. FP16 HARD inference remains supported;
FP16 HST/SOFTP and training explicitly refuse until implemented/verified.

Then peer progression/communication/training,five-preset screening,representative
and full-size CPU/mixed/resident comparisons and migration/version/CUDA evidence.
F1–F7 remain incomplete. Use affected-path checks/terminal byte-verified reuse;
never restart all core tests or historical slow timing without a concrete reason.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows;RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service;RUN/status.json and task.log own lifecycle.
Gate/profile result.json/per-case logs are under their named subdirectories.
Module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes dated personal guide under user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical device indices.
Last checked space:236GiB data,25GiB root;recheck before large writes.
Core builds:placement-cpu-clean01,placement-npu-clean01 (standalone),
placement-npu-python-clean01 (Python-owned). Never load standalone SDK into Python.
Freeze with `python TASK/launchers/freeze_run.py --name NAME --snapshot NEW
[--commit REV] [--npu --npu-count N --max-wait 120] -- timeout --signal=TERM
--kill-after=10s 900s '{python}' ...`. Long jobs use background.slice,Nice10,
two build workers,finite bounds. Never mutate active snapshots/terminal evidence.
Use durable_records.py for atomic handoffs.

## Preserved historical boundaries

The24 earlier dirty files are SHA256-verified on pushed archive/restricted-flow-20260930
at964bf628c67270200dabe55b1bca026bd403cd37;TASK/restricted-flow-archive.json owns hashes.
Historical historical-cpu-attention-01 remains intentionally SIGSTOP;pause.json
outweighs running status. Do not blindly resume/stop. It retains host memory and
TASK/timing.lock. Resolve interrupted timing before formal throughput.
Historical Add CPU78.793172/NPU4 47.932888ms/token is throughput1.6438x faster;
it does not certify resident execution. No complete CPU Attention training ratio.
