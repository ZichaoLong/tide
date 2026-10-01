# Current handoff

Updated 2026-10-01T04:48:52.388029+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Pushed implementation **9f010c9cf41d6077df90cb8228b8c250663a333b** adds FP16 state/Read,
SwiGLU/emission,normalized Aggregate and the single selected CMake check target.
Clean qualification/audit is complete:
[report](evidence/resident-fp16-forward-components-20261001.md),
[audit](evidence/resident-fp16-forward-components-20261001.json).
Snapshot:low-precision-forward-clean01. All12 jobs terminal PASSED/exit0:
- build-low-precision-forward-clean01:normal full standalone build,5 CTests,loader.
- build-low-precision-forward-python-clean01:separate Python-owned build,loader.
- low-precision-forward-{components,windows,adjoints,updates}-clean01:
  all62 registered cells split23/20/13/6;physical9/1/3/13->logical0.
- low-precision-forward-python-clean01:196 NPU cases,zero skips,physical9.
- low-precision-forward-host-clean01:76 interface tests,193 optional NPU skips.
- low-precision-{state,aggregate,swiglu,emission}-profile-clean02:four separate FP16
  traces,256MB/180s each. State98 vector;Aggregate4736 vector+128 MIX_AIV;
  SwiGLU326 vector+40 AI_CORE;emission665 vector+27 AI_CORE. No observed AiCPU or
  logged CPU fallback. Construction/CPU assertions included;not throughput.
State/Read48 configurations/336 windows/18 refusals per dtype;Aggregate64/192;
SwiGLU16 components/4 refusals and256 FP32 windows;emission16/6 and66 FP32 windows.
Only the existing Full training regression uses conditioned controls:13 frames,
max4.470348358154297e-6. Other training controls remain strict. AdamW trajectories
keep eps1e-5;public default eps1e-8/normalization epsilons unchanged.
Audit launcher:TASK/launchers/low_precision_forward_evidence.py.

Prior milestones remain scoped:66a6ca5 complete FP32 event/fiber attention training
([evidence](evidence/resident-attention-training-20261001.md));466b4c3 FP16 numerical/
Full/LH/sum components ([evidence](evidence/resident-fp16-components-20261001.md)).
The clean9f010c9 qualification does NOT enable complete resident FP16 sessions.
Raw state-dev01 FAILED/exit1 (implicit half Read operand) and state-dev02
CANCELLED/exit143 (repeated vendor builds) remain preserved. Later dev03 passed.

## Current uncommitted development and next action

Uncommitted production/tests implement FP16 tiled/dense attention with FP32
normalization/weighted accumulation, half event/fiber KV, per-tick half bias
rounding, widened diagnostic journals and HARD public inference sessions.
New checks:attention-payload,precision-flow,test_resident_precision.py.
FP16 HST/SOFTP and training explicitly refuse; FP32 master publication/VJP remain.
These changes are UNVERIFIED, not support promotion. Docs/evidence above are for
committed9f010c9 only and may be committed separately from this working code.

build-low-precision-attention-dev01 FAILED/exit1:missing grad_mode include in new
attention helper;fixed,raw failure preserved.
build-low-precision-inference-dev02 RUNNING from immutable dirty snapshot
low-precision-inference-dev02;build same suffix. Uses byte-verified terminal CANN
archives,two workers,900s;task-local launcher build_precision_overlay_v4.py.
Do not edit its source/build. It targets attention-payload,precision-flow,
attention-tile,event/fiber training and public resident check.
**Do not run its precision-flow checker:** review caught the large-int64 Add
fixture would ask the CPU oracle to replay2^55 empty ticks. Working source now
initializes all Add owner clocks at the common cut. This is a test-only correction,
not a runtime shortcut. Other dev02 binaries remain eligible after build success.
Next freeze low-precision-inference-dev03 and compile only the corrected checker
against terminal dev02 libraries after byte-comparing all other component inputs;
record the reuse and source identity. Then run affected FP32/FP16 gates.
Python build launcher prepared:TASK/launchers/build_precision_python_overlay.py;
it requires completed dev02 and clean forward builds,targets _tide_resident,
uses Python-owned core placement-npu-python-clean01 and verified CANN archives.
Run new19 FP16 client checks plus affected existing resident tests after build.
No new FP16 inference or throughput result exists yet.

Continue FP16 VJP/master publication/control modes after inference closure;
then peer progression/communication/training,five-preset screening,full-size
CPU/mixed/resident comparisons and migration/version/CUDA evidence. Do not stop at
this component increment or revive the historical slow timing as the main task.

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
