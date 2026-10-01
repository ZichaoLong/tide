# Current handoff

Updated 2026-10-01T05:41:00.069455+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Pushed implementation 8b05c0495fd5bc3cb874f6cb1548b25f3d5ac4b5 and evidence b645e9a:
[FP16 inference report](evidence/resident-fp16-inference-20261001.md),
[audit](evidence/resident-fp16-inference-20261001.json).
Seven immutable-source jobs passed: byte-verified standalone/Python build reuse,
eight standalone cells, 215 Python device cases (zero skips), 76 host cases
(212 optional NPU skips), and separate FP16 trace (2390 AI_VECTOR_CORE,76 AI_CORE,
2 MIX_AIV; no observed AiCPU or logged CPU fallback). Full flow covers 78
configurations/312 windows per dtype; attention payload 64 configurations/192
replays per dtype. Only single-device FP16 HARD inference is qualified.
The event/fiber FP32 training regressions retain strict comparisons. Profiling
is not throughput; no new speed ratio exists. Original failures remain recorded.

## Current implementation increment

FP32 master optimizer plus FP16 payload publication is development-tested; clean
qualification is next. Public FP16 training and HST/SOFTP guards remain in place.
DeviceOptimizer preserves FP32 master/gradient/slots, rounds only at payload
publication, and refuses a nonfinite half proposal before committing any live
owner/slot/counter. Restore validates half representability. A 65512 master
rounds to finite 65504; 65520 refuses. Sub-ULP master progress is retained.
Fused publication updates ordinary/event/fiber banks and HARD Read aliases;
FP32 normalization banks round to payload dtype before widening.

Terminal development jobs (TASK/runs/NAME):
- build-low-precision-master-dev01 and dev02: PASSED.
- low-precision-master-dev02: PASSED, FP32 32 trajectories/256 updates; FP16
  32 trajectories/248 updates/two independently predicted half-range refusals.
- low-precision-master-regression-dev01: PASSED FP32 event/fiber training.
- build-low-precision-publication-dev02: PASSED, two publication kernels/six
  host objects rebuilt; byte-matched terminal master/inference dependencies reused.
- build-low-precision-publication-dev03: PASSED, checker only rebuilt.
- low-precision-publication-dev03: PASSED, four cells, physical9→logical0;
  each dtype publication has 48 trajectories/240 continued inference windows/
  192 public synthetic gradient updates, both schedules/optimizers, six module
  groups, widths3/33, exact quantized aliases and sticky-error no-write checks.
- low-precision-publication-regression-dev02: PASSED, physical3→logical0;
  FP32 event/fiber complete training:66/172 roots and8/20 trajectories.

These synthetic-gradient update/publication checks are not graph VJP or full
FP16 training evidence. CPU master updates and Streaming run independently.
Preserved new failures:
- low-precision-master-dev01: old test incorrectly required every finite FP32
  AdamW result to fit half; CPU-predicted refusal checks replaced that assertion.
- build-low-precision-publication-dev01: const checker registry failed compilation.
- low-precision-publication-dev02: checker used stride3 for a sample with2 inputs;
  corrected per-port continuation counts. Runtime validation remained unchanged.
No new task job is live; historical CPU baseline below remains deliberately paused.

## Next action

Commit/push this coherent implementation, then freeze its clean full hash as
low-precision-publication-clean01. Build standalone via
TASK/launchers/build_precision_publication_v2.py low-precision-publication-clean01;
Python-owned runtime via build_precision_publication_python.py
low-precision-publication-python-clean01. Both 900s, max2 build workers.
The latter reuses byte-identical publication kernels from terminal dev02,
optimizer kernels from terminal master-dev01, and remaining qualified dependencies.
Run standalone optimizer/master-publication and FP32 event/fiber training gates,
Python affected resident tests, and separate master-publication FP16 profile-smoke.
Commit reviewed evidence separately after source/binary/loader/log/CSV audit.

Then implement actual-half-forward VJPs, retained windows, master checkpoint,
HST/SOFTP and public complete FP16 training. Do not merely remove dtype guards
or widen half forward into FP32: recomputed adjoints must respect each actual
forward rounding point. Next peer progression/communication/training, five-preset
screening, representative/full-size CPU/mixed/resident performance and version/
migration/CUDA evidence. F1–F7 remain incomplete.

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
