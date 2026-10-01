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

Implementation **d2a1afccac85c1e53afe45968d6ff0d5f256ce09** is committed/pushed.
[State/basic Full half-adjoint report](evidence/resident-fp16-basic-vjp-20261001.md),
[audit](evidence/resident-fp16-basic-vjp-20261001.json).
All seven clean-source jobs PASSED/exit0:
- build-low-precision-basic-vjp-clean01: affected Full host/fixtures/checkers
  rebuilt; terminal source-byte-matched state kernel/other dependencies reused.
- build-low-precision-basic-vjp-python-clean01: three affected Python-owned host
  objects rebuilt; matching CANN archives reused; client relinked.
- low-precision-basic-vjp-components-clean01:four cells,physical1→logical0.
- low-precision-basic-vjp-regression-clean01:two FP32 cells,physical3→logical0.
- low-precision-basic-vjp-python-clean01:215 passed,zero skips,physical9→logical0.
- low-precision-state-vjp-profile-clean01:physical13→logical0,5614 AI_VECTOR_CORE.
- low-precision-full-vjp-profile-clean01:physical11→logical0,5424 AI_VECTOR_CORE,
  162 AI_CORE. No observed AiCPU/logged CPU fallback. Profiling is not throughput.

State:216 CPU FP32/FP64 cases;108 half quantized-forward/FP32-adjoint cases;
four real forward tapes per dtype;two strict1024-tick rounding anchors.
Identity/tanh Full:96/48 cases,each with19/7/0-row replay;two real tapes per dtype;
one strict matmul/bias/tanh rounding anchor. Both retain None/zero/poison and
exact discrete checks. Half tolerances2e-3/2e-5; FP32/FP64 1e-5/1e-6; anchors strict.
Whole-graph half reverse and public training guards remain. Event/fiber FP32
training regressions retain strict comparisons. No new full-size speed ratio.
Audit: TASK/launchers/precision_basic_vjp_evidence.py d2a1afc(full hash),including
source/core/binary/loader/CSV/logs and every reused content/fixture archive member.

Preserved failures:build-low-precision-state-vjp-dev01 ambiguous int/int64 tensor
initializer;low-precision-state-vjp-dev02 CPU pure-half backward reference mismatch
(max about5.05e-5),corrected to the declared quantized-forward/FP32-adjoint oracle
without changing forward inputs/runtime/tolerances;build-low-precision-basic-vjp-dev01
missing at::Tensor in fixture. Raw failures remain in TASK/runs.

Prior clean qualifications:
- 9a84432 master/publication,[report](evidence/resident-fp16-master-publication-20261001.md).
  Per dtype publication48 trajectories/240 windows/192 synthetic gradient calls;
  FP32 optimizer256 updates,half248 updates/two predicted range refusals.
  Master65512 publishes finite65504;65520 refuses transactionally. Not full training.
- 8b05c04 FP16 HARD inference,[report](evidence/resident-fp16-inference-20261001.md).
- 66a6ca5 FP32 event/fiber training,[report](evidence/resident-attention-training-20261001.md).
The older Full trajectory's conditioned-control policy remains separate.

## Active work and next action

Uncommitted next increment: normalized Aggregate FP16 local adjoint. Host
validates half physical source scales; kernel reconstructs the source product
with half rounding before coefficient differentiation. Normalization and
message/scale/coefficient adjoints remain FP32. New39-case/117-replay half cell
uses non-dyadic payload fixtures and independent quantized-forward CPU FP32/FP64
references at unchanged1e-5/1e-6. Whole-graph reverse remains guarded.

Active unit: tide-execution-flows-build-low-precision-aggregate-vjp-dev01.service.
Snapshot/build: TASK/{sources,builds}/low-precision-aggregate-vjp-dev01.
Launcher: TASK/launchers/build_precision_aggregate_vjp.py,900s/two workers.
One payload kernel/host/checker rebuilt; remaining terminal basic-VJP dependencies
byte-verified. Inspect TASK/runs/build-low-precision-aggregate-vjp-dev01/{status.json,task.log}.
On success run verify_device_control.py --checks aggregate-vjp and separately
--checks event-training fiber-training, frozen snapshot/build,max120s lease wait,
600s execution. Preserve failures,do not loosen established FP32 comparisons.

Commit the completed basic-VJP evidence separately (only docs) and push.
Continue Aggregate development,then LH/SwiGLU,attention/control/graph adjoints,
retained windows,master checkpoint/public FP16 training. Do not just remove guards
or recompute an entire half forward in FP32. Use actual rounded saved operands.
Then peer progression/communication/training,five-preset screening,representative/
full-size CPU/mixed/resident performance and version/migration/CUDA evidence.
F1–F7 remain incomplete. Authorization active; no requested pause.

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

