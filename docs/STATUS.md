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

Implementation **9a84432167a82900c0472b02b2e33ddace6da3eb** is pushed.
[FP32 master/FP16 publication report](evidence/resident-fp16-master-publication-20261001.md),
[audit](evidence/resident-fp16-master-publication-20261001.json).
All six clean qualification jobs PASSED/exit0:
- build-low-precision-publication-clean01: standalone, two publication kernels,
  six host objects/checkers rebuilt; byte-matched terminal dependencies reused.
- build-low-precision-publication-python-clean01: independent Python-owned host
  rebuild/client relink, byte-matched CANN archives reused.
- low-precision-publication-components-clean01: four cells,physical9→logical0.
- low-precision-publication-regression-clean01: FP32 event/fiber,physical3→logical0.
- low-precision-publication-python-clean01:215 passed,zero skips,physical13→logical0.
- low-precision-publication-profile-clean01:two trajectories/ten windows/eight
  updates,physical1→logical0;6718 AI_VECTOR_CORE,288 AI_CORE,5 MIX_AIV; no observed
  AiCPU or logged CPU fallback. This is not a throughput measurement.

Per dtype publication:48 trajectories/240 continued inference windows/192 public
synthetic gradient updates, both schedules/optimizers, six module groups,widths3/33.
FP32 optimizer:32 trajectories/256 updates. FP16:32 trajectories/248 updates/two
independently predicted range refusals. Master65512 publishes finite65504;65520
refuses before any live owner/slot commit. Checkpoint refusal/sub-ULP retention,
None poison, exact payload aliases and normalization-bank rounding passed.
Synthetic-gradient update/publication is not FP16 graph VJP or complete training.
Public FP16 training and HST/SOFTP guards remain. All retained master/publication
failures and their causes are in the report; none were relabelled successful.
Source/core/binary/loader/raw log/CSV and every reused content-archive member were
audited by TASK/launchers/precision_publication_evidence.py9a84432(full hash).
No repeat of the unchanged core gate and no new full-size speed ratio.

Prior:FP16 HARD inference qualified on8b05c04
([report](evidence/resident-fp16-inference-20261001.md)); FP32 event/fiber training
qualified on66a6ca5 ([report](evidence/resident-attention-training-20261001.md)).
The older Full trajectory's conditioned-control policy is separate; this
increment's event/fiber regressions retain strict comparisons.

## Active work and next action

Uncommitted FP16 state and basic Full adjoints:
- State: half parameters/journals, forward-precision EMA coefficients, each Add
  tick rounded; FP32 adjoints. Graph reverse continues to refuse half.
- state-vjp dev03 PASSED both dtypes:216 FP32/FP64 cases;108 quantized-forward/
  FP32-adjoint CPU cases;four actual tapes per dtype;two strict 1024-tick rounding
  anchors. Dev02 FP32 event/fiber training regression PASSED.
- Preserve build-state-vjp-dev01 ambiguous int/int64 initializer failure and
  state-vjp-dev02 pure-half backward comparison failure (about5.05e-5). Oracle
  now matches the FP32-adjoint contract with identical half forward inputs;
  original FP32 and half tolerances unchanged;runtime unchanged in dev03.
- Basic Full: identity/tanh half matmul,bias,tanh recomputation;FP32 adjoints.
  A cancellation-sensitive half rounding anchor has strict FP32 checks. Runtime
  and new fixture/checker not yet built. LH/SwiGLU half adjoints still unavailable.

Basic-vjp-dev01 build FAILED: new fixture omitted at:: on Tensor (header has
no tide::Tensor alias). Fixed fixture declaration;runtime unchanged. Preserve log.
Build-low-precision-basic-vjp-dev02 PASSED. Low-precision-basic-vjp-dev02 PASSED
four cells,physical9→logical0: state216/108 cases and4 actual tapes per dtype;
Full96/48 cases and2 actual tapes per dtype;three strict half rounding anchors.
Low-precision-basic-vjp-regression-dev02 PASSED both FP32 Attention training
cells,physical1→logical0. No task live job except the deliberately paused history.

Next commit/push this coherent state/basic Full increment. Freeze clean full hash
as low-precision-basic-vjp-clean01. Build standalone via
build_precision_basic_vjp.py low-precision-basic-vjp-clean01; Python via
build_precision_basic_vjp_python.py low-precision-basic-vjp-python-clean01.
Both rebuild only affected host/checker objects and use byte-verified terminal
CANN/remaining dependencies; two single-worker host builds may run together.
Then four state/full dtype cells, FP32 event/fiber training, affected Python
resident tests, separate FP16 state/full traces. Commit evidence separately after
the source/binary/archive/loader/log/CSV audit. No global CPU repeat.

After basic state/Full adjoints, complete actual-half-forward LH/SwiGLU/Aggregate/
attention/control adjoints, retained windows, master checkpoint and public FP16
training. Then peer progression/communication/training,five-preset screening,
representative/full-size CPU/mixed/resident performance and version/migration/
CUDA evidence. F1–F7 remain incomplete; authorization active, no requested pause.

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
