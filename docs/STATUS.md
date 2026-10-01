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

Uncommitted FP16 state-VJP component work:
- state_tape accepts half forward banks/journals; graph reverse still refuses half.
- EMA reads coefficients in forward precision; Add replay rounds every literal
  forward tick. Cotangents/adjoints stay FP32. No FP32 whole-forward substitution.
- state-vjp adds108 CPU half autograd cases, four actual half tapes and two
  1024-tick rounding-sensitive lifted-autograd anchors at strict FP32 tolerance.
  FP32/FP64 cases/tolerances remain. Not yet runtime-qualified.
- build-low-precision-state-vjp-dev01 FAILED in checker compilation: ambiguous
  mixed int/int64 tensor initializer. Fixed with explicit vector<Index>; preserve log.
- build-low-precision-state-vjp-dev02 is RUNNING, frozen dirty snapshot/build
  low-precision-state-vjp-dev02, launcher build_precision_state_vjp.py,900s/two workers.
  One state reverse kernel,two host objects/checker rebuilt; other dependencies
  byte-verified against terminal publication clean01. Inspect status/task.log.

Commit the master/publication evidence separately (only docs). On state build
success run verify_device_control.py --checks state-vjp and separately
--checks event-training fiber-training from that frozen snapshot/build, each
max120s lease wait/600s execution. Fix actual failures without relaxing FP32
checks. Then commit state implementation, qualify clean fixed source, profile
separately and commit reviewed evidence. Do not reopen global CPU qualification.

After state adjoints, complete the remaining actual-half-forward Full/Aggregate/
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
