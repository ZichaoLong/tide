# Current handoff

Updated 2026-10-01; attention cache qualification complete, public FP16 training active. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Attention graph/retained cache reverse is qualified on
f26f3b07f5e24a2ea835dbbfe352f3acf478eb57; evidence committed/pushed as d8cb3ae.
[Report](evidence/resident-fp16-cache-graph-20261001.md) and
[audit](evidence/resident-fp16-cache-graph-20261001.json): all5 immutable jobs
PASSED/exit0,10 cells. Per dtype:event66 trajectories/264 windows;
fiber/mixed152/608;6 base/extended regressions passed. Two-case half profile:
9557 AI_VECTOR_CORE/430 AI_CORE/156 MIX_AIV; no observed AiCPU/logged fallback.
Production unchanged from25e996c. No throughput conclusion.
Prior [extended graph evidence](evidence/resident-fp16-extended-graph-20261001.md)
qualifies db9e985:110 trajectories/440 windows per dtype,6 cells,4 terminal jobs.
The corresponding task-local audit scripts passed. Earlier qualification is
indexed in ROADMAP; no earlier suite needs a mechanical rerun.

## Active work and next action

Public FP16 training implementation is ready for its clean qualification.
Four public C++ training objects now accept FP16 payload owners, require FP32
roots, retain FP32 masters/slots and export half named checkpoint parameters.
Restore checks exact master-to-payload correspondence. Python exposes the same
boundary; this is a C++/CANN client, not an independent Python device scheduler.
Tests keep actual half forward rounding, independent CPU FP32/FP64 adjoints,
separate CPU FP32 masters, None/zero distinction and checkpoint continuation.

Development gates passed:basic43 trajectories/688 windows/172 updates;
cache40/640/160;FP32 control98/1568/392,event66 roots/8 trajectories,
fiber172 roots/20 trajectories. Latest basic dev03 verifies nonfinite-root
refusal through the public API without mutating returned gradient views.
Python dev03 passed36 lifecycle+3 fresh-process cases;dev05 passed2 independent
scalar autograd/master cases+97 existing affected cases (99, no skips).
All development jobs terminal. Do not claim immutable public half qualification yet.
Retained failures:dev01 build missing checker object;Python dev02 missing explicit
native library path;dev03 unsupported identity memory fixture;dev04 unsupported
ordinary identity Full fixture. Final fixture uses public Node(identity=True).
No tolerance relaxation or production from failed builds reused.

Next:commit/push this implementation, freeze low-precision-training-clean01 at
its40-digit revision, and qualify only affected paths. Build launchers:
TASK/launchers/build_precision_training_v2.py low-precision-training-clean01;
TASK/launchers/build_precision_training_python_v1.py low-precision-training-python-clean01.
Standalone parent low-precision-cache-graph-clean01;Python-owned parent
low-precision-graph-python-clean01. Rebuild4 public objects and affected checkers;
reuse byte-matched terminal CANN/core, without a vendor rebuild.
Seven planned jobs:build-low-precision-training-clean01,
build-low-precision-training-python-clean01, and low-precision-training-
{basic,cache,regression,python,profile}-clean01. Each uses the same clean snapshot.
Basic/cache select half-training/half-cache-training;regression selects
control-training,event-training,fiber-training. Python runs
 test_resident_half_training.py,test_resident_precision.py,
 test_resident_event_training.py,test_resident_fiber_training.py (138 expected).
Profile half-cache-training --dtype float16 --storage-limit-mb512
 --application-arg=--profile-smoke:one trajectory/16 windows/4 updates,placement only.
NPU lease120s,run600s,build900s. Terminal audit:
`python TASK/launchers/precision_training_evidence.py FULL_REV`.
Then write the report/update backlog and commit/push evidence separately.

Remaining main work after public half qualification:device peer progression,
completion/communication and training;five-preset consumer screening;representative
and full-size performance;version/migration/CUDA records. F1–F7 remain incomplete.
Historical CPU Attention stays paused and does not block implementation.
No pause requested;continue after qualification and each authorized push.

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
