# Current handoff

Updated 2026-09-30T23:35:23.136021+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository `/home/zlong/llm/graph-execution-foundation`,real path
`/var/tmp/zlong-graph-execution-foundation/repository`;branch `graph-execution-foundation`.
HEAD **0ba9fc6**, pushed; normalized Aggregate implementation committed. LH/SwiGLU5143de3 is formally qualified. No authorization pending. No subagents.
Reference repositories and ObsidianVault remain read-only. **F1–F7 are incomplete.**

## Contract and order

[execution-flows.md](execution-flows.md) is the contract;[ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Current user alignment outranks run-ml-experiments: retain
source/input/config/environment identity,raw results/failures,synchronized complete
timing,bounded resources/stops. Trackio never blocks implementation.
Training means independent forward/backward/VJP/optimizer/continuation and complete
throughput. Model convergence is a later consumer experiment.

Candidates consume their own inputs,state and parameters: no CPU numerical routing
prepass. General online greedy accepts legal arbitrary topology/input,including
positive-delay PDG feedback,and may degenerate to streaming. Preserve int64,
stable order,duplicate edges,missing/zero and None/zero. CPU FP64/FP32 stay independent.
Residence includes actual online decisions. PDG LibTorch;TimedDAG/Settle LibTorch
and PyTorch;CPU/NPU × streaming/prefill × inference/complete training. Five presets
and fine switches;FP32 main,FP16 separate;three processes for recommendations.
Prioritize semantic/ownership/public/device closure,peer progression and resident
training,then representative/full-size performance. CUDA execution is target-pending.

## Completed and qualified

All earlier forward/placement/VJP evidence and exact commits are indexed by ROADMAP.
Do not rerun the unchanged portable core's **8,954 CPU tests/23 optional skips**.

- Placement d412541 qualified;public resident inference740fa87 and disk restore622dbb2
  qualified. Public package is separate from the portable core.
- State VJP3e2d54d,identity/tanh Full3285b13,graph reverse37430e1,parameter alias
  reduction1ef23f3,device optimizer/publication3b31ee2 and retained windowsc2423f0
  have immutable full component gates and separate profiles. Reports in docs/evidence.
- Public C++ training591e907 qualified and evidence pushed243cdb3:
  docs/evidence/public-resident-training-20261001. Four CTests,42 component cells,
  18 CPU FP32/FP64 trajectories/288 windows/72 updates;installed consumer3 inference
  and3 training windows/retained backward/optimizer restore. Profile107,345 AIV+
  1,857 AI_CORE+1,673 MIX_AIV,304 optimizer records,no AiCPU/fallback. Checker scope.
- Python client0f363b8 qualified and evidence pushed9022636:
  docs/evidence/python-resident-training-20261001. Clean Python-owned build,
  76 CPU interface tests/37 optional skips,40 device cases,real NPU loss cotangents,
  three families/both schedules/SGD+AdamW,aliases,None/zero,disk and new-process
  continuation. One PDG greedy AdamW profile1,569 AIV+14 AI_CORE+42 MIX_AIV;
  optimizer12/graph_reverse66/window_bridge6/ready_pack12,no AiCPU/fallback.

Those training qualifications cover single-NPU FP32 HARD,sum/broadcast,
identity/EMA/Add state and identity/tanh Full. Python is a C++/CANN client,
not an independently qualified pure-PyTorch resident scheduler. These profiles
are not throughput. Updated device parameters export through checkpoints;the
caller's original model is a frozen construction template. Step/detach explicitly
truncate generations. All trainable aliases matter;an explicit empty optimizer
group selects no parameters. Do not regress these ownership details.

## Latest qualification and current increment

LH/SwiGLU implementation5143de3 is committed/pushed and now formally qualified.
See docs/evidence/resident-extra-full-20261001.{md,json}: two clean builds,five
standalone CTests,all44 component cells,90 local configurations/270 replays/540
CPU comparisons,46 trajectories/736 windows/184 updates,76 CPU interface tests
with51 optional skips,and54 Python device cases including fresh-process resume.
The separate Full correctness profile observed48,566 AIV+533 AI_CORE+760 MIX_AIV;
no observed AiCPU/fallback. This is not throughput or pure-PyTorch scheduling.

Numerical conditions remain explicit: Full trajectories use AdamW eps1e-5;
public eps1e-8 and normalization epsilons remain unchanged. The RMSNorm eps1e-8
near-zero case passes VJP/same-gradient optimizer checks but fails independent
trajectory tolerance. --full-training-control-check conditioned was explicit:
13 control frames,max4.470348e-6; scores/other tensors/VJPs/updates strict and
routes exact. Strict failures remain failed. Read docs/resident-full-vjp.md.
Width257 retained SwiGLU requires explicit2GiB reverse budget.

## Normalized Aggregate immutable qualification

Implementation0ba9fc6 is committed/pushed and formally qualified. Evidence:
docs/evidence/resident-aggregate-training-20261001.{md,json}. Task-local audit:
TASK/launchers/aggregate_training_evidence.py (pinned to0ba9fc6, audit PASSED).

Both clean standalone and Python-owned builds PASSED;five CTests,all46 component
cells,39 VJP configs/117 replays/234 CPU FP32+FP64 comparisons. Public C++:
57 trajectories/912 retained windows/228 updates;Python106 NPU cases with no
skips;host76/103 optional NPU skips. Aggregate uses strict tensor/control/route
checks and explicit AdamW eps1e-5. The existing Full cell alone uses its explicit
conditioned control policy;its older failures remain failures.

Coverage:mean/weighted mean/active and all-source softmax;None/zero,missing and
present-zero,zero scale with nonzero raw data,physical/logical permutations,
exclusive aliases,shared coefficient/physical owners,feedback/self-loops,
width1/7/257,logical domain257,capacity refusals and checkpoint continuation.
Separate full Aggregate checker profile PASSED:72,933 AIV+578 AI_CORE+1,213 MIX_AIV;
1,153 Aggregate-VJP,1,704 optimizer,2,867 graph-reverse,410 window-bridge records;
no observed AiCPU/fallback. Not throughput or pure-PyTorch scheduling.

All six immutable runs are PASSED under TASK/runs:
build-aggregate-training-clean01,build-aggregate-training-python-clean01,
aggregate-training-clean01,aggregate-training-profile-clean01,
aggregate-training-host-clean01,aggregate-training-python-clean01.
Their status.json/task.log and gate/profile records are hashed in the audit.
The gate/Python used physical9 and profile13,each remapped to logical0.
Retain compiler failures dev01/python-dev01/dev02 and cancelled diag03(exit143),
with cancellation.json. Development archive reuse is separate from qualification.

## Active development after Aggregate

Uncommitted control/read increment in tools/device_online and Python resident clients:
HST/SOFTP forward, complete-frame softmax VJP, content/old/proposal linear and
FP32-norm Read adjoints, graph/alias/optimizer connections, retained raw Full values,
and checkpoint mode/zeta validation. Device testing is PENDING the builds.
Identity boundaries bypass Emit; connected zero HST and zero-norm VJPs remain explicit.
Local control-vjp checker:36 configs/108 replays planned, CPU FP32/FP64.
Public control-training checker:98 trajectories planned plus Python three-family
cases and fresh-process resume. HARD remains the default and its parameter layout
is unchanged. New checkpoint fields default legacy records to HARD/zeta1.

RUNNING:build-control-training-dev01 and build-control-training-python-dev01,
frozen sources/control-training-dev01 and sources/control-training-python-dev01
(base0ba9fc6 plus recorded dirty source). Matching placement cores,jobs2/1800s.
PASSED:control-training-host-dev01,76 tests/135 optional NPU skips.
Next after builds:verify_device_control.py --checks control-vjp control-training;
then targeted HARD graph/parameter/retained/training regressions and separate
--check control-training profile. Python:tests/test_resident_library.py plus
tests/test_resident_training.py (138 cases expected),matching Python-owned core
and backend. Every device run uses queue120s and runtime900s;unique names.
Do not repeat unchanged portable core tests or the older Aggregate qualification.
Commit only Aggregate evidence/status/roadmap now; do not stage the untested
control/read code. No automatic pause at this commit.
Attention adjoints, FP16, device peer progression and performance remain after it.

## Older work to preserve

Do not stage/clean old dirty scripts/build_accelerator_scale.py,
tools/accelerator_scale/{CMakeLists.txt,bounded.h,bounded_export.cpp,bounded_program.cpp,
bounded_select.cpp,bounded_update.cpp,peer_transport.cpp,resident.cpp},
scripts/{benchmark_execution_flow.py,verify_execution_flows.py},
tests/test_flow_semantics.py and tools/accelerator_scale/flow_*.
They are restricted DAG/rank-aligned consumers,not the general-online delivery.
All historical failed jobs remain in TASK/runs,including state/graph/optimizer
and public-training build failures. Preserve strict FP32 near-zero and FP16
Add-gradient failures/reproducers;later passing scopes do not relabel them.

historical-cpu-attention-01 is deliberately SIGSTOP. Its pause.json overrides
running status;it holds host memory and TASK/timing.lock. Do not blindly resume
or stop. Resolve interrupted timing deliberately before formal performance.
Historical Add CPU78.793172/NPU4 47.932888 ms/token means NPU throughput1.6438×
faster;it does not certify this resident backend. No valid CPU Attention train ratio.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service. Status/logs are RUN/status.json and RUN/task.log.

Module libtorch-npu/2.10.0-cann9.0.0;PYTHON=
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized public /opt stack takes precedence over the dated personal guide.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;preserve module PYTHONPATH and
prepend snapshot/python. SoC Ascend910_9392;16 chips64GiB. Queue picks physical
cards;programs use logical npu:0. Recent disk266GiB data/24GiB root;recheck large writes.

`python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT [--commit SHA]
[--npu --max-wait 120] -- timeout --signal=TERM --kill-after=10s 900s '{python}' ...`

All jobs use background.slice,Nice10,isolated frozen sources/builds,jobs2 and bounded
stops. Never mutate files read by active jobs. norm32_after_core.py waits at most600s;
place dependencies before the NPU queue and do not submit it when a long build has
just begun. Source/core/binary/loader hashes and actual exit/markers are authoritative.
Use scripts/durable_records.py atomic fsync/read-back for handoffs. Re-entry:
git status --short --branch;python scripts/status.py;read this file.
