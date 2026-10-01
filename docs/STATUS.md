# Current handoff

Updated 2026-10-01T00:03:21.028994+00:00. **PAUSED at the user's explicit request after this commit/push.**
Do not resume implementation, qualification or experiments until the user confirms
the next alignment. No subagents. Reference repositories and ObsidianVault are read-only.
Repository `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
This handoff accompanies the control-training implementation commit, parent **a78e3fc**;
use `git rev-parse HEAD` for its exact revision. Earlier qualified commits are pushed.
**F1–F7 remain incomplete.** All newly submitted work is terminal; no new NPU job remains.
The older historical CPU Attention process remains deliberately SIGSTOP (see below).

## Contract and overall scope

[execution-flows.md](execution-flows.md) is the execution contract;
[ROADMAP F1–F7](ROADMAP.md) is the only backlog. Current user alignment outranks
run-ml-experiments. Keep source/input/config/environment identity, original results
and failures, synchronized complete timing, finite resource budgets and stops.
Trackio never blocks implementation. Training means independent forward/backward,
VJP, optimizer, continuation and complete throughput; model convergence belongs
in later consumer experiments.

No CPU numerical prepass may supply candidate routes, events, values or gradients.
General online greedy accepts legal topology/input, including positive-delay PDG
feedback, and may degenerate to streaming. Preserve int64, stable order, duplicate
edges, missing/zero and None/zero. CPU FP64/FP32 references remain independent.
Required performance cells: PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch;
CPU/NPU, streaming/prefill, inference/complete training. Five presets and finer
placement switches; FP32 main, FP16 separate. CUDA execution is target-pending.

## Qualified work since alignment

Exact source and full evidence are indexed by ROADMAP and docs/evidence.
Do not rerun the unchanged portable core's **8,954 CPU tests / 23 optional skips**.

- Packed single-NPU forward: vector Aggregate/state/Read, periodic clocks, stable
  InputOrigin ordering, LH/SwiGLU Full, slot/phase emission, event GQA/window and
  fiber attention/pooling, key tiling, shared memory budgets, online time batching,
  normalized Aggregate. Topology/input-specific schedules are not required.
- Public CPU/mixed placement d412541 qualified: independent Read/control/selection/
  event settings, Python/native and standalone checks across families and schedules.
- Public resident inference740fa87 and disk restore622dbb2 qualified. Standalone C++
  and Python-owned plugin stay separate. Python is a native/CANN client, not an
  independently qualified pure-PyTorch resident scheduler.
- State/Full/graph VJPs, alias-owner reduction, device SGD/AdamW/publication and
  retained-window gradients qualified at3e2d54d/3285b13/37430e1/1ef23f3/3b31ee2/c2423f0.
- Public C++ training591e907 and Python/disk lifecycle0f363b8 qualified. They include
  actual independent forward/backward/update, None/zero, shared owners, carried
  windows, device loss cotangents and new-process checkpoint continuation.
- LH/SwiGLU5143de3 qualified:44 component cells,90 local configurations/270 replays,
  46 trajectories/736 windows/184 updates and54 Python NPU cases. Evidence4c00aa5.
- Normalized Aggregate0ba9fc6 qualified:46 component cells,39 configurations/117
  replays/234 CPU comparisons,57 trajectories/912 windows/228 updates,106 Python
  NPU cases and host76/103 optional skips. Evidencea78e3fc; reports
  docs/evidence/resident-aggregate-training-20261001.{md,json}.

Numerical conditions remain explicit. LH/SwiGLU and Aggregate trajectories use
AdamW eps1e-5; public eps1e-8 and normalization epsilons are unchanged. The retained
RMSNorm eps1e-8 near-zero case passes local VJP/same-gradient optimizer checks but
fails independent trajectory tolerance. Full's explicit conditioned control policy
covers13 frames,max4.470348e-6; other tensors/VJPs/updates keep their tolerances and
routes remain exact. Original strict failures remain failed. Width257 retained
SwiGLU declares a2GiB reverse budget. See resident-full-vjp.md and its evidence.

## Current control/Read increment — development verified, qualification pending

HST/SOFTP broadcast forward and adjoints now execute in the resident program.
Complete candidate-frame softmax VJP includes unselected Read gradients; content,
old/proposal linear and FP32-norm Read adjoints retain None/connected-zero semantics.
SOFTP saves actual unmixed Full values; retained tapes freeze Read banks. Alias
reduction, device updates/publication and checkpoint mode/zeta are connected.
HARD remains default and keeps its packed parameter layout. Legacy training v1
records without mode/zeta mean HARD/zeta1. Non-HARD slot-affine explicitly refuses.
See [resident-control-vjp.md](resident-control-vjp.md).

All paths below are relative to TASK (defined under Environment):

- PASSED build-control-training-dev01 and build-control-training-python-dev01:
  frozen base0ba9fc6+dirty sources; standalone and Python-owned production builds.
  Standalone passed5 CTests. Never mix the two runtime owners.
- PASSED control-vjp cell in control-training-dev01:36 configurations/108 long,
  short and empty replays/216 CPU FP32+FP64 comparisons. That overall job FAILED
  later in its separate control-training cell; do not relabel the job passed.
- PASSED build-control-training-dev02: isolated fixture-only relink, with recorded
  production identity/comment equivalence and untouched original dev01 binaries.
  Snapshot sources/control-training-dev02; build builds/control-training-dev02.
- PASSED control-training-dev02:98 trajectories/1568 retained windows/392 updates,
  CPU FP32/FP64, both modes/schedules/optimizers, strict comparisons, eps1e-5.
- PASSED control-training-regression-dev02:9 affected HARD cells: content,window,
  resident,graph-vjp,parameter-vjp,retained,resident-training,full-training,
  aggregate-training. Only Full uses its existing explicit conditioned policy.
- PASSED control-training-python-dev04:154 tests, no skips. Frozen client-dev04,
  matching placement-npu-python-clean01 core and control-training-python-dev01
  backend. Additional source differences are fixture/header comments, not changed
  production computation. Includes4 HST/SOFTP event/fiber attention inference cases.
- PASSED earlier host gate control-training-host-dev01:76 tests/135 optional skips;
  it predates additional test cases. Final immutable host scope remains pending.
- PASSED control-training-profile-dev03: complete98-trajectory checker, explicit
 1024MB raw collection;558031 AI_VECTOR_CORE,7604 AI_CORE,12390 MIX_AIV records.
  Control kernels19264,Emit mix1376,optimizer1568,graph reverse24352,window bridge2352.
  No observed AiCPU or host-fallback diagnostic. Construction/oracles are included;
  this is development placement evidence, not throughput or immutable qualification.
  Frozen client-dev05, unchanged dev02 component; queue120s/run360s bounded.

Retained failures: control-training-python-new-dev02 used unsupported Python Full
"identity" (30 passed before fixture error); replaced with supported identity-LH.
control-training-dev01 exited SIGSEGV(-11) when the retained fixture indexed absent
edge scales on an edgeless graph. Gdb identified retained_fixture; optional alias
guards fixed it. control-training-gdb-dev01 exit0 is diagnostic success only.
control-training-profile-dev02 passed its application but failed operator export:
200MB aging plus CANN COMPUTE_TASK_INFO constraint19/missing task association.
The1024MB retry succeeded. The profiling CLI now exposes a bounded1..4096MB cap,
default200. Preserve all these original logs/statuses; none is rewritten successful.

## Next actions only after user confirmation

1. Qualify this implementation's exact commit with two clean builds, all48
   component cells,154 Python NPU cases and host interface tests (expected76/151
   optional NPU skips); separate control-training profile with1024MB storage.
   Use explicit --full-training-control-check conditioned only for Full regression.
   TASK/launchers/control_training_evidence.py is PREPARED, NOT EXECUTED; it accepts
   the40-character revision argument. Add the retained profile failure to its audit
   before use. Commit evidence separately; do not reuse Aggregate's pinned audit.
2. Attention graph adjoints and public cache roots/initial gradients/window bridges.
   Forward KV exists, but state.value roots alone cannot certify KV continuation.
   Attention proposal depends on old KV, not old visible state.value; do not reuse
   EMA's proposal-to-old-value carry rule. Preserve sliced/cleared/empty KV None/zero.
3. General multi-device progression/completion/communication and training, resident
   FP16 and the remaining declared module/capacity combinations.
4. Complete representative then full-size consumer/performance matrix: independent
   CPU, screened mixed candidate and resident, both schedules, safe aggressive
   chunking, actual continuous state, separate profiles and3 fresh processes for
   recommendations. No new full-size speedup is established by these checks.
5. Final portable packets/commands, immutable evidence audit and environment/version
   coverage. Actual CUDA remains another-machine validation. No automatic restart.

## Older work to preserve

Do not stage/clean old dirty scripts/build_accelerator_scale.py,
tools/accelerator_scale/{CMakeLists.txt,bounded.h,bounded_export.cpp,bounded_program.cpp,
bounded_select.cpp,bounded_update.cpp,peer_transport.cpp,resident.cpp},
scripts/{benchmark_execution_flow.py,verify_execution_flows.py},
tests/test_flow_semantics.py and tools/accelerator_scale/flow_*.
These24 files are unchanged in this increment; task-local pre-commit hashes exist.
They are restricted DAG/rank-aligned consumers, not general-online delivery.

historical-cpu-attention-01 is deliberately SIGSTOP. Its pause.json overrides
running status; it holds host memory and TASK/timing.lock. Do not blindly resume
or stop. Resolve interrupted timing before formal performance. Historical Add
CPU78.793172/NPU4 47.932888ms/token means throughput1.6438x faster; it does not certify
this resident backend. No valid complete CPU Attention training ratio exists.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service. Persistent status/logs: RUN/status.json and
RUN/task.log. Gate/profile results are under RUN/gate or RUN/profile.
Module libtorch-npu/2.10.0-cann9.0.0; PYTHON=
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized public /opt stack overrides the dated personal guide.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Queue selects physical devices;
programs use logical npu:0. Recheck disk before large writes (last253GiB data/24GiB root).
Standalone core=TASK/builds/placement-npu-clean01; Python core=placement-npu-python-clean01;
CPU core=placement-cpu-clean01. Never load the standalone SDK in torch_npu Python.

Future launcher, only after confirmation:
`python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT --commit REV
[--npu --max-wait 120] -- timeout --signal=TERM --kill-after=10s 900s '{python}' ...`
Build command: scripts/build_device_control.py --core-build CORE --build-dir NEW
--ascendc-soc Ascend910_9392 --jobs 2; build timeout1800s.
All tasks use background.slice,Nice10,frozen sources and bounded stops. Do not edit
active snapshots. Use durable_records.py atomic fsync/read-back for handoffs.
