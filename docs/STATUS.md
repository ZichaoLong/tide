# Current handoff

Updated 2026-09-30T22:08:21.589854+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository `/home/zlong/llm/graph-execution-foundation`,real path
`/var/tmp/zlong-graph-execution-foundation/repository`;branch `graph-execution-foundation`.
HEAD **9022636**,pushed. No authorization pending. No subagents.
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

## LH/SwiGLU implementation increment

Implementation is ready to commit after development gates;immutable qualification
has not run yet. Prior qualified HEAD9022636 remains pushed. No authorization pending.

- FullExtraTape/Vjp,packed LH activation/norm and SwiGLU matrix VJPs are wired into
  graph reverse,retained tapes,alias-owner reduction,device optimizer publication
  and public C++/Python training. No public API or checkpoint schema change.
- Isolated extra-full-vjp-dev02 PASSED90 configurations/270 long-short-empty replays,
  each CPU FP32/FP64 (540 comparisons),width1/7/257,None/zero,poisoned absent owners.
- Full production standalone dev02 build/four CTests PASSED. Python-owned
  full-training-python-dev01 build PASSED;host76 tests/51 optional skips;device54
  tests PASSED including three families,both schedules,real NPU loss cotangents,
  new module cases and disk/new-process continuation. No standalone/wheel mixing.
- build-full-training-diag07 PASSED audited test-only relink plus CPU control-policy
  guards. Production binaries are byte-verified dev02;source full-training-diag07.
- full-training-dev07 PASSED11 selected component gates: full-training,
  resident-training,full-vjp,graph-vjp,parameter-vjp,training-step,retained,state-vjp,
  norm32,lh-full,swiglu. Full gate46 trajectories/736 windows/184 updates includes
  all9 LH profiles,SwiGLU,mixed,shared owners,both schedules and CPU FP32/FP64.
  Width257 SwiGLU uses explicit2GiB reverse budget for four retained programs.

Numerical conditions are explicit;do not overstate a strict pass:
- Full trajectories use AdamW eps1e-5 (test fixture only);public default1e-8 and
  norm eps1e-7/1e-5 unchanged. The retained kind7 eps1e-8 near-zero reproducer
  fails independent end-to-end parameter tolerance;CPU FP32/FP64 themselves differ
  more than twice that tolerance. Actual device gradient and same-gradient CPU
  optimizer checks PASS both schedules and dtypes. Read training_numerics.cpp.
- Full qualification selects --full-training-control-check conditioned. Strict
  control checks remain default and original width257 LH failures remain failed.
  Scores/other tensors/gradients/updates use original tolerance;routes exact.
  For mismatched controls,check both own-score complete-candidate softmaxes in
  FP64 and the rigorous range(score error)/4 propagation bound plus local rounding.
  Maximum recorded control difference4.470348e-6;13 frames used this comparison.
  No candidate value or next-step state receives reference data.
- Original failures are retained in full-training-dev05/dev06,profiles dev04/dev06,
  rms-float32/float64-diag01,wide-5-diag03/04/06. Build-dev01 constness,build-dev03
  comment audit,dev04 wrong CLI and build-diag05 reuse-metadata failures are also
  retained. Passing later scopes do not relabel any of these failures.

## Next commands and qualification

TASK=/mi/data2T/zlong/tide-execution-flows;RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service. All new development jobs are terminal.
Commit only this increment,then push and freeze that exact commit twice:

- build-full-training-clean01: snapshot/build full-training-clean01,
  scripts/build_device_control.py --core-build TASK/builds/placement-npu-clean01
  --build-dir TASK/builds/full-training-clean01 --ascendc-soc Ascend910_9392 --jobs2.
- build-full-training-python-clean01: same tool with matching Python-owned core
  TASK/builds/placement-npu-python-clean01,own snapshot/build. Both1800s.
- After standalone build: full44-cell verify_device_control.py gate with explicit
  --full-training-control-check conditioned;queue120s/run900s. Separate profile
  --check full-training --application-arg=--control-check=conditioned. The profile
  includes correctness oracles and is not throughput. Check five build CTests.
- After Python build: same54 tests from tests/test_resident_library.py and
  tests/test_resident_training.py,with TIDE_RESIDENT_DEVICE=npu:0 and the new
  TIDE_RESIDENT_LIBRARY;matching Python-owned core. 76 CPU interface tests were
  already passed in development;recheck unchanged-source provenance before reuse.

Audit exact source/component/core/binary/loader/result/log hashes,all exit codes,
explicit numerical scope and actual placement. Commit reviewed evidence separately.
Do not repeat unchanged portable core's8954 CPU tests. Then continue normalized
Aggregate adjoints,attention/HST/SOFTP,FP16,peer progression and the full performance
matrix. None of those later adjoints has new implementation yet. Normalized
Aggregate must retain absent all-softmax denominator gradients,physical/logical
source identity,and present-zero contributions;never recover coefficients by
message-value division. F4/F5 remain incomplete.

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
