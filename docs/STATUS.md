# Current handoff

Updated 2026-09-30T18:57:04.182526+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
HEAD **3b31ee2**,device optimizer/publication implementation pushed;parameter qualification933d6fe;
resident disk restore/reset622dbb2 qualified; evidence is this increment. No pending authorization/pause. No subagents.
Reference repositories and ObsidianVault are read-only. **Overall F1–F7 remain incomplete.**

## Contract and priorities

[execution-flows.md](execution-flows.md) is the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Current user alignment takes precedence over run-ml-experiments:
keep source/input/config/environment identity, raw results/failures, synchronized
complete timing, bounded resources/stops. Trackio must not block implementation.
Training means independent forward/backward/VJP/optimizer/continuation and complete
throughput; task convergence is a later experiment.

Candidates consume their own inputs/state/parameters, with no CPU numerical routing
prepass. General online greedy accepts legal arbitrary topology/input, including
PDG positive-delay feedback, and may degenerate to streaming. Preserve int64,
stable order, duplicate edges, missing/zero and None/zero gradients. CPU FP64/FP32
remain independent anchors. Device residence includes actual online decisions.
PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Five presets/fine switches; FP32 main, FP16 separate;
three independent processes for recommendations. CUDA execution is target-pending.
Prioritize complete semantic/ownership gates, public/device integration, peer
progression and resident training, then representative/full-size performance.

## Completed increments

- Placement d412541 qualified and evidence pushed **7b31f48**:
  `docs/evidence/execution-placement-20261001.{json,md}`.8,952 CPU tests,10 CTests,
  26 NPU Python cases,C++121 schedules/363 updates,Read precision6/18 and accelerator
  parity/gradient/optimizer/checkpoint/non-default-stream. All clean placement jobs
  terminal0. Its profile includes66 AiCPU INT64 sort/scatter tasks;38.32% is summed
  device task time, not wall time; no host CPU fallback.
- Public resident backend **740fa87**,disk restore/reset **622dbb2**,both pushed.
  `tide::ResidentSession` is a separate installed C++/CANN package; Python-owned
  `_tide_resident` never links standalone SDK into a torch_npu wheel process.
  GraphRuntime.advance_device retains device state; result/snapshot explicitly
  export CPU values, never feed them back into execution. Parameter mutation
  requires rebuilding; poisoned owners cannot silently continue. Disk load validates
  CPU complete cuts/aliases/weights before close, then restores weights/state;
  construction failure leaves closed. Invalid checkpoints leave live state unchanged.
  **Single NPU FP32 HARD inference only. Python is a native client, not an
  independently qualified pure-PyTorch resident scheduler.**
- Prior device forward increments and immutable evidence are indexed in ROADMAP F4:
  general online queues/selection,vector Read/state/sum,normalized Aggregate,
  clocks,identity/tanh/LH/SwiGLU Full,phase/slot-affine emission,InputOrigin,
  event/fiber attention with pooling/key-axis tiling/node-time batches,budgets.
  These do not certify training,FP16,peer progression or full-size throughput.

## Public resident formal qualification

Frozen source **resident-restore-clean01 at622dbb2**, unchanged compiled backend
**resident-public-clean01 at740fa87**,matching cores **d412541**. Full clean backend
builds `build-resident-public-{python,standalone}-clean01` PASSED;standalone4 CTests.

All names below map to `TASK/runs/NAME/{status.json,task.log}` and
`tide-execution-flows-NAME.service`; inspect actual terminals and output hashes.

- `resident-public-cpu-clean01`: PASSED8,954 tests/23 optional-device skips,1690.66s,
  FP32/FP64 full scripts/verify.py,
  `TASK/builds/placement-cpu-clean01`,2400s,one ATen/BLAS thread.
  Output RUN/verified/{result.json,tests.log}. Do not start duplicate CPU gate.
- `resident-public-standalone-clean01`: PASSED terminal0,all33 component cells,
  including resident64 windows,window/content and other forward regressions.
- `resident-public-python-clean01`: PASSED25 cases,three graph families,two schedules,
  EMA/event/fiber attention,CPU/NPU input,non-default stream,restore/reset,
  frozen-parameter/alias/config/capacity and no-implicit-export behavior.
- `resident-public-profile-clean01`: PASSED.10,534 AIV+16 MIX_AIV,no AiCPU/fallback.
  This is the resident checker scope,not all module performance or throughput.
- `build-resident-consumer-clean01` + `resident-consumer-clean01`: PASSED;
  installed public-header-only client,prefix with spaces,loader and3 feedback windows.
- Directed restore development `resident-restore-host-dev01`:126 passed/23 optional
  skips;`resident-restore-python-dev01`:25 passed. All terminal0.

All public resident qualifications are terminal0. Task-local
TASK/launchers/resident_public_evidence.py audited exact sources,backend/core
fingerprints,raw results/logs/profile CSVs. Reports
`docs/evidence/public-resident-20261001.{json,md}` are committed/pushed8b89080.
Do not relabel public inference as resident training. Historical failures remain.

## Current training increments and next actions

State VJP qualified at3e2d54d:216 CPU FP32/FP64 autograd cases,4 actual tapes,
all34 device cells and7,743 AIV-only profile tasks. Evidence committed4a29b7c:
`docs/evidence/device-state-vjp-20261001.{json,md}`.
Full identity/tanh VJP qualified at3285b13:96 CPU FP32/FP64 cases,2 actual tapes,
all35 cells;8,036 AIV+288 AI_CORE, no AiCPU/fallback. Evidence658943a:
`docs/evidence/device-full-vjp-20261001.{json,md}`.
These are local components,not full training/optimizer/throughput claims.
Retained failures:state-vjp-dev02 bool-cache race;0459195 clean gate/profile
and state-vjp-add-dev01 assertion-helper regression (fixed without tolerance changes).

Graph reverse37430e1 is committed/pushed. HARD single-NPU FP32 sum/broadcast,
identity/EMA/Add state and identity/tanh Full only; actual-stage reverse loop
and physical message/scale links stay on device. Contract resident-graph-vjp.md.
Development graph-vjp-dev04 passed98 complete windows against independent CPU
FP32/FP64,64 link windows,empty/poison/None/replay/refusals and relevant regressions.
Dev04 profile:88,699 AIV+1,269 AI_CORE+792 MIX_AIV,no AiCPU/host fallback;
includes correctness assertions and is not throughput. Retain build-dev01
Muls scalar-template failure and dev03 wide-fixture forward-budget refusal.

Formal build-graph-vjp-clean01,graph-vjp-clean01 and graph-vjp-profile-clean01
all PASSED terminal0 from exact37430e1. Clean build4 CTests,full37-cell gate,
98 graph windows,64 link windows. Audited source/binary/log/CSV hashes match.
Profile88,699 AIV+1,269 AI_CORE+792 MIX_AIV,5,934 graph reverse records,
no AiCPU/fallback. Evidence docs/evidence/device-graph-vjp-20261001.{json,md}.

Parameter-owner reduction1ef23f3 committed/pushed;all formal jobs terminal0:
build-parameter-vjp-clean01 (4 CTests),parameter-vjp-clean01 (38 cells),
parameter-vjp-profile-clean01.36 actual aliased graph/root cases,CPU FP32 oracle;
23,580 AIV+334 AI_CORE+144 MIX_AIV,74 owner kernel records,no AiCPU/fallback.
Audit exact sources/binaries/logs/CSVs passed. Evidence:
docs/evidence/device-parameter-vjp-20261001.{json,md}. Contract resident-parameter-vjp.md.

Device optimizer + publication is this tested implementation increment:
- device_optimizer.{h,cpp},optimizer_layout.h,optimizer_check.cpp;
  Ascend C optimizer_vector.h and optimizer plan/values/commit kernels.
- parameter_publish.{h,cpp},Ascend C publisher,training_step_check.cpp;
  ContentFlow parameter-bank view and per-window source-scale diagnostic snapshot;
  build/check registration and docs/resident-optimizer.md.
- Packed SGD/AdamW finite proposals and all-owner commit;None skips every update,
  connected zero advances slots/counters/decay. Shared TensorImpl update once;
  distinct owners sharing storage explicitly refused. Publication includes Read aliases.
- build-training-step-dev01 PASSED isolated checked build;
  training-step-dev01 PASSED9 component checks,optimizer32 trajectories/256 updates
  against CPU FP32/FP64 and complete chain18 trajectories/72 actual windows,
  including2 wide257 trajectories. Each optimizer boundary explicitly truncates.
  No public training/retained-window/performance qualification implied.
- build-training-step-dev02 PASSED checked isolated optimizer plan/check finalization;
  training-step-dev02 PASSED9 checks,including new nonfinite slot/correction
  transaction tests. training-step-profile-dev02 PASSED79,156 AIV+1,808 AI_CORE+
  288 MIX_AIV,288 optimizer records,no AiCPU/fallback;not throughput.
- build-training-step-clean01 PASSED from exact3b31ee2 (four CTests);training-step-clean01 and training-step-profile-clean01 are running after bounded dependency waits (1800s/jobs2/no lease),then training-step-clean01 full40-cell gate and
  training-step-profile-clean01 (900s/one NPU/queue120s each),audit and commit evidence.
- Retained-window backward is this tested implementation increment:retained_tape.{h,cpp},window_bridge.cpp,parameter_accumulate.cpp,
  retained_fixture/check and three Ascend C kernels,plus parameter_vjp.h and build/check registration.
  Device tape snapshots after close,actual pending/state cotangent links and alias accumulation;
  26 retained trajectories/104 windows planned against independent CPU FP32/FP64.
  build-retained-dev01 PASSED checked isolated build;retained-dev01 PASSED8 checks,26 trajectories/104 retained windows.
  retained-profile-dev01 PASSED49,324 AIV+838 AI_CORE+832 MIX_AIV,314 bridge records,no AiCPU/fallback.
  Contract docs/resident-retained.md. After commit,launch build-retained-clean01 from the exact hash (1800s/jobs2/no lease),then retained-clean01 full41-cell gate and retained-profile-clean01 (900s/one NPU/queue120s each). This is not the public training lifecycle or a throughput claim.

Retain task-local build-optimizer-dev01 failure (static archive after as-needed
ascendcl) and build-optimizer-dev01b failure (public consumer has no direct
ascendcl option). External relinker fixed;the old runs remain failed.
optimizer-dev01 failed before acquiring NPU due to its build dependency.
optimizer-kernel-dev01 separately PASSED the already linked optimizer binary:
32 trajectories/256 updates. Later full training-step development build supersedes
its limited build scope. Never treat either failed build as clean qualification.

Next: finish the optimizer40-cell and retained41-cell immutable qualification/evidence;then public training ownership,parameter-generation guards and checkpoint integration. Borrowed tapes
expire on next advance/close/parameter update. LH/SwiGLU/normalized Aggregate/
attention/HST/SOFTP adjoints,peer progression,FP16 and the public/performance
matrix remain. Do not mark F4/F5 complete at this restricted HARD chain.
Portable core/Python unchanged:do not repeat the8,954 CPU tests/23 optional skips.

## Preserved older work and interrupted timing

Do not stage/clean old dirty scripts/build_accelerator_scale.py,
tools/accelerator_scale/{CMakeLists.txt,bounded.h,bounded_export.cpp,bounded_program.cpp,
bounded_select.cpp,bounded_update.cpp,peer_transport.cpp,resident.cpp},
scripts/{benchmark_execution_flow.py,verify_execution_flows.py},
tests/test_flow_semantics.py and tools/accelerator_scale/flow_*.
They are restricted DAG/rank-aligned consumers,not general-online delivery.
Keep strict FP32 near-zero failures and FP16 Add-gradient failures/reproducers.

`tide-execution-flows-historical-cpu-attention-01.service` is deliberately SIGSTOP.
TASK/runs/historical-cpu-attention-01/pause.json overrides its running record.
It holds host memory and TASK/timing.lock. Do not blindly resume/stop; resolve the
interrupted timing deliberately before formal performance. Historical Add full
training CPU78.793172/NPU4 47.932888 ms/token means NPU throughput1.6438× faster;
it does not qualify the new resident backend. Attention has no valid full CPU ratio.

## Environment and commands

TASK=/mi/data2T/zlong/tide-execution-flows. Builds and frozen sources live there;
artifact links artifacts/execution-flows-NAME. Use scripts/durable_records.py for
atomic fsync/read-back handoffs. Latest disk check278GB data/25GB root; recheck large writes.
Module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized public /opt stack overrides dated personal-anaconda defaults.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;retain module PYTHONPATH and
prepend snapshot/python. SoC Ascend910_9392;16 chips64GiB. Queue selects physical
cards; programs use logical npu:0. Never mix standalone SDK with torch_npu wheels.

```
python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT [--commit SHA] [--npu --max-wait 120] -- timeout --signal=TERM --kill-after=10s 900s '{python}' scripts/COMMAND ...
```

No source/build mutation while jobs read them. `norm32_after_core.py` is a600s
bounded dependency wrapper:put it BEFORE the queue helper,never hold a card while
waiting for compilation. All units in background.slice; inspect status plus
expected result markers/hashes. Re-entry:git status --short --branch;python scripts/status.py.
