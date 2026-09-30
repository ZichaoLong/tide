# Current handoff

Updated 2026-09-30T17:29:38.557896+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
Last pushed HEAD **622dbb2** (resident disk restore/reset). This increment adds a
device identity/EMA state-chain VJP. No pending authorization/pause. No subagents.
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

- `resident-public-cpu-clean01`: RUNNING,FP32/FP64 full scripts/verify.py,
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

After CPU terminal0, run task-local `TASK/launchers/resident_public_evidence.py`.
It audits exact sources,backend/core fingerprints,raw results/logs/profile CSVs
and writes docs/evidence/public-resident-20261001.json. Add a short reviewed .md
report,update ROADMAP F5 and commit/push evidence separately. Do not relabel
public inference as resident training. Historical development failures remain.

## Current device state VJP increment

`docs/resident-state-vjp.md` states the narrow first-order identity/EMA HARD
contract. Borrowed actual forward tape,device predecessor links and reverse
progression,explicit cotangent connection bits,packed feature tiles. This is
an internal module,NOT complete graph backward/autograd/optimizer. It returns
sample partials; alias reduction/graph message dependencies remain to integrate.

- `build-state-vjp-dev01` PASSED full build and4 CTests.
- `build-state-vjp-dev02` PASSED isolated relink in a new directory.
- `state-vjp-dev02` FAILED: content VJP/connection mismatch. Retain this failure.
- `build-state-vjp-dev03` PASSED isolated rebuild of both VJP kernels plus relink,
  checking unchanged parent production/kernel hashes,archive and binary identities.
  Task-local launcher `state_vjp_relink_kernels.py`;all commands/derivation are in
  `builds/state-vjp-dev03/control-build.json`. Development only.
- `state-vjp-dev03` PASSED four gates: state-vjp,content,window,resident.
  State VJP:24 CPU FP32/FP64 autograd cases plus2 actual device-forward tapes;
  identity/EMA,adopt/clear,connected-zero/absent,poisoned padding,feature tails,
  >2^55 clocks/counters,empty reverse replay,invalid metadata and bounded allocation.
  Connection flags now have a single metadata writer; vector cores write only
  numerical tiles,avoiding scalar bool writes to a shared GM cache line.
  No formula/tolerance was relaxed. Current source only adds failure diagnostics
  after dev03;production bytes match the passing snapshot.
- `state-vjp-profile-dev03` PASSED:1,664 AIV only;51 state_vjp+51 state_vjp_plan
  tasks,no AiCPU/fallback. Includes forward construction/assertions,not throughput.

Next: commit/push tested increment. Freeze its exact commit as **state-vjp-clean01**;
full `build-state-vjp-clean01` (1800s,jobs2,no lease) using scripts/build_device_control.py
--core-build TASK/builds/placement-npu-clean01 --build-dir TASK/builds/state-vjp-clean01
--ascendc-soc Ascend910_9392 --jobs2. Then full34-cell verify_device_control.py
(one NPU900s,queue120s) and separate --check state-vjp profile. Commit evidence only
after terminal/hash audit. Existing core/Python code did not change; avoid another
redundant full CPU run. Continue complete resident training/peer/FP16/public matrix,
then F6 representative/full-size comparisons. Do not stop at the local VJP.

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
