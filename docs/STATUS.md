# Current handoff

Updated 2026-09-30T17:44:05.205509+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
HEAD **658943a**,Full VJP qualification committed/pushed;state-chain qualification4a29b7c;public resident evidence8b89080;
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
  No formula/tolerance was relaxed. The later failure-diagnostic edit accidentally changed an assertion branch in
  commit0459195;see the retained Add-dev01 failure and correction below.
- `state-vjp-profile-dev03` PASSED:1,664 AIV only;51 state_vjp+51 state_vjp_plan
  tasks,no AiCPU/fallback. Includes forward construction/assertions,not throughput.

The full clean build at0459195 PASSED (including4 CTests), but
`state-vjp-clean01` and `state-vjp-profile-clean01` both FAILED on the accidentally
changed assertion helper. Keep these formal failures; no qualification claim at0459195.
The helper wrongly demanded zero for a defined gradient that matched the oracle;
this increment fixes the branch without changing the original tolerance/None checks.

This increment extends the component to Add-repeat. It carries retention/periodic
clock tables and sample-feature retention partials. Device tick bounds and bounded
literal-multiply replay chunks preserve the actual multiplication sequence without
pow/division; chunk2/32 changes physical work only. All connection flags retain a
single metadata writer. Full graph dependencies/parameter aliases/optimizer remain.

- `build-state-vjp-add-dev01` PASSED;`state-vjp-add-dev01` FAILED on the same
  assertion helper before Add cases. Raw failure remains.
- `build-state-vjp-add-dev02` PASSED isolated kernel build/relink with checked
  parent sources/binaries. `state-vjp-add-dev02` PASSED state-vjp/content/window/resident.
  **216** independent CPU FP32/FP64 autograd cases +**4** actual device tapes:
  identity/EMA/Add,retention0/1/negative/fractional,periodic clocks,tick chunk2/32,
  adopt/clear,connection semantics,poison/empty replay and malformed/capacity/work bounds.
- `state-vjp-add-profile-dev02` PASSED:**7,743 AIV only**,438 state_vjp+438 state_vjp_plan;
  no AiCPU/fallback. This is component placement,not throughput.

Implementation3e2d54d is committed/pushed. Frozen **state-vjp-add-clean01**.
Full `build-state-vjp-add-clean01` PASSED (1800s,jobs2,no lease; four CTests)
using scripts/build_device_control.py --core-build TASK/builds/placement-npu-clean01
--build-dir TASK/builds/state-vjp-add-clean01 --ascendc-soc Ascend910_9392 --jobs2.
`state-vjp-add-clean01` full34-cell verify_device_control.py PASSED and
`state-vjp-add-profile-clean01` --check state-vjp profile PASSED;one NPU900s/queue120s each.
No redundant full CPU gate:portable core/Python code unchanged and public622dbb2 gate passed.
Exact-source/hash/terminal audits passed. State VJP qualification reports:
docs/evidence/device-state-vjp-20261001.{json,md};34 cells,216 isolated cases,4 real tapes,
7,743 AIV-only profile tasks. Earlier failed clean runs remain failures.

Committed increment3285b13: identity/tanh Full VJP in tools/device_online/full_vjp*,
three Ascend C plan/payload/reduction kernels, CannProgram zero/tanh_backward,
ContentFlow borrowed full_tape and PackedFull parameter accessors, build/gate registration.
Dev build-full-vjp-dev01 PASSED via task-local full_vjp_relink.py: verified unchanged
state-vjp-add-clean01 objects plus three new kernels and changed C++ objects in a new directory.
full-vjp-dev01 PASSED:96 CPU FP32/FP64 autograd cases (each full/short/empty replay),
2 real device tapes, numerical FP32/FP16,failure,state-vjp,content,window,resident.
full-vjp-profile-dev01 PASSED:8,036 AIV+288 AI_CORE,no AiCPU/host fallback.
This is local-component placement,not throughput.
No immutable qualification yet; contract docs/resident-full-vjp.md.
build-full-vjp-clean01 PASSED from frozen3285b13 (1800s,jobs2,no NPU,four CTests).
full-vjp-clean01 full35-cell gate PASSED;full-vjp-profile-clean01 --check full-vjp PASSED,
each one NPU900s/queue120s. Exact-source/binary/result/profile audits passed.
Reports docs/evidence/device-full-vjp-20261001.{json,md} committed/pushed658943a.

Uncommitted reverse_links increment: actual forward message/event hash index on NPU,
stable producer/consumer/scale contributor chains,stage offsets and cut-boundary
classification; ContentFlow::reverse_tape rejects unavailable modules.
Device validation tests cover64 actual feedback/DAG/edgeless windows and malformed
metadata/budget. build-reverse-links-dev01 submitted from reverse-links-dev01,
900s/2 build jobs,no NPU;600s bounded wait for full-vjp clean build then
reverse_links_relink.py in a new directory. Build PASSED;reverse-links-dev01
PASSED64 windows and full-vjp/state-vjp/content/window/resident regressions.
Uncommitted graph_vjp.cpp/h and three Ascend C kernels now compose actual-stage
Full -> state-chain -> sum/message adjoints with device loop progression and
physical parameter reduction. Tests compare whole-window CPU Streaming autograd,
all input/initial/pending/node/scale gradients for independent roots,zero/None,
feedback and warm continuation. Graph build-graph-vjp-dev01 from frozen graph-vjp-dev01 is being submitted
(900s,jobs2,no NPU),using graph_vjp_relink.py against full-vjp-clean01:
compile four new kernels and changed objects into a new directory.
build-graph-vjp-dev01 FAILED:Ascend C Muls template cannot accept a __gm__ float
lvalue directly;load the scale into an ordinary scalar before the vector instruction.
No numerical/tolerance change. build-graph-vjp-dev02 submitted from a new frozen snapshot
with the same900s build bound and checked parent. graph-vjp-dev02 waits at most600s
for that build BEFORE requesting one NPU (queue120s),then runs graph-vjp,reverse-links,
full-vjp,state-vjp,content,window,resident gates (overall900s). Both build/gate PASSED:
96 graph windows,independent CPU FP32/FP64 forward/autograd,replay and64 link windows.
Additional graph tests cover widths1/257,empty connected-zero roots,malformed preflight,
budget/dtype refusal. build-graph-vjp-dev03 from new snapshot submitted with these tests,
900s/jobs2/no lease;build passed,graph-vjp-dev03 FAILED on forward budget admission
for the added257-wide fixture (default64MiB). Explicitly budget512MiB only for
that wide forward fixture;no numerical changes. Also exercise phase-absent edges/outputs.
build-graph-vjp-dev04 uses graph_check_relink.py with checked unchanged production/kernels
from dev03 and recompiles only two test files;new frozen snapshot/output,no lease.
build-graph-vjp-dev04 and graph-vjp-dev04 PASSED:98 complete graph windows against
CPU FP32/FP64,empty/poison/zero/None/budget/dtype/malformed checks;64 reverse-link
windows;full-vjp,state-vjp,content,window,resident. Component/source/binary audit matched.
graph-vjp-profile-dev04 PASSED:88,699 AIV+1,269 AI_CORE+792 MIX_AIV,no AiCPU/host fallback.
This is the full correctness checker including forward/CPU assertions,not throughput.
Next commit this coherent graph reverse increment,then full clean build from its
immutable commit (1800s,jobs2) and full37-cell gate/separate graph-vjp profile
(each900s,one NPU,queue120s). Alias/optimizer/public training remain next.

Continue F4/F5: Full/Aggregate/transport adjoints and complete resident training,
peer progression,FP16 and public matrix,then F6 representative/full-size comparison.
A useful graph-backward design is to reverse actual forward stages on device:
all message dependencies cross to earlier stages; each stage's node-time state
chains can consume Full cotangents and return prior-state/content cotangents.
Do not apply a complete state-chain VJP once while ignoring interleaved message
cotangents. Full VJP must recompute/save tanh itself (full-content subtraction
would lose precision),pack selected connected rows and keep None/zero flags.
No implementation of that complete graph reverse loop exists yet. Do not stop
or claim training complete at the isolated state VJP.

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
