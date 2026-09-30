# Current handoff

Updated 2026-09-30T14:32:48.002592+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
HEAD **523b323**, memory implementation committed/pushed. No pending authorization or pause. No subagents.
Reference repositories and ObsidianVault are read-only. Preserve older dirty work below.

## Contract and priorities

[execution-flows.md](execution-flows.md) is the execution contract;
[ROADMAP F1–F7](ROADMAP.md) is the only backlog. **F1–F7 are not complete.**
Latest user alignment overrides run-ml-experiments: keep only source/input/config/
environment identity, raw results/failures, synchronized timing and bounded resources/
stops. Reuse records; Trackio must not block implementation. Training means independent
forward/backward/VJP/optimizer/continuation and complete throughput, not convergence.

Candidates independently consume inputs/state/parameters. No CPU numerical route
prepass, whole-window potential expansion or fixture-specific schedule. General
online greedy prefill accepts each family's legal topology/input, including PDG
positive-delay feedback. Device residence includes the actual online decisions.
Preserve exact int64 coordinates/counts, stable order, duplicate edges, missing/zero,
None/zero gradients. CPU FP64/FP32 remain independent references.

Prioritize independent correctness/failure gates and bounded profiling; actual modules,
safe chunking and node-time batches; public five presets/matrix, peer progression and
resident backward/optimizer; then representative/full-size timing. PDG LibTorch;
TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill × inference/complete
training. CPU,mixed A/B/C,resident; FP32 main,FP16 separate. Three fresh processes
before performance recommendations. CUDA execution remains target-machine pending.
Current device flow is **single-device FP32 HARD inference**, no resident backward.

## Active work and next actions

1. Shared forward budgets/serial workspace reuse523b323 clean qualification
   PASSED: full build/four CTests,all29 cells and independent memory profile.
   All three memory-clean01 jobs are terminal0; reviewed evidence recorded in
   docs/evidence/device-memory-20260930.{json,md}, ready to commit/push separately.
   This is FP32 HARD inference,not training or throughput.
2. Event batching development remains uncommitted. Its full build is at host
   targets after Ascend C kernels compiled. Submitting device-event-batch-gates-dev01
   with bounded600s dependency,queue120s,one NPU,gate900s; event-batch first then
   all30 cells. Separate profile follows its semantic pass. Never modify that
   frozen source/build while it runs.
3. Event node-time batching implemented but UNTESTED. Submitting full frozen
   build-device-event-batch-dev01: snapshot event-batch-dev01,build
   device-event-batch-dev01,core origins-npu-clean01,jobs2,1800s. New event_batch
   gate planned72 analytic/restore cases and216 general topology windows/restores.
   Run it first,then all30 cells,then separate placement profile. All jobs use
   new immutable paths; never mutate memory-clean01 while it runs.
   Changed event plan/payload/cache/index kernels,shared key tile address adapter,
   packed_event_attention,profile capability/footprint,registry/CMake; new
   ascendc/event_sequence.h and event_batch_check.cpp. Actual ready events get
   packed QKV in one device loop,then causal/window queries in another. Old KV
   and compact new rows stay immutable until final owner commit; no per-event
   KV copying. Observe-all without selected clear admits node-time batches;
   selection-dependent adoption/clear keeps complete single-frame fallback.
4. After correctness/profile,update content-flow docs and commit implementation;
   qualify exact source separately. Continue fiber time batches,remaining modules,
   public presets/matrix,peer progression,resident backward/VJP/optimizer and
   representative/full-size performance. F1–F7 are not complete.

Memory implementation: CPU-deferred profile tables; reserve all six module minima
before growing physical chunks; conservative/aggressive surplus headroom; bounded
CANN workspace size queries and one serial arena allocated at finish; unique
retained tensor bytes and allocator calibration. Values/capacities are unchanged.
The plan excludes caller tensors,training,communication,vendor internals and allocator
fragmentation. Actual peak allocated delta across24 fixtures stays within budget;
width257/512MiB aggressive uses167,218,688 bytes,while conservative149,631,488.
No free-HBM guarantee is claimed. Complete calibrated training-memory work remains.

Preserved memory development failures:
- dev01 full build/four CTests passed; gate rejected incorrect external positions
  before execution;seven workspace/refusal checks passed. Positions must be0,1,2
  for sparse logical times0,2,5. dev02 review caught reversed fields before gating.
- dev03/dev04 gates failed strict FP32 parity in the original symmetric input:
  node1/time5 SiLU+LayerNorm Full,variance1.6354e-5,middle~-0.0028,error1.45053e-6.
  Original failure remains reproducible, not relabelled passed. New allocation
  fixture uses asymmetric square-pattern channels; tolerance remains1e-5/1e-6.
  Dedicated LH conditioning gate also passes. Production formulas unchanged.
- dev05 passed48 windows/16 calibrations,then width257 refused a CANN workspace
  before allocation. Per-op retained workspaces unnecessarily summed serial
  lifetimes. dev06 fixes that with one maximum-size arena and verifies aliasing,
  sequential dependent operations and exact-cap repeated execution. All29 pass.
- prior-workspace regression-dev03 passed all28 preexisting cells. Immutable old
  snapshots/logs remain; dev06 relinked every program after production hash checks.

## Key-axis attention tiling: implementation7b03614

`tiled_attention.{h,cpp}`, three Ascend C tile/softmax/merge kernels and
`attention_tile_check.cpp` are committed. Default key bound128; a tiled key is
at most256; a bound covering capacity selects complete-key CANN softmax. Device
metadata chooses actual key blocks; online max/denominator/weighted sums preserve
global normalization. Budget shrinkage never changes logical visibility/cache.
Counters report actual tile calls and real/padded score entries at boundaries.
All28 development cells and profile passed on dev03;96 long/ragged/dense/tiled/
restore windows and12 extreme/saturation/budget cases, rtol1e-5/atol1e-6 unchanged.
Clean build/gates/profile on7b03614 also passed; extract exact counts from records.

Development provenance and retained failures:
- `build-device-key-tile-dev01` PASSED full frozen build/four CTests. Initial
  `gates-dev01` FAILED before execution: event fixture had257 KV rows but17
  observations. Fixed to legal large int64 counts; no production formula change.
- `build-device-key-tile-dev02` PASSED isolated incremental build: regenerated
  only softmax/merge kernels in a new CANN tree, recompiled the new checker and
  relinked. All other source/binary identities were verified. It adds a review-found
  case: finite repeated decay can saturate old bias to-inf, giving an empty-mass
  key tile followed by finite new keys. Skip its mass; do not evaluate-inf-(-inf).
  Actual all-zero-mass queries remain nonfinite, distinct from harmless padding.
- `gates-dev02` passed96 windows, then FAILED because the shared finite-only
  comparator rejects intentional-inf intermediate bias. `dev03` changes only
  that checker: exceptional bias slots must match **exactly**, then copied results
  neutralize only those already-equal slots for the unchanged full finite gate.
  Original results remain intact. No tolerance change. `build-dev03` PASSED;
  kernel production sources/objects unchanged. New dev03 tile gate passed.
- Immutable original logs/snapshots remain. Launchers `key_tile_incremental02.py`
  and `key_tile_incremental03.py` under TASK/launchers record hashes/commands.

Tile coverage: widths1/4/33/257, MHA/GQA/MQA, ragged initial KV0/5/257, key limits
1/7/128/300, physical query chunk4, exact uniform global-denominator anchors,
extreme logits, saturated old bias, continuation and changed tile size on restore.
Existing fiber/pooling/event suites now exercise key limits1/7 and retain all
feedback/Read/clear/origin/clock/selection coverage. rtol1e-5/atol1e-6 unchanged.

## Latest clean qualifications (committed/pushed)

- Event attention **b668f2f**, evidence commit **c0dcd4e**: full clean build/four
  CTests/all27 cells;98 anchors,192 complete windows,40 lifecycle windows,5 refusals.
  Profile92,569 AIV+2,530 AI Core,no AiCPU/fallback. One compact KV row per event,
  GQA/window, mixed fiber/event proposals, selected-only/clear and lean continuation.
  [Evidence](evidence/device-event-attention-20260930.md). Dev failures preserve the
  wrong configuration label and scalar-cache-line race (32 metadata writers fixed
  to one; vector/matrix payload stays packed).
- Fiber pooling **691cb31**, evidence **e3d339c**: full build/four CTests/all26 cells;
 80 anchors,18 wide/extreme/empty domains,160 windows,5 refusals;
 76,953 AIV+1,586 AI Core,no AiCPU. Sum/mean/linear/active/all-softmax act after query
  attention, before projection. [Evidence](evidence/device-fiber-pool-20260930.md).
- Fiber-sum **081f567**, evidence **aa1cef3**: full build/four CTests/all25 cells;
 16 anchors,192 windows,40 lifecycle windows,6 refusals;64,617 AIV+1,984 AI Core,
  no AiCPU. [Evidence](evidence/device-fiber-20260930.md).
- Earlier raw device increments: clocks49ff108, vector Read8a735ef, LH Fullcd03ca8,
  origins1be3619, slot emission6445121, SwiGLU74cec2f. Evidence linked in ROADMAP.
  Native core norm32 is0e66d89; stable origin-order fix832a881 underlies the matching
  core build. These finite profiles do not certify all module/training/matrix work.

All device attention adapters still use one complete region-time frame per stage.
Independent owners/messages remain packed. These clean reports establish parity/
placement, not throughput or training. Cache bounds refuse explicitly, not evict.

## Preserved older work and timing

Do not stage/clean old dirty `scripts/build_accelerator_scale.py`,
`tools/accelerator_scale/CMakeLists.txt`, `bounded{.h,_export.cpp,_program.cpp,
_select.cpp,_update.cpp}`, `peer_transport.cpp`, `resident.cpp`,
`scripts/benchmark_execution_flow.py`, `scripts/verify_execution_flows.py`,
`tests/test_flow_semantics.py`, and `tools/accelerator_scale/flow_*`.
Those are limited DAG/rank-aligned consumers, not general-online delivery.
flow-dev05 CPU24 passed;NPU18 FP32 passed before an FP16 resident Add gradient
failure;dev06 builds passed with no gates. Peer CPU8/NPU16 passed separately.

`tide-execution-flows-historical-cpu-attention-01.service` remains deliberately
SIGSTOP. TASK/runs/historical-cpu-attention-01/pause.json overrides running status.
It holds host memory and TASK/timing.lock. Do not blindly resume/stop; resolve
interrupted timing/lock deliberately before formal timing, preserving records.
Historical Add complete training CPU78.793172/NPU4 47.932888 ms/token means NPU
throughput1.6438× faster. It does not qualify this resident backend. Attention has
no valid CPU complete-training ratio. Retain historical failures/reproducers.

## Durable environment and commands

TASK=`/mi/data2T/zlong/tide-execution-flows`. Job unit:
`tide-execution-flows-NAME.service` in background.slice; artifact symlink
`artifacts/execution-flows-NAME`. Inspect `TASK/runs/NAME/status.json`, `task.log`,
`verified/result.json` or `profile/result.json`; terminal status plus expected
outputs are required. Never modify old frozen source/builds. Last disk check295GB
free data/26GB root; recheck before large writes. Build jobs2,CPU/BLAS threads1.

Module `libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
The user-authorized public /opt stack overrides dated personal-anaconda defaults.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0. Retain module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392;16 chips64GiB. Cooperative leases choose
physical devices, code uses logical npu:0. No preselection/hardcoded physical ID.

```
python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT [--commit SHA] [--npu --max-wait 120] -- timeout --signal=TERM --kill-after=10s 900s '{python}' scripts/COMMAND ...
```

Placeholders {python},{base},{source},{out}. Clean qualification uses --commit.
Full build: `scripts/build_device_control.py --core-build {base}/builds/origins-npu-clean01
--build-dir {base}/builds/NEW --ascendc-soc Ascend910_9392 --jobs 2`, bound1800s.
`norm32_after_core.py` is a600s bounded dependency wrapper; submit gate dependencies
only after build advances. No card is held while waiting for a build.
Profile summaries[].engines is a dict,operators_by_type a list,inputs contains CSV
hashes. Do not print entire profile JSON (large raw task arrays). Handoffs use
scripts/durable_records.py atomic fsync/read-back. Re-entry: git status --short
--branch;python scripts/status.py; inspect actual job terminals before next action.
