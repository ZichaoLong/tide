# Current handoff

Updated 2026-09-30T16:51:08.036979+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
HEAD **d412541**, public placement implementation committed and pushed.
Aggregate evidencefdd2b86 and fiber evidence7b43f4d pushed.
Event batch evidence28295c1 pushed; implementation26aa09f.
Memory evidence0b724a3 is pushed. No pending authorization or pause. No subagents.
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

1. Completed and pushed: event batches26aa09f/evidence28295c1; fiber batches
   c83aec3/evidence7b43f4d; normalized Aggregate f1b7168/evidencefdd2b86.
   Each exact-commit clean full build/four CTests/full component regression/profile
   passed. Latest Aggregate32 cells;180 anchors/restores,160 general windows,
   4 boundaries,7 domains;67,422 AIV+139 MIX_AIV+223 AI Core,no AiCPU/fallback.
   Fiber31 cells;96 anchors,480 windows,31 multi-time windows,3 saturation checks;
   clean221,788 AIV+8,181 AI Core. Reports under docs/evidence. Those jobs terminal0.
2. Public host placement implementation d412541 is committed. C++ placement.h + placement*.cpp,
   descriptor_device validation, recognized built-in adapters, native binding;
   Python ExecutionPlacement + GraphRuntime/Native integration, same parameter
   leaves/checkpoint names; controls/ranking/read placements and scoring precision.
   Five preset resolver can describe resident, but host adapter refuses device
   event progression. Resident integration/FP16/multi-device/matrix still pending.
   cpp/test/placement.cpp,tests/test_placement.py and profile_execution_placement.py
   are new. Core fingerprint changed: do not reuse origins-npu-clean01 for new code.
3. Development evidence BEFORE the latest region-policy correction:
   - build-placement-npu-dev01 and build-placement-cpu-dev02 passed full builds.
   - placement-cpp-{cpu32,cpu64,npu}-dev01 failed illegal test budget0; preserved.
   - isolated new checker build-placement-check-dev02 passed after test correction;
     placement-cpp-cpu32-dev02 and cpu64 each49 schedules/147 updates passed;
     placement-cpp-npu-dev02 passed121 schedules/363 updates,terminal0.
   - NPU int64 ArgSort explicitly reports AiCPU. No lossy float count conversion.
     This is mixed host-dispatched execution,not all-AiCore or resident evidence.
   - placement-python-cpu-dev01 failed test body-vs-encoded Settle naming after2;
     dev02 failed test Full lookup on an unselected node after12. Production unchanged.
     Corrected placement-python-cpu-dev03 passed39; npu-dev03 passed20,terminal0.
     Three-family Python forward/backward/SGD/resume/schedule-switch passed.
4. Latest production correction: C++ region budget/count priority now comes from
   request.layout (as original/Python),not cached construction policy. New test
   reuses the model with a compatible changed layout. All following use frozen
   placement-dev03. build-placement-patched-dev03 and
   build-placement-cpu-patched-dev03 PASSED. Their task-local
   placement_policy_build.py allows only cpp/src/placement_region.cpp to differ,
   checks all original production/binary hashes,patches a copied static archive,
   relinks new checker/CPU binding in new directories. Originals stay unchanged.
   Builds: placement-patched-dev03 and placement-cpu-patched-dev03;
   checker-build.json records derivation,hashes,commands. Development evidence only.
5. Corrected development gates PASSED,all terminal0: placement-cpp-cpu-dev03 (two CPU dtypes),
   placement-cpp-npu-dev03 (NPU FP32),placement-public-cpu-dev03 (new placement +
   Read/Region/greedy-library regression with corrected CPU binding),461 tests.
   CPU C++49 schedules/147 updates per dtype;NPU C++121/363.
   Fresh build-placement-npu-python-dev03 PASSED (1800s,jobs2),source
   placement-dev03,build placement-npu-python-dev03. placement-public-npu-dev03
   PASSED all26 FP32 Python/native cases. All NPU runs one lease,queue120s.
6. placement-profile-dev03 PASSED,terminal0,from frozen placement-profile-dev03.
   Corrected checker3 schedules:4,079 AIV+862 MIX_AIV+86 AI Core+66 AiCPU tasks.
   AiCPU:Sort(INT64)30 tasks/1834.48us;ScatterElements(BOOL,INT64,BOOL)36/3171.42us.
   Their38.1% is summed device task time,not complete wall-time fraction. No host
   fallback diagnostic. Mixed host dispatch remains explicit; no speed claim.
7. Running exact-commit qualification from placement-clean01 at d412541:
   build-placement-cpu-clean01 -> builds/placement-cpu-clean01 (CPU+bindings),
   build-placement-npu-clean01 -> builds/placement-npu-clean01 (standalone),
   build-placement-npu-python-clean01 -> builds/placement-npu-python-clean01.
   Each fresh build1800s,jobs2 (at most6 aggregate workers),no NPU lease for builds.
   After terminal0: CPU full scripts/verify.py plus10 CTests (bound1800s);
   NPU standalone placement,Read-precision,accelerator checks (900s,one NPU);
   NPU public test_placement.py FP32/all26 cases (900s,one NPU); separate profile
   using clean standalone build. Queue wait120s;dependency before lease.
   All three builds passed terminal0; placement-public-npu-clean01 passed all26
   FP32 cases,terminal0. Submitting placement-cpu-clean01 (full verify,1800s),
   placement-ctest-clean01 (10 CTests,300s),placement-standalone-npu-clean01
   (three checks,900s),placement-profile-clean01 (separate trace,900s).
   NPU jobs one lease each,queue120s; source placement-clean01/d412541.
   Evidence must be committed separately. Do not stage old accelerator_scale dirty.
   Clean CTests10,standalone three gates,and independent profile now PASSED,terminal0;
   full CPU pytest still running. Qualification evidence awaits its terminal result.
8. Public resident integration is under development (uncommitted): optional shared
   C++ ResidentSession and Python-owned backend binding; GraphRuntime resident
   sessions; bulk CPU/NPU external payload validation/upload; explicit CPU export,
   immutable parameter checking and checked close. No resident backward or multi-card.
   Building frozen resident-public-dev01 against exact matching placement cores,
   jobs2,1800s each,no NPU lease. New builds resident-public-python-dev01 and
   resident-public-standalone-dev01. No pass claimed until build and live gates.
   Python dev01 configuration FAILED: compile definitions referenced a disabled
   standalone checker; also found wrong auto-selected Python. Both corrected;
   build-resident-public-python-dev02 uses fresh snapshot resident-public-dev02,
   explicit matching Python,build resident-public-python-dev02. Original failed
   record preserved. Standalone dev01 remains running (unaffected correction).
   New tests/test_resident_library.py covers three families/two schedules,
   EMA/event/fiber attention,CPU/NPU inputs,non-default stream,export/restore,
   mutations,capacity poison and unsupported modules. Python dev02 build passed,
   terminal0; resident-public-python-dev02 launching its20 device tests plus
   one CPU-only configuration test (900s,one NPU,queue120s). Host regression
   resident-host-cpu-dev02 passed125,20 optional NPU skips. Main-tree newer
   content_input.cpp removes dummy per-input CPU numerical validation in favor
   of metadata-only ledger validation and one packed finite check; pending rebuild.
   Added lean-no-export and loader guard tests after dev02 snapshot; not run yet.
   resident-public-python-dev02 FAILED at checkpoint save after three windows
   matched the CPU oracle: exported CPU state was validated against NPU model.
   Fixed with memoized CPU model view preserving parameter aliases,only at save.
   Old failure remains. Building input-only isolated relink resident-public-python-dev03,
   source resident-public-dev03,from immutable python-dev02;record all production
   file hashes,parent archive/binary hashes and commands,new directory only.
   Both parent builds and both dev03 isolated relinks passed,terminal0. Standalone
   resident-public-standalone-dev03 passed resident/window/content;resident64 windows.
   resident-profile-dev03 passed10,534 AIV+16 MIX_AIV,no AiCPU/fallback.
   Python dev03 then FAILED at invalid test Settle ranks (node ranks supplied
   instead of positive region ranks),after2 cases. Corrected dev04 passed12
   EMA/event-attention cases then FAILED at test fiber GQA request (fiber requires
   equal Q/KV heads). Corrected fixture uses equal heads for fiber only; full
   resident-public-python-dev05 next,23 tests,900s,one NPU,queue120s.
   Production save fix remains; all original failure logs/snapshots retained.
   resident-public-python-dev05 passed23;dev06 passed24 after adding invalid
   CPU/NPU-input transaction checks and numerical continuation after parameter
   update. All terminal0. build-resident-consumer-dev01 and resident-consumer-dev01
   PASSED: installed standalone package under a prefix with spaces,public headers
   only,three live feedback windows. Uses original standalone-dev01 library;
   final clean installed-consumer must use the final source. No new throughput claim.
   Ready to commit resident implementation,then full clean component builds from
   that commit against matching immutable d412541 placement cores (cpp hash same).
   Python build/core registration and standalone SDK remain separate.
   Complete CPU d412541 qualification is still running; no result claimed yet.
   No prior frozen source/build is modified.
   Then public resident integration,FP16/multi-device,remaining F4 adapters/peer
   progression,resident backward/VJP/optimizer and representative/full-size timing.
   F1–F7 overall NOT complete. Historical CPU Attention remains paused.

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

Earlier attention qualifications used one complete region-time frame per stage.
Event batching26aa09f and fiber batchingc83aec3 extend that with clean evidence. These reports establish parity/placement,not throughput or
training. Fiber cache bounds refuse explicitly,not evict.

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
