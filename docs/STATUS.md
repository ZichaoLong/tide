# Current handoff

Updated 2026-09-30T13:21:45.719542+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository /home/zlong/llm/graph-execution-foundation, real path
/var/tmp/zlong-graph-execution-foundation/repository; branch graph-execution-foundation.
HEAD b668f2f committed/pushed after all27 development gates and profiling passed. Clean pooling qualification691cb31 is recorded. No subagents. Reference repositories and ObsidianVault
are read-only. Preserve the older accelerator_scale/flow dirty work listed below.

## Authoritative scope and next actions

[execution-flows.md](execution-flows.md) is the execution contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Latest user alignment overrides run-ml-experiments: minimal
source/input/config/environment identity, raw results/failures, synchronized timing,
bounded resources/stops. Reuse records; Trackio must not block implementation.
Training means forward/backward/VJP/optimizer/continuation and complete throughput,
not convergence or model effects. No pending permission or pause.

Candidates independently consume input/state/parameters; no numerical CPU route
prepass, whole-window potential expansion or fixture-specific schedule. General
online greedy prefill accepts each family's legal topology/input, including PDG
positive-delay feedback. Device residence includes actual online decisions.

Priorities: independent correctness/failure gates and bounded profiling; actual
modules/attention/KV and complete safe chunking; public five presets/matrix, peer
progression and resident backward/optimizer; then representative/full-size timing.
PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill × inference/
complete training. CPU,mixed A/B/C,resident; FP32 main,FP16 separate,CPU FP64 oracle.
Three fresh processes before performance recommendations. CUDA device execution is
target-machine pending. **F1–F7 are not complete.** Raw device flow is still
single-device FP32 HARD inference; no resident backward/optimizer.

1. Clean event implementation b668f2f PASSED full build/four CTests,all27 cells,
   and independent profile. All three clean01 runs terminal0; evidence recorded
   in docs/evidence/device-event-attention-20260930.{json,md}. Commit/push evidence.
2. Working tree now adds shared device key-axis attention tiling with global
   online softmax normalization, configurable key rows and budget shrinkage.
   Integration and new tests are ready for build-device-key-tile-dev01 from
   key-tile-dev01, build device-key-tile-dev01. Full build/CTests, jobs2,1800s.
   Build is RUNNING, not verified. Dense path retained; key rows1/7/128/300, ragged
   KV0/5/257, MHA/GQA/MQA, global denominator and extreme logits are tested.
   Existing fiber/pool/event gates now exercise key tiles1/7.
   Continue safety, node-time batching, remaining modules, public presets/matrix,
   multi-card and resident training. F1–F7 remain open.

## Qualified event attention: b668f2f

Working tree:packed_event_attention.{h,cpp},event_cache.cpp,
ascendc/tide_event_{plan,payload,indices,cache}.cpp,event_attention_check.cpp;
ContentFlow/profile/export/CMake/check registry and content-flow docs integrated.
Canonical Node.memory="attention" implements documented event-gqa-v1 semantics.
Static groups share query/KV head geometry; persistent KV is compact,one row per
actual event. Device decides actual chunks,window retention and GQA head indices.
Shared proposals use overwrite/copy,never adding stale scratch arrays. Event and
fiber attention coexist. Diagnostics export all old/proposal/comparison/next
slots; selected-only/clear and lean continuation keep their independent caches.
One complete attention region frame per stage remains the adapter fallback.
Group minimum workspace is reserved before fiber chunks grow. No VJP/training.

build-device-event-dev01 full snapshot build PASSED;dev02/03/04 are isolated
incremental builds with checked source/binary hashes. Snapshot event-dev04,
build device-event-dev04,matching core origins-npu-clean01. Only immutable old
objects reused; no prior snapshot/build was modified. Final clean rebuild required.
Device-event-gates-dev04 PASSED:98 MHA/GQA/MQA/shape/window anchors,192 complete
windows,40 lifecycle windows,5 refusals. Widths1/4/7/33/257,heads1/3/4,window0/1/3,
chunk1/4,compact KV,message vs event count,three Read modes,feedback/DAG,mixed
fiber/event profiles,InputOrigin,large int64 counters,clear/selected-only,
periodic clocks,window recycling,restore and lean continuation. Strict tolerances
rtol1e-5/atol1e-6 and exact discrete comparisons unchanged.

Device-event-regression-dev04 PASSED all27 cells, including the event gates above.
Device-event-profile-dev04 PASSED:92569 AIV +2530 AI Core tasks, no AiCPU/fallback.
These establish placement and parity only, not throughput or resident training.
Source identity matched the working device sources byte-for-byte before commit.

Retained failures:
- gates-dev01: semantic doc name event-gqa-v1 mistaken for existing memory value
  attention. Failed in CPU reference configuration before numerical comparison.
  Fixed only profile recognition/test config;profile-dev01 dependency-failed.
- gates-dev02: numeric mismatch;profile-dev02 dependency-failed without a card.
- gates-dev03: failure-only diagnostics isolated width1/heads1/kv1/window0/chunk1/
  empty cache. K=.125,V=.25 correct,but read=.083550 vs expected.125, varying
  between processes. Scalar index/mask words written by32 AIV blocks shared cache
  lines,causing writeback races. dev04 changes only indices launch to one AIV block;
  all98 anchors and complete windows then pass. Matrix/vector payload work remains
  packed. Original failed records/snapshots stay;no formula or tolerance change.

## Qualified implementation: general post-attention pooling

Implemented sum/mean/linear/active/all-softmax under the five existing profile
names. Coefficients apply after query attention, before output projection/bias;
QKV/cache/source presence are unchanged. Device loops pack actual softmax events,
normalize actual/all logical domains and place coefficients. Physical aliases do
not enlarge the domain. Pool workspace is reserved before selecting query chunk.
Files: packed_fiber_pool.{h,cpp},ascendc/tide_fiber_pool.cpp,fiber_pool_check.cpp;
integrated profile/cache/payload/build/check registry and content-flow docs.

- build-device-fiber-pool-dev01 PASSED clean-from-snapshot full build/four CTests.
- device-fiber-pool-gates-dev01 FAILED at known illegal fixture clock phases,
  after80 anchors and17 domains passed. Original logs/snapshot are retained.
- build-device-fiber-pool-dev02 PASSED isolated incremental build; only corrected
  fiber_pool_check.cpp recompiled; production source/objects unchanged, hashes
  and link commands recorded. Added empty-domain and refusal coverage.
- device-fiber-pool-gates-dev02 PASSED all26 cells, including80 analytic/alias/cache
  anchors,18 wide/extreme/empty-domain cases,160 complete windows,5 refusals.
- device-fiber-pool-profile-dev02 PASSED on physical9/logical0:76953 AIV+1586
  AI Core tasks,no AiCPU/fallback. Placement only, not throughput.

TASK/sources/fiber-pool-dev02;TASK/builds/device-fiber-pool-dev02;core origins-npu-clean01.
Gate rtol1e-5/atol1e-6, exact discrete values unchanged. Widths1/7/33/257,heads1/3,
chunk1/4,missing vs zero,negative/zero weights,source aliases,257-slot domains,
missing dominant logits,old cache,three Read modes,feedback/DAG,mixed profiles,
selected-only/clear,large clocks/counters,InputOrigin,restore and lean continuation.
Clean build-device-fiber-pool-clean01 and device-fiber-pool-gates-clean01 PASSED
all26 cells;device-fiber-pool-profile-clean01 PASSED. Evidence manifest records
76953 AIV+1586 AI Core and no AiCPU/fallback on clean691cb31.
Current scope remains single-device FP32 HARD inference, one attention region frame
per stage. This does not close node-time batching,key-axis tiling,FP16,VJP/training.

## Current immutable qualification: bounded same-fiber attention

Implementation081f567: packed_fiber_attention.{h,cpp},fiber_cache.cpp,fiber_check.cpp,
six Ascend C kernels and ContentFlow integration. Source fiber-clean01 at081f567;
build device-fiber-clean01; matching core origins-npu-clean01.

- build-device-fiber-clean01 PASSED,four CPU CTests,jobs2.
- device-fiber-gates-clean01 PASSED all25 cells (physical9 -> logical0).
- device-fiber-profile-clean01 PASSED,64617 AIV+1984 AI Core,no AiCPU/fallback
  (physical13 -> logical0). Qualification records are terminal, clean081f567.

All records: TASK/runs/NAME/{status.json,task.log}, result verified/result.json or
profile/result.json. Qualify the clean terminal results, not development records.
Do not print entire profile JSON; it includes large raw task arrays.

Behavior: lh-fiber-attention-sum-repeat-v1 consumes real per-source weighted rows
in local slot order. NPU metadata chooses QKV/query/output chunks; batched matrix
multiplication,gather,head permutation,softmax and vector payload tasks run in the
runtime model. Each query sees all old keys and all current fiber keys; no triangle.
Exact repeated bias decay uses local ticks. Bounded owner arenas retain key/value/
log_bias and int64 lengths; observation count is distinct. Selection determines
adoption; clear empties the cache after saving comparison for Full. Optional old/
proposal cache journals export all four state-slot views without feeding execution.

One complete region frame per attention owner per stage is an explicit adapter
fallback, including content Read. Independent owners/message rows remain batched;
other regions retain legal prefixes. Full attention node-time state batching and
key-axis tiling are pending. Queries are chunked with the complete global denominator.
kv_rows128 default; attention_chunk_rows8; kv_trace_rows4096 for diagnostics. Actual
query limit can shrink under the byte budget, reserving at least one tanh Full row.
Capacity11,stage diagnostic12,tick-work8 refuse explicitly; cumulative journal uses
its existing refusal. Lean windows keep live KV without exporting it.
Other fiber pools are currently development work;event-GQA/window,FP16,training
are not implied.

Development build-device-fiber-dev04 PASSED CPU/loader checks and reused unchanged
production objects with recorded hashes. device-fiber-gates-dev04 PASSED all25
cells:16 analytic/shape anchors,192 full windows,6 refusals,40 lifecycle windows.
Coverage includes width1/7/33/257,heads1/3,mixed owners,chunk1/4,slot permutation,
InputOrigin,feedback/DAG,content/old/proposal Read,scalar/vector Read/state,
clear/selected-only/empty selection,large clocks/counters,periodic phases,
cache length vs observations,continuation/restore/lean export and repeated recycling.
Strict rtol1e-5/atol1e-6 and exact discrete comparisons unchanged.
Profile-dev04 PASSED64617 AIV+1984 AI Core tasks,no AiCPU/fallback; placement only.

Retained development failures:
- build-device-fiber-dev01: Ascend C Muls cannot accept a __gm__ scalar reference;
  fixed by reading the scale into a plain float. Dependent gates-dev01 failed
  without acquiring NPU. Original source/logs remain.
- gates-dev02: attention budget left too little for one tanh Full row at width257/
  chunk4. Fixed by reserving PackedFull's own minimum footprint before growing
  the attention chunk. build-dev03 was an isolated incremental build of three
  affected C++ files; generated kernels remained byte-identical.
- gates-dev03: final lifecycle fixture used invalid zero region budget. Corrected
  test uses positive-v1 and negative content Read for legal empty selection. Build
  dev04 recompiled that test only; no runtime formula/tolerance change.
The independent fresh clean build above closes incremental-build provenance.
Clean qualification repeats16 anchors,192 windows,6 refusals,40 lifecycle windows.

## Earlier immutable qualifications (committed/pushed)

- Norm-FP32 public core0e66d89:8901 CPU tests/eight CTests,Python18/native18 NPU
  fixtures and standalone6 combinations/18 updates. Host-scheduled tensor training,
  not resident. [Evidence](evidence/norm32-20260930.md).
- Native origin stable-sort correction832a881: eight CTests,two CPU Aggregate
  gates,362 focused tests and NPU Aggregate. Old0e66d89 regression truly fails;
  wrapper success means the old failure was observed. Not a new full regression.
  [Evidence](evidence/native-origin-order-20260930.md).
- Device clocks49ff108: all19 cells;480 mapped/18 multi-phase windows/12 refusals.
  [Evidence](evidence/device-clock-20260930.md).
- Vector Read8a735ef: all20 cells,530 windows,96173 AIV+1083 AI Core,no AiCPU.
  [Evidence](evidence/device-read-20260930.md).
- LH Fullcd03ca8: nine profiles,40 components/96 strict windows,all21 cells.
  Low-variance component LayerNorm uses independent FP64 conditioning evidence;
  full graph tolerances remain unchanged. [Evidence](evidence/device-lh-full-20260930.md).
- Device origins1be3619: all22 cells,24 ordering cases/128 windows/8 refusals,
  27256 AIV,no AiCPU. [Evidence](evidence/device-origins-20260930.md).
- Slot projection/phase emission6445121: all23 cells,16 components/66 windows/
  6 refusals,14375 AIV+176 AI Core. [Evidence](evidence/device-emission-20260930.md).
- SwiGLU74cec2f: all24 cells,16 components/256 windows/4 refusals,53667 AIV+1195
  AI Core,no AiCPU/fallback. [Evidence](evidence/device-swiglu-20260930.md),evidence
  commit5bfe36d. These device increments qualify finite FP32 HARD inference only.

## Preserved work and historical timing

Preserve dirty tools/accelerator_scale/flow_*,bounded/resident/peer files,their CMake
and scripts/build_accelerator_scale.py,scripts/benchmark_execution_flow.py,
verify_execution_flows.py,tests/test_flow_semantics.py. These are limited DAG/
rank-aligned consumers, not revised general-online delivery. flow-dev05 CPU24
passed;NPU18 FP32 passed before FP16 resident Add gradient failure. dev06 builds
passed,no gates. Peer CPU8/NPU16 passed separately. Do not stage/clean this work.

Historical tide-execution-flows-historical-cpu-attention-01.service remains SIGSTOP.
TASK/runs/historical-cpu-attention-01/pause.json overrides its running status. It
holds host memory and TASK/timing.lock. Do not blindly resume/stop it; resolve the
interrupted timing/lock deliberately before formal timing, preserving logs.
Historical Add complete training CPU78.793172/NPU4 47.932888 ms/token is NPU
throughput1.6438× faster; it does not qualify this new resident backend. Attention
has no valid CPU complete-training ratio. Older failure reproducers/logs remain.

## Environment and durable operation

TASK=/mi/data2T/zlong/tide-execution-flows. Job symlink artifacts/execution-flows-NAME;
unit tide-execution-flows-NAME.service in background.slice. Frozen snapshots/builds
must not be overwritten/modified. Last disk check302GB data/27GB root free; recheck
before large writes. Build parallelism2;CPU/BLAS threads1.
Module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized public /opt stack overrides dated guide defaults. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0. Retain module PYTHONPATH and prepend snapshot/python.
SoC Ascend910_9392;16 chips64GiB. Cooperative leases choose physical devices and
remap to logical npu:0; no hardcoded placement.

```
python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT [--commit SHA] [--npu --max-wait 120] -- timeout --signal=TERM --kill-after=10s 900s '{python}' scripts/COMMAND ...
```

Placeholders {python},{base},{source},{out}. Existing snapshots reused read-only;
qualification requires --commit. norm32_after_core.py is a bounded600s dependency
wrapper; do not reserve NPU while waiting. Profile summaries[].engines is a dict;
operators_by_type is a list; summaries[].inputs has raw CSV hashes. Preserve failed
runs; terminal status plus expected outputs are required before reporting success.
Handoff writes use scripts/durable_records.py atomic fsync/read-back. Re-entry:
git status --short --branch;python scripts/status.py;inspect actual terminal records.
