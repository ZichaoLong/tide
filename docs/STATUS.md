# Current handoff

Updated 2026-10-02. **ACTIVE; continue autonomously.** User authorized continued
implementation, commits and pushes. No pause instruction; no subagents.
Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository, branch graph-execution-foundation.
Implementation 48e44b0 and evidence ca559b5 are committed/pushed and qualified.
Uncommitted valid-prefix retained journal increment below is not built/tested.
Reference repositories and ObsidianVault remain read-only.
[execution-flows.md](execution-flows.md) is authoritative; [ROADMAP F1–F7](ROADMAP.md)
is the sole backlog. Overall goal incomplete.

## Contract

Candidates independently consume common inputs/parameters/initial state; never
reference events/routes/results/gradients. General online greedy covers legal
family topology/input including positive-delay PDG feedback. Preserve int64,
stable order, duplicate edges, missing/zero messages, None/zero gradients,
complete continuation and explicit differentiation boundaries. Performance:
PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Five presets plus fine switches. FP32 main, FP16
separate. Python resident is a native C++/CANN client. No convergence requirement.
Implementation commit → immutable affected qualification → separate evidence
commit; push each. Contract outranks run-ml-experiments; reuse minimal records.
Do not rerun unchanged8,954 CPU checks. Own heavy timings are serial.

## Latest qualified increments; do not repeat

- Training diagnostics:ca26b47/af263b5, public51/four component cells.
- Eager sample chunks:e6cc52b/e05fe80, CPU60/NPU38.
- Device gradient accumulation:830904b/a3c8d4b, Python21/native384windows.
- Device continuation switching:c96ebcd/d2a425d, Python30/native768windows.
- Resident sample slicing:75543a7/8d97736, CPU15/NPU45; actual representative
  Attention whole→B2 allocator2.244→1.840GiB(-18.0%),0AiCPU profile.
- Optional compact snapshots:f8cc052/ca54c4a, Python18/four native cells;
  saved tensor storage144,506,048→2,755,300bytes,0AiCPU profile.
- **Per-device total live context pool48e44b0**, clean source
  TASK/sources/context-pool-clean01. All eight jobs PASSED:
  build-context-pool-{standalone,python,consumer}-clean01,
  context-pool-{cpu,npu,components,profile,memory}-clean01.
  CPU17,NPU64,no skips; four dense/compact FP32/FP16 native cells,
  32trajectories/768windows/96updates. Independent actual two-card consumer trace
  46,432Vector/1,797AI_CORE/607MIX_AIV,0AiCPU. Same AttentionD128/B8/T4/V257,
  fourB2 slices,oneFP32AdamW/two connected windows: dense allocator1,975,361,024→
  compact1,868,793,344bytes(-5.39%); saved state144,506,048→2,755,300bytes.
  Same loss5.612767696380615/3145events/64outputs/cut,within estimates.
  Audit launchers/context_pool_evidence.py passed.
  [Evidence](evidence/resident-context-pool-20261002.md).

Pool flag --resident-context-bytes0 keeps dense. Positive enables compact
snapshots, caps all simultaneously saved tensor/index bytes on EACH logical NPU,
and separately reserves packing workspace. New handle.device_bytes and optional
snapshot_device(...,device_budgets={index:bytes}) check all cards before payload
copies. Empty/omitted mapping keeps total-only API; older C++ overloads preserved.
Offline plan_execution_flow.py now accepts sample and pool controls too.
Dynamic nonzero extents synchronize at explicit detached snapshot boundaries;
no payload/index/length-vector export, no per-event host scheduling. Samples
retain logical batch/global IDs/loss denominator/one shared update. FP32 and
FP16 numerical policies unchanged. Live KV and retained tapes remain dense.
This is memory calibration, not new-source throughput or original-wide evidence.

## Finite representative matrix — complete

All ten required family/client/schedule submatrices have completed and passed
source/binary/input/raw-record audit. Last parent matrix-remaining02 PASSED at
05:45:32UTC and released its one-device lease. Old matrix-remaining01 remains
CANCELLED at a completed-child boundary; its completed children are retained.
Settle/Python prefill+streaming add40pilot+72confirmation processes at frozen
80dae6e14d41614d0cdb1056bb39b57ca10d07ed. Reviewed report:
[evidence/representative-settle-python-20261002.md](evidence/representative-settle-python-20261002.md).
Each submatrix uses five-preset pilot then CPU/selected mixed/resident,three
fresh processes,one continued warmup/three measured two-window steps,FP32.
Settle/Python resident warm ratios2.698–8.848× versus default Python CPU;
cold Add inference remains slower. These are not ratios against tuned LibTorch
CPU, not later sliced/compact-source timings and not original-wide evidence.
No measured processes need rerunning. Parent boundary hold records are resumed;
there is no live matrix recipe to signal.

## Next work and environment

Prioritize staged scale execution and retained/reverse memory. Original-wide inference now has feasible static
plans with explicit compact pools; validate larger shapes before original-wide.
Prepare only a bounded staged job and avoid interfering with the running matrix.

Staged scale job RUNNING/original-wide at TASK/runs/wide-inference-staged01;
unit tide-execution-flows-wide-inference-staged01.service,dispatcherPID369638.
Lease physical1,2,3,4,5,7,9,11→logical0..7. Source/helper frozen for its lifetime.
D512/B32 stage PASSED (44.877s outer process,11.432s complete step); original-wide
is running. These are capacity/execution observations, not throughput recommendations.
Frozen48e44b0/qualified consumer; helper launchers/wide_inference_stages.py.
It waits at most2400s for matrix-remaining02 to PASS before requesting8NPUs,
queue cap120s. Then D512/B32 stage (600s) followed only on success by original
D2048/B51217.5B Attention inference (1800s). Both FP32 resident TimedDAG/prefill,
one cold complete step/two connected windows; sample4,KV256,queue/arrivals2048,
8GiB compact pool/card,60GiB total/card,head512MiB,workspace capability512GiB,
locality. Logical original batch is unchanged. No throughput recommendation.
Driver/whole-run admission rechecks live memory. Each timed child has its own
process group killed on timeout; no blind retry. Entire dispatcher cap6000s.
Stage packet TASK/inputs/wide-inference-stage512-b32-01; original packet unchanged.
Static wide estimate53.798GiB maximum/8cards is planned, NOT execution evidence.
Exact command/environment are retained in
TASK/launchers/wide-inference-staged01.sh and TASK/runs/wide-inference-staged01/status.json.
Inspect status.json/stages.json/queue.json/task.log; no devices while dependency waits.

Training still needs reductions/calibration: padded retained journals and reverse
arenas plus parameter/gradient copies dominate, beyond saved contexts. Investigate
valid-prefix journal retention (different windows may have different extents),
parameter-bank lifetime sharing, and complete per-card admission. Never loosen
estimates without an allocation/lifetime derivation and independent validation.
Automatic sample admission, eager mixed multi-card placement and actual wide
execution/comparisons remain open. F7 final migration/evidence audit pending.

Uncommitted retained-journal increment (development builds RUNNING; no runtime validation yet):
- New tools/device_online/retained_journals.{h,cpp} creates device-derived valid
  prefix views once per count; preserves tensor aliases and prefix row IDs.
  Empty journals keep one unused sentinel row. Pending/outputs/KV unchanged.
- Retained tape/sharded tape overloads optionally pack those views before cloning.
  Public training owners enable it under existing aggressive chunk policy;
  conservative remains dense. Admission reserves the next dense window BEFORE
  advance; after copying, charge actual tape+state bytes instead of dense padding.
  Backward adds retained_dense_bytes/retained_compact_journals stats.
  Python/C++ consumer planners separately charge prefix metadata workspace;
  dense retained envelopes remain. CPU capacity parity test covers this field.
- CMake source list, native sharded-session check flag --compact-journals and
  registered peer-resident-compact-journals FP32/FP16 check/profile updated.
- New tests/test_resident_retained_journals.py: six independent CPU VJP/update
  cases plus empty windows; helper accepts chunk_policy. Sample-slice Attention
  tests select aggressive for both clients/FP32/FP16. Contract updated.
- Diff whitespace and Python AST only. Build helper launchers/build_retained_journals.py
  running as build-retained-journals-{standalone,python}-dev01 on frozen
  TASK/sources/retained-journals-dev01; reuses qualified context-pool runtime core/CANN and
  unchanged host units, rebuilds three archive/four owner units plus native check.
  No public struct-layout change. Review before freezing retained-journals-dev01.
- Do not compile/test during measured matrix children. Staged inference will
  launch automatically afterward; coordinate new heavy work with that job rather
  than silently contaminating it. New dev gates: seven new public cases, dense
  projection-retention3, CPU capacity17, actual sample-slice34, compact native FP32/FP16 broad
  trajectories/CPU references. Consumer CMake can reuse unchanged public-header
  objects. Then implementation commit, clean gates/profile/memory and evidence.
  Complete consumer capacity still charges dense tapes; do not loosen its
  estimate just to admit a wide run. New compaction is not wide qualification.

TASK=/mi/data2T/zlong/tide-execution-flows. Public module
libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized /opt stack supersedes dated personal guide. Preserve module
PYTHONPATH,prepend frozen source/python. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0.16logical64GiB Ascend910_9392;lease/remap only.
freeze_run.py uses detached background.slice/Nice10,2buildworkers,bounded tasks.
Last disk free:data200GiB/root12GiB. Current timing lock TASK/online-measurement.lock.
Atomic handoff writes use durable_records.replace_text.
Qualified libraries TASK/builds/context-pool-{standalone,python}-clean01.
Qualified installed consumer TASK/builds/context-pool-consumer-clean01/consumer/tidegraph-online-bench.
Core builds placement-{cpu,npu,npu-python}-clean01; eager binaries
sample-chunks-{cpu,npu}-clean01/consumer/tidegraph-online-bench. Distinct runtime
owners; source/header/options-checked object reuse only. New builds passed loader
closure. Do not modify any frozen source or active helper.

Original wide packets TASK/inputs/fullsize-{add,attention}01/workload.json:
480body/2208edges,D2048/B512/T12/V50304,9,468,053,696 or17,521,117,376params,
clear=true/stride17. Region(budget) defaultsobserve_all=true: unselected nodes
retain KV; clear applies only to selected nodes. Do not change logicalB512 or
lose KV and claim original-wide passed. Static estimate refusal is not physical
impossibility or completion. CUDA real hardware and other CANN tuples target-pending.

Historical historical-cpu-attention-01 intentionally remains SIGSTOP and holds old
timing.lock; never resume/kill/clean it. Historical1.6438× means faster throughput,
not current online-flow evidence. Prior restricted work remains on pushed
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
