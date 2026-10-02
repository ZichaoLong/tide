# Current handoff

Updated 2026-10-02. **ACTIVE; continue autonomously.** User authorized continued
implementation, commits and pushes. No pause instruction; no subagents.
Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository, branch graph-execution-foundation.
Implementation48e44b0/evidenceca559b5 are qualified; matrix evidence270ee2e is committed/pushed.
Valid-prefix retained journals passed all development gates; implementation ready
for commit. Immutable qualification and separate evidence remain next.
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

## Current increment and next action

Valid-prefix retained journal implementation is ready to commit after all
relevant development gates PASSED. No clean-source qualification yet.
- Aggressive policy saves only valid prefixes of event/source/emission/cache
  journals at completed-window boundaries. Empty journals keep one unused row;
  preserve prefix IDs/aliases and complete pending/output/KV layouts. Dynamic
  nonzero extents synchronize at this boundary, never per-event scheduling.
- Admission reserves the next dense window before advance, then charges actual
  retained tape/state bytes. New retained_dense_bytes/retained_compact_journals
  statistics; conservative mode unchanged. Consumer planning still charges dense
  retained envelopes and separately reserves metadata packing workspace.
- All three builds PASSED: build-retained-journals-{standalone,python,consumer}-dev01,
  frozenretained-journals-dev01. Changed host units only, byte-checked unchanged
  core/CANN/public-header dependencies, fresh links/loader closure.
- Frozenretained-journals-dev02 corrects only test configuration (source code
  unchanged). Public-dev02:10 passed (new7+dense projection3). CPU-dev01:17passed.
  Samples-dev01:34passed. Components-dev01:FP32/FP16 each32trajectories,
  512windows/128updates,independent CPU FP32/FP64,SGD/AdamW,cache/control profiles,
  explicit owner maps and two-device to legacy single-device resume.
- Preserve public-dev01 FAILED:four unsupported controlled slot-affine test
  combinations and one missing explicit seal. Corrected tests use supported
  broadcast for HST/SOFTP and declare the empty-window seal. No tolerance change.

Next: commit/push implementation, then freeze retained-journals-clean01 at that
exact commit. Build both runtimes with launchers/build_retained_journals.py
NAME --runtime standalone|python --reuse-host retained-journals-RUNTIME-dev01.
Consumer uses build_capacity_client.py --reuse-client retained-journals-consumer-dev01.
Clean gates:CPU17,combinedNPU44,components2; separate actual-consumer two-card
profile via profile_retained_journals.py and one-card allocator comparison via
retained_journals_memory.py. Use freeze_run.py, bounded jobs/no blind retries.
Allocator calibration can overlap separately leased capacity/correctness jobs;
no timing recommendation from these runs. Matrix has terminated. Evidence must
be a separate reviewed commit. Do not relax full-size memory admission based on
journal compaction alone; complete planner retains dense bounds.

## Active eight-device scale execution

Unit tide-execution-flows-wide-inference-staged01.service;
TASK/runs/wide-inference-staged01/{status,stages,queue}.json and task.log.
Helper launchers/wide_inference_stages.py, source48e44b0 at context-pool-clean01,
qualified installed consumer context-pool-consumer-clean01. Never edit live helper.
Lease physical1,2,3,4,5,7,9,11→logical0..7. D512/B32 stage PASSED:
1,133,889,728parameters,44.877s construction,11.432s complete step. Original
17.5B D2048/B512/T12/V50304 Attention is still RUNNING (construction last observed).
One cold complete inference step/two connected windows,FP32,TimedDAG/prefill,
locality/sample4/KV256/queue-arrivals2048/outputs512/trace16384/kv-trace65536,
8GiB compact pool/card,60GiB total/card,head512MiB,forward capability512GiB.
Original packet/logical batch unchanged. Static max53.798GiB/8cards is NOT a pass.
Original-wide cap1800s from about05:46:40UTC; child process group cleanup on failure,
no blind retry. This is capacity/execution evidence, not throughput; dev builds
and gates on other devices overlap. Inspect terminal outputs, model/input/cut,
allocator estimate, complete output counts and finite loss before recording pass.
Do not reuse the historical old timing.lock.

## Remaining goal work

Original-wide real execution and finite CPU/screened-mixed/resident comparisons;
training retained/reverse/gradient memory and safe complete admission; automatic
sample admission; eager mixed multi-device parameter/payload placement. F7 final
migration/evidence audit. CUDA execution and additional tuples require target
machines. Do not replace these with a static refusal or representative result.

Original packets TASK/inputs/fullsize-{add,attention}01/workload.json:
480body/2208edges,D2048/B512/T12/V50304,9,468,053,696 or17,521,117,376params,
clear=true/stride17. Region(budget) defaultsobserve_all=true:unselected nodes
retain KV;clear applies only to selected nodes. Do not reduce logical B512 or
silently drop KV to claim original-wide passed.

## Environment and preservation

TASK=/mi/data2T/zlong/tide-execution-flows. Public module
libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized /opt stack supersedes dated personal guide. Preserve module
PYTHONPATH,prepend frozen source/python. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0.16logical64GiB Ascend910_9392;lease/remap only.
freeze_run.py:detached background.slice/Nice10,2buildworkers,bounded tasks.
Last free:data195GiB/root12GiB. Current formal timing lock TASK/online-measurement.lock.
Atomic handoff via durable_records.replace_text. Qualified resident libraries
context-pool-{standalone,python}-clean01; consumer context-pool-consumer-clean01.
Core placement-{cpu,npu,npu-python}-clean01; eager clients sample-chunks-{cpu,npu}-clean01.
Standalone and Python-owned runtimes remain separate; no library mixing.

historical-cpu-attention-01 intentionally SIGSTOP,holding old timing.lock;
never resume/kill/clean it. Historical1.6438× was faster throughput,not current
online-flow evidence. Historical restricted work stays archived on pushed
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
