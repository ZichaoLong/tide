# Current handoff

Updated 2026-10-02. **ACTIVE; continue autonomously.** User authorized continued
implementation, commits and pushes. No pause instruction; no subagents.
Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository, branch graph-execution-foundation.
Implementation48e44b0/evidenceca559b5 are qualified; matrix evidence270ee2e is committed/pushed.
Implementation0fbc1b2 committed/pushed and all eight clean jobs PASSED.
Retained-journal and original-wide Attention evidencefe8a08b committed/pushed.
Fused CPU initializerbe380db committed/pushed and qualified; evidence c307fde committed/pushed. Original Add scale run passed below.
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

## Latest closed increment

Valid-prefix journals0fbc1b24ca8d8b178f83dfce32279ba23ce7e51f qualified at
TASK/sources/retained-journals-clean01. All8clean jobs PASSED:
build-retained-journals-{standalone,python,consumer}-clean01 and
retained-journals-{cpu,npu,components,profile,memory}-clean01. Audit
launchers/retained_journals_evidence.py passed. CPU17,NPU44,no skips; native
FP32/FP16 total64trajectories/1,024windows/256updates,independent CPU FP32/FP64.
Separate actual two-card consumer trace46,612Vector/1,797AI_CORE/667MIX_AIV,0AiCPU.
Same representative Attention/fourB2slices/oneFP32AdamW/two connected windows:
allocator1,868,793,344→1,455,752,704bytes(-22.10%);retained388,563,192→184,628,448
(-52.48%),identical loss5.612767696380615/3145events/64outputs/cut. Total admission
still charges dense tapes plus explicit prefix scratch; live KV remains dense.
[Report](evidence/resident-retained-journals-20261002.md).

Aggressive policy compacts candidate-owned event/source/emission/cache journal
prefixes at completed-window retention boundaries. Dynamic nonzero shapes sync
there,never per event. Preserve row IDs/aliases/empty sentinel,complete pending,
outputs/KV and VJP links. Conservative remains dense; old C++ overloads unchanged.
Retained statistics distinguish dense bounds and actual bytes. Capacity is checked
before advance. Do not relax full-size training admission from this result alone.
Dev public01 remains FAILED (four unsupported controlled slot-affine fixtures and
one omitted seal); corrected public02 and all clean cases passed without tolerance
changes. Source/runtime contracts unchanged. Other completed dev/build logs retained.

## Original-wide Attention execution passed

wide-inference-staged01 PASSED/released8NPUs06:05:57UTC;
TASK/runs/wide-inference-staged01/{status,stages,queue}.json and task.log.
Qualifiedsource48e44b0/context-pool-clean01;installed consumercontext-pool-consumer-clean01.
Lease physical1,2,3,4,5,7,9,11→logical0..7. Prerequisite D512/B32 passed.
Original17,521,117,376parameters,D2048/B512/T12/V50304,FP32 TimedDAG/prefill,
128physicalB4groups,one complete step/two connected windows PASSED:12,288outputs,
1,184,430events,cut408,finite loss21.380956649780273. Construction817.996s,
step325.278s. Per-card allocator11.777–14.562GiB(max15,635,636,736bytes),within
conservative max53.798GiB plan. CPUpeak208.770GiB;compact pools<8GiB/card.
No logical batch or KV reduction. No full-size CPU oracle,profile,warmup/repeats
or throughput recommendation; independent dev/calibration jobs overlapped.
Audit launchers/wide_inference_evidence.py passed.
[Report](evidence/original-wide-inference-20261002.md).

## Latest initializer qualification

Generic named-lcg31-v1 C++ CPU initializerbe380db7451742c3fe50b57cb02263e81c64a4d6
composes three integer affine steps modulo2^31-1 and fills only final FP32 under
existing ATen threads. Python unchanged; no graph/core/NPU changes. Frozen
source-values-clean01. All5clean jobs PASSED:
build-source-values-{cpu,npu}-clean01,source-values-{cpu,npu}-clean01,
source-values-benchmark-clean02. CPU25 (380 scalar cases+24 actual FP32/FP64
consumer comparisons);NPU30 (12resident training,12resident inference,6mixed),
no skips. Libraries reused from qualified0fbc1b2. Audit source_values_evidence.py
passed; [report](evidence/exact-initializer-20261002.md).
Three fresh CPU processes,one warmup/three measured old/new fills per shape,
full memcmp outside timer,actual model.cpp object vs priorfe8a08b ATen code:
projection0.092060→0.014736s(6.247×),QKV0.287816→0.044218s(6.509×),
head2.300079→0.362125s(6.352×). Only initialization speed,not complete construction
or graph/NPU throughput. Separate reviewed evidence c307fde committed/pushed.

Retain FAILED source-values-benchmark-clean01 (auxiliary linker omitted packet.cpp,
no measurements;v2 corrected),and wide-add-inference01 (dependency stopped before
NPU allocation). No library/tolerance change was needed. Helpers for those failed
runs remain unchanged; corrected helpers have_v2 suffixes.

## Original-wide Add execution passed

wide-add-inference02 PASSED/released8NPUs06:39:49UTC;frozenbe380db/source-values-clean01,
qualifiedbinaryTASK/builds/source-values-npu-clean01/consumer/tidegraph-online-bench.
Lease physical1,2,3,4,5,7,9,11→logical0..7. Original Add9,468,053,696parameters,
D2048/B512/T12/V50304,FP32 TimedDAG/prefill,128B4groups,onecoldstep/two connected
windows:construction182.037855s,step278.573709s,12,288outputs,1,183,429events,
cut408,finite loss30.508380889892578. Max allocator8,387,755,008bytes(7.812GiB),
CPUpeak100,636,151,808bytes. All allocator/pool bounds passed; unchanged packet,
logicalB512/KV. Auditlaunchers/wide_add_evidence.py passed. No formal throughput
recommendation/full-sizeCPUoracle/profile/repeats. No task jobs active except
intentionally stopped historicalCPU. [Report](evidence/original-wide-add-inference-20261002.md).

## Latest increment: resident capacity observations

Resident C++/Python consumers now export window_events_max/window_stages_max,
window_outputs_max and pending_peak from int64 device counters at existing
boundaries; one combined transfer after timing. Pending high-water history
continues with snapshots. Settle counts encoded boundary nodes,while projected
body diagnostics omit them. No graph/library/ABI/schedule/parameter change.
Development build-window-peaks-consumer-dev01 PASSED;window-peaks-npu-dev01
retained FAILED solely for six new Settle count assertions(160encoded vs120body).
Eleven other cases passed;corrected Settle six passedwindow-peaks-npu-dev02,
without library/numerical changes. Binary sources identical between dev snapshots.
Next: commit/push implementation,freeze window-peaks-clean01,build installed
consumer from qualifiedretained-journals-standalone and source-values-npu objects,
then run all34 affected sample-slicing/continuation/pool cases across FP32/FP16.
Next scale task planned: AddSGD11cards,B1,queue/arrivals512,outputs64,trace2048,
KV256/KVtrace8192,4GiBhead/context,60GiB/card. Offline conservative minimum≈51.132GiB.
Three bounded stages:D512/B8 then D2048/B2 then unchanged originalD2048/B512,
allT12/V50304/two connectedwindows. Stage inputs prepared underTASK/inputs/
wide-training-stage{512-b8,2048-b2}-add01. Not launched yet;qualification first.
No project jobs active except deliberately stopped historicalCPU.

## Next work

Prioritize full-size training memory
and actual comparisons. Offline4GiB-head dense plans still refuse minimum shapes:
Add8/12/16cards≈106/89/82GiB maximum;Attention≈182/148/132GiB(B4,trace16384,
KV256,pool8GiB,AdamW). These are conservative bounds,not physical impossibility.
Current gradients/parameter communication already use bounded owner streams.
Inspect retained parameter copies and per-card allocation lifetimes; derive and
validate bounds before relaxing admission. A per-card bounded retained tape pool
analogous to saved contexts may be useful, but needs explicit failure/transaction
semantics and independent training tests; not implemented or decided yet.
Automatic sample admission and mixed multi-device placement remain open. No new
performance matrix queued. Do not rerun completed representative timings.

## Remaining goal work

Original-wide other required flows,finite CPU/screened-mixed/resident comparisons;
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
Atomic handoff via durable_records.replace_text. Latest qualified resident libraries
retained-journals-{standalone,python}-clean01; consumer retained-journals-consumer-clean01.
Latest standalone CPU/NPU consumer binaries:source-values-{cpu,npu}-clean01.
Earlier context-pool consumer remains qualified for its original-wide evidence.
Core placement-{cpu,npu,npu-python}-clean01; eager clients sample-chunks-{cpu,npu}-clean01.
Standalone and Python-owned runtimes remain separate; no library mixing.

historical-cpu-attention-01 intentionally SIGSTOP,holding old timing.lock;
never resume/kill/clean it. Historical1.6438× was faster throughput,not current
online-flow evidence. Historical restricted work stays archived on pushed
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
