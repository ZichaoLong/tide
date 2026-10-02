# Current handoff

Updated 2026-10-02. **ACTIVE; continue autonomously.** Implementation, commits and
pushes authorized. No pause instruction; no subagents. Repository
/home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository, branch graph-execution-foundation.
Reference repositories and ObsidianVault remain read-only. Re-entry:
`git status --short --branch`; `python scripts/status.py`.
[execution-flows.md](execution-flows.md) is authoritative; [ROADMAP F1–F7](ROADMAP.md)
is the sole backlog. Overall goal incomplete.

## Contract

Candidates independently consume common inputs/parameters/initial state, never
reference events/routes/results/gradients. General online greedy covers legal
family topology/input including positive-delay PDG feedback. Preserve int64,
stable order, duplicate edges, missing/zero messages, None/zero gradients,
complete continuation and explicit differentiation boundaries. Performance:
PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Five presets plus fine switches; FP32 main, FP16
separate. Python resident uses native C++/CANN. No convergence requirement.
Implementation commit → immutable affected qualification → separate evidence
commit; push each. Contract outranks run-ml-experiments; reuse minimal records.
Do not rerun unchanged8,954 CPU checks or completed representative timings.
Own formal heavy timings are serial; no unbounded queue/scale/automatic retry.

## Completed scale evidence; do not repeat

All10 required representative family/client/schedule submatrices completed and
were audited. Last matrix-remaining02 PASSED/released at05:45:32UTC; no live
matrix recipe/hold remains. Last evidence270ee2e;frozen80dae6e.
[Settle/Python report](evidence/representative-settle-python-20261002.md) links
other completed matrices. Warm resident2.698–8.848× versus defaultPythonCPU is
not versus tunedLibTorchCPU, later memory-source timing, or original-wide timing.

Both original-wide FP32LibTorch resident TimedDAG/prefill inference runs passed:
480body/2208edges,D2048/B512/T12/V50304,two connectedwindows,128physicalB4groups.
- Attention17,521,117,376params:source48e44b0,jobwide-inference-staged01,
  construction817.996s,step325.278s,12,288outputs,1,184,430events,cut408,
  loss21.380956649780273,maxallocator14.562GiB,CPUpeak208.770GiB.
  [Report](evidence/original-wide-inference-20261002.md).
- Add9,468,053,696params:sourcebe380db,jobwide-add-inference02,
  construction182.038s,step278.574s,12,288outputs,1,183,429events,cut408,
  loss30.508380889892578,maxallocator7.812GiB,CPUpeak93.725GiB.
  Evidenceaab2ed2 [report](evidence/original-wide-add-inference-20261002.md).
These are cold execution/capacity evidence; no warm repeats, full-sizeCPUoracle,
profile or throughput recommendation. Neither logical batch nor KV was reduced.

## Latest qualified source and dependencies

- Per-device saved-context pool48e44b0/evidenceca559b5:all8jobs passed,CPU17/NPU64,
  independent C++ trajectories and0AiCPU trace. Context buffers may compact at
  explicit boundaries; caps apply across all live saved handles per card.
- Valid-prefix retained journals0fbc1b2/evidencefe8a08b:all8jobs passed,CPU17/NPU44,
  native64trajectories/1,024windows/256updates,CPUFP32/FP64,0AiCPUtrace.
  Representative128node AttentionD128/B8/T4/V257,fourB2slices,oneAdamW/twowindows:
  allocator1,868,793,344→1,455,752,704bytes(-22.10%),retained388,563,192→184,628,448.
  [Report](evidence/resident-retained-journals-20261002.md). Dense admission stays;
  nonzero extent sync only at retention boundaries, not per-event scheduling.
- Exact C++ named initializerbe380db/evidencec307fde:all5jobs passed,CPU25/NPU30,
  unchanged independent Python byte oracle;380scalar cases+actual consumers.
  3fresh CPUprocesses:6.25–6.51× for initializer only, not whole construction.
  [Report](evidence/exact-initializer-20261002.md).
- Window capacity counters475d4af:cleanbuild-window-peaks-consumer-clean01 and
  window-peaks-npu-clean01 PASSED;34checks,no skips,bothclients/precisions,
  sample slicing,whole/warmup,complete independentCPU updates and pool refusal.
  Auditlaunchers/window_peaks_evidence.py passed;
  [report](evidence/resident-window-peaks-20261002.md). Evidence9f305ab committed/pushed.
  New window_events_max/window_stages_max/window_outputs_max/pending_peak use
  int64 device counters,one transfer after timing. Pending history follows
  contexts;Settle encoded counts include boundary nodes omitted by body diagnostics.
  Core/library/ABI/kernels unchanged. Dev01 remains FAILED for6wrong newSettle
  assertions(160encoded vs120body);corrected6dev02 andall34clean passed.

## Original-width Add training and profiling

Reviewed [training/profile evidence](evidence/original-width-add-training-20261002.md),
source475d4af,original11card standaloneconsumerwindow-peaks-consumer-clean01.
Two parent stagedjobs remain FAILED:staged01 D512/B8child passed butoriginalD2048/B2
refused2TiBreversecapability;staged02 originalD2048/B2child PASSED after8TiB
hierarchicalcapability allowance withunchanged60GiB/card physicaladmission.
9,468,053,696params,FP32SGD,physicalB1,two connectedwindows:
construction249.211s,step20.2794s,48outputs,cut408,loss32.0892944,
maxallocator42.216GiB,pending384,maxevents1177.
PredeclaredB512cost gate5191.5s>3000s stoppedparent;no originalB512training.
RecordsTASK/runs/wide-add-training-staged0{1,2};originalhelpersunchanged.

Profile01 FAILED120s eleven-cardqueue;Profile02 PASSEDten cards07:31:52UTC.
Source475d4af,helperlaunchers/profile_original_width_v2.py;records
TASK/runs/original-width-profile02/profile/{result.json,consumer/result.json,raw/}.
OriginalD2048/B2 FP32SGD:construction201.214s,instrumentedstep21.8897s,
48outputs,loss32.0892944,maxallocator43.431GiB.701139operators/zeroobservedAiCPU;
includes construction/cleanup,not throughputrecommendation. Allleases released.
Canonicalstreampacket reservations total25.504GB(10cards)/31.007GB(11cards).
Next examine reuse across globallyorderedpairs/ordinals;source/receive arenas
must beseparate,preplan phase maxima,retain metadata/status and error/replay semantics.
Never lower admission constants or assert speedup withoutnewallocator evidence.

## Automatic sample admission qualified

Implementation3c3b4e7 committed/pushed;frozenTASK/sources/auto-samples-clean01.
CPU9/NPU18 checks PASSED,0skips;cleaninstalledconsumerbuildpassed. Audited
[evidence](evidence/resident-auto-samples-20261002.md),helperauto_samples_evidence.py.
Bothclients/precisions,inference/fullupdates plusfixedpath regressions,
16 independent complete trajectories;automatic training17→9sample rows.
New explicit resident-only --auto-sample-chunks triesoperatoradmission then
boundedhalving onMemoryRefusal,reaccountsalllogicalstate;firstfit orB1refusal.
Fixeddefault,no logicalbatch/KV/capacity/dtype/window/updatechanges.
Dev01 remainsFAILED8queuecapacity128 cases;testcapacitycorrected withsamecode/
tolerances;dev02all9passed. Allrecords retained;no activetestjob.
LatestNPUconsumerTASK/builds/auto-samples-consumer-clean01;core/deviceABI unchanged.

## Shared canonical packet arenas qualified

Implementationa72868f committed/pushed;frozenTASK/sources/shared-packets-clean01.
Alleightqualificationjobs PASSED;reviewed[evidence](evidence/resident-shared-packets-20261002.md).
Standalone/Python/clientbuilds,7componentcells,33consumerchecks,explicitFP32/FP16
2→3restore32trajectories,independentprofile. Per-phase unique send/receive arenas
preserve contributionorder anderrors;no devicekernel/publicABIchange.
D512/B8Attention two-card old/newallocator comparison:eachcard -268,437,504bytes;
packetreservation805,629,352→268,758,440bytes,allothercounters/loss identical.
New36group/5replaycheck peak4,720,640bytes. FP16trace53,274operators/zeroobservedAiCPU.
Retaincomponentdev01 failure(vendorfusion_result.json in sourcecwd;allchildrenpassed,
frozenfilesunchanged) andmemoryclean01capabilityrefusal. Successfulclean gates
useoutputcwd;memoryclean02usedsame60GiB/card physicalcap withlargerhierarchicalceilings.
No live shared-packet jobs;leasesreleased. Latestlibraries shared-packets-{standalone,python}-clean01,
client shared-packets-consumer-clean01. Original-width orB512memorybenefitnotyetmeasured.

## Current work: Add-specific gradient admission

Worktreecapacity.{h,py} removes fictitiousAttentionQKV/O gradientcharge fromAdd;
newattention_parameter_gradients subtotal includedinexisting physicalgradientcomponent.
ScalarAggregate,vectorLH/state/Read,projections,canonicalbanks andsafetymargins unchanged.
Actualmaterializedmodelinventorytest plus C++/Pythonplans pass10CPUchecks;
NPUdev9checksPASSED(two clients,FP32/FP16,fullupdates),buildpassed.
SnapshotTASK/sources/add-capacity-dev01;clientbuilds/add-capacity-consumer-dev01.
Commitimplementation,push,freezeadd-capacity-clean01;cleanCPU/NPUaffected qualification,
thenboundedoriginalD2048trainingvalidation. StaticoriginalB512 withphysicalB2:
10cardsqueue896/arrivals896/outputs64/trace3072,KV256/KVtrace8192,4GiBcontext,
60GiB/card admits52.0119GiBmax. Thisisnotexecutionproof. Earlier512queue/B1observed
pending384/maxwindowevents1177;largerbatchrequires explicitcapacities andchecks.
UsefiniteoriginalwidthB4 pilotbeforeB512;stoponcapacity/costbound,noOOMsearch.
HistoricalCPU remainsdeliberatelystopped.

## Next work and remaining goal

Reduce proven storage/lifecycle overheads,then continue bounded original training assessment. Continue actual
full-size complete training and finite CPU/screened-mixed/resident comparisons;
all required other family/client/schedule flows remain separate. Attention dense
training plan still refuses despite B1; investigate true allocation lifetimes,
retained immutable banks/gradient buffers rather than lowering safety constants.
Original inference context peaks≈1.53–1.58GiB/card at8cards, not a training proof.
Eager mixed multi-device parameter/payload placement remains open. F7 final migration/evidence audit;CUDA execution and other environment
tuples require target machines. HistoricalCPUAttention supplementary does not block.

Original packetsTASK/inputs/fullsize-{add,attention}01/workload.json are unchanged.
Region(budget) defaultsobserve_all=true:unselected nodes retainKV;clear only applies
to selected nodes. Do not reduce logicalB512 or silently dropKV to claim wide passed.

## Environment and preservation

TASK=/mi/data2T/zlong/tide-execution-flows. Public module
libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized /opt stack supersedes dated personalguide. Preserve module
PYTHONPATH,prepend frozen source/python. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0.16logical64GiBAscend910_9392;lease/remap only.
freeze_run.py:background.slice/Nice10,2buildworkers,boundedtasks. Lastfree:
data190GiB/root11GiB. Atomic handoff usesdurable_records.replace_text.
Formal timing lockTASK/online-measurement.lock.

Latest resident librariesretained-journals-{standalone,python}-clean01;core
placement-{cpu,npu,npu-python}-clean01. LatestNPUconsumershared-packets-consumer-clean01;
CPUsource-values-cpu-clean01. Earlier source-values-npu/context-pool consumers
stay qualified for their evidence. Standalone/Python runtimes remain separate.

historical-cpu-attention-01 is intentionallySIGSTOP,holding oldtiming.lock;
never resume/kill/clean it. Historical1.6438× meant fasterthroughput,not current
online-flow evidence. Restricted history remains on pushed
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
