# Current handoff

Updated 2026-10-04 (Asia/Shanghai). **ACTIVE.** The user explicitly authorized
resuming and completing the overall goal. Continue implementation,measurement,
audit,commit and push;no automatic per-commit pause. No subagents. Follow
[execution-flows](execution-flows.md);current user alignment outranks experiment
skill overhead. Reference repositories and ObsidianVault stay read-only.

Repo `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`,branch
`graph-execution-foundation`. Re-entry:git status,python scripts/status.py,this
file,and [ROADMAP F1–F7](ROADMAP.md). `TASK=/mi/data2T/zlong/tide-execution-flows`.
All handoff writes use scripts.durable_records.replace_text atomically.

**Only current heavy job:formal-resident-fp16-add-training01**,running.
Verified transient/background.slice/Nice10,MainPID717698 at launch;physical cards
1,3,4,5,7,9,11,13 map to logical0–7. Started2026-10-04T05:40:31.375630Z.
Its input hashes and clean workload e69b3bd/controller103f5b6 were rechecked.
Eight cards,originalB512,physical sample rows2,FP16 payload/FP32 adjoints/loss/masters,
explicit ACLNN_CACHE_LIMIT=0 and the passed93-check gate. Bounds remain4500s step,
9400s child,120s queue,9640s whole. No competing heavy measurement.
Launcher/plan:TASK/{launchers,plans}/formal-resident-fp16-add-training01.{sh,json};
records:TASK/runs/formal-resident-fp16-add-training01. Terminal auditor:
audit_resident_fp16_companion_v4.py --name JOB --output NEW_AUDIT_JSON.

Both originalB512 FP16 inference companions passed v4 terminal audit:
Add inference03 measured194.098069530s,warmup194.477604910s,construction60.926761025s;
Attention inference02 measured222.541188209s,warmup222.400112901s,
construction133.807347373s. Each used8cards/physical sample rows4,with12288 outputs
and cut816;actual events1188301(Add)/1190764(Attention). Both exit0/empty/released.
[Add evidence](evidence/formal-b512-resident-fp16-add-20261004.md) and
[Attention evidence](evidence/formal-b512-resident-fp16-attention-20261004.md) retain
their single-process observations,separate from120 FP32 cells;no recommendation.
Add's first v3 audit rejected an incorrect inference optimizer expectation;retain
that attempt and original helper. Separate v4 requiresNone for inference and the
configured optimizer for training,with all other checks unchanged.
Audits:TASK/audits/formal-resident-fp16-{add-inference03,attention-inference02}.json.
Protect v3 execution helpers,gate audit,plan02 and frozen sources/builds. The
original default-cache failures remain failed. The historical stopped worker is
separate and protected below.

## Contract and acceptance

Independent CPU,mixed and NPU-resident flows;general online greedy prefill and
streaming,PDG positive-delay feedback,packed operations,bounded capacities,
continued windows,complete backward/optimizer/checkpoint training. No CPU
precomputed routes/events/gradients feed candidates. Preserve int64,stable ties,
physical duplicate-edge identity,missing/zero and None/zero semantics and declared
VJPs. Five presets and fine switches remain configurable.

Performance scope:PDG LibTorch;TimedDAG/Settle LibTorch and Python;both schedules,
Add/Attention,inference/training,CPU/screened mixed/resident. FP32 primary,FP16
separate. At least three fresh processes within one measurement series before a
formal recommendation;profiling and independent references are separate.
Python-owned native is distinct from pure Python and standalone LibTorch.
Training quality/convergence/recipes belong to later experiments,not this base.

User accepted **“接受，保留严格失败并单列限制，不阻塞”**. Keep the located original-scale
near-tie strict failure and checks unchanged. Its sample17/window1/time280/region7
witness has CPU246 ahead by one FP32 ULP;resident245/246 tie and correctly choose245.
Proposal error7.7039e-6,events2325/2327. Do not call the pair strictly equivalent,
change fixtures/tie rules/tolerances,or assume this witness explains every future
discrepancy. Report different actual work counts.
[Retained witness](evidence/original-add-route-witness-20261004.md).

Keep all original3000s/1.15 refusals. Separately justified finite longer bounds
are allowed;no automatic retry,increased timeout or indefinite queue. Normally
≤2 measured improvement rounds/~90min active diagnosis per issue. No further CPU
worker sweep. Implementation commit→affected clean immutable gate→separate evidence
commit/push. Do not rerun unrelated passed gates. No new tracking infrastructure.

## CPU streaming and FP16 copy diagnosis

formal-bound-cpu-streaming01 ended exit1 at2026-10-04T04:38:31.304689Z,empty cgroup.
Terminal audit TASK/audits/formal-bound-cpu-streaming01.json is audited-failed:
cell24 PDG/LibTorch/CPU Add streaming passed,570.014357140s;cell36 TimedDAG
completed its consumer and monitor but measured699.995796604s against600s step
allowance(warmup529.082825666s). Keep the failed allowance unchanged;do not count
cell36 as a formally accepted case. Cells60/84/108 never started. No automatic
retry or larger budget;continue independent work. Auditor audit_bound_matrix_v2.py
accepts one case and records four partial/unstarted groups.

Failed formal-resident-fp16-add-inference02 used the frozen controller103f5b6
and workloade69b3bd,original B512 Add,8devices,FP16 payload/FP32 adjoints/loss/masters,
physical sample rows4,one warmup and one measured step,two windows each.
The same900s step/2200s child allowances remain;queue120s,whole2440s. No heavy overlap.
- Shell:TASK/launchers/formal-resident-fp16-add-inference02.sh.
- Exact command/hashes:TASK/plans/formal-resident-fp16-add-inference02.json.
- Started2026-10-04T04:43:39.917700Z;queue leased physical1,3,4,5,7,9,11,13
  as logical0–7. Unit:tide-execution-flows-formal-resident-fp16-add-inference02;
  records:TASK/runs/formal-resident-fp16-add-inference02/{status.json,task.log,queue.json,assessment/result.json}.
- Terminal audit:audit_resident_fp16_companion_v2.py --name
  formal-resident-fp16-add-inference02 --output NEW_AUDIT_JSON.
- Terminal audit TASK/audits/formal-resident-fp16-add-inference02.json is audited-failed;
  exit1,empty cgroup,lease failed/released. Monitor88.009s;consumer failed2.
  Preserve v2 helpers,plan02,referenced source/builds and raw records. No full step result.

FP16 plan resident-fp16-companions02.json statically admits8cards for Add inference
23.0107GiB/card,Attention inference45.6709GiB/card,Add training51.9472GiB/card.
Attention training retains11cards(52.8223GiB/card);8card admission was refused.
LogicalB512,requested physical sample rows4/4/2/1,capacities and time bounds remain.
Generic placement changes owner maps/card counts;no isolated dtype speedup claim.
TASK/plans/fp16-companions-v2-prepare02/result.json verifies four command/memory
preparations and rejects wrong dtype/sample rows/device count,without executing
or leasing devices. First preparation failed solely on the inspection's wrong
'npu' spelling expectation(actual normalized'npu:0');retain prepare01.
FP16 profile helpers now read8/11 device count and require the audited reference
cache policy;syntax checked only. They require a successful unprofiled companion before submission.

## Terminal bound group and dependency failures

formal-bound-next01 ended exit1 at the final queue stage. Cells12,14,48,50,96
passed all consumer,monitor,source/runtime/placement/memory and terminal checks.
Cell98 **never started**:queue-5 timed_out with insufficient_free_devices,9 free
reported against11 required. Configured wait120s;actual final observation137.089s
reflects queue polling. No model timeout or device execution failure occurred.
TASK/audits/formal-bound-next01.json is **audited-failed with five accepted cases**,
not a passed parent. The original auditor omitted timed_out from terminal queue
states;its rejected attempt is retained in formal-bound-next01-attempt01.json.
Separate audit_bound_matrix_v2.py handles an unstarted timeout without relaxing
any completed-consumer checks. Original auditor and raw run records are unchanged.

Four services ended exit1 with AssertionError('failed') on their dependency,
all empty cgroups,without an assessment/queue/consumer being created:
formal-resident-fp16-add-inference01,formal-resident-fp16-attention-inference01,
formal-bound-streaming-add01,profile-formal-resident-attention01. Preserve every
receipt/dispatch record/plan/helper. Do not resume these old services. The first
streaming inspection's live-memory refusal and later passed inspection also stay.
Future NPU work requires changed availability and a new finite submission;do not
increase queue waits or repeat already accepted CPU/NPU processes mechanically.

## Qualified implementation — reuse these gates/builds

All **10/10 representative family/client/schedule submatrices are complete**.
Full-size F6 remains open. Do not repeat representative screens or broad pilots.

- CPU integration78e9df6:9458 passed,654 scoped optional skips,12/12 CTests.
  [CPU gate](evidence/integrated-cpu-20261004.md).
- Exact discrete comparator e69b3bd:CPU121/NPU49,including int64>2**53.
  [Comparator](evidence/exact-discrete-comparison-20261004.md).
- Current-core NPU integration e69b3bd:49 eager+93 resident,no skips;all families,
  schedules,FP32/FP16,SGD/AdamW,continuation and fresh-process2→3-owner restore.
  Core archives were byte-verified reuse;resident libraries and combined installed
  consumer freshly built. Retain the three failed packaging attempts.
  [NPU gate](evidence/integrated-npu-consumers-20261004.md).
- CPU training RSS allowance c6ef224:108 passed;first width recheck425.6376s passed.
  Second worker16 policy timed out900s and remains failed;no worker sweep.
  [RSS](evidence/cpu-training-rss-20261004.md).
- CUDA-linked aarch64 installed client:187 CPU+7 relocation;updated eager FP16/RSS
  client108 CPU checks. **Actual NVIDIA/x86_64 execution remains target-pending.**
  [Target commands](eager-target-validation.md).

Core C++ hash ca3597e96eb7a09dc68542b7b1c0904ec39f93375fe358524bf9b6273508868c.
TASK/builds:integration-core-{standalone-clean02,python-clean03},
integration-resident-{standalone-clean02,python-clean03},
integration-online-clean01/consumer/tidegraph-online-bench.
CPU consumer:eager-rss-training-cpu-clean01 (matching c6ef224 consumer source).
CUDA host client:eager-half-cuda-clean01. Keep these hashed inputs unchanged.

## Audited full-size measurements and remaining coverage

Packets:D2048/B512/T12/V50304,480 body nodes/2208 edges;Add9,468,053,696 parameters,
Attention17,521,117,376. Logical input tokens12288 per step. All table entries are
original B512,one continued warmup and one measured step,two windows each,
outputs12288/cut816,no diagnostics/profiler. **14/120 audited first processes;
0 cells have three-process evidence.** The first eight below are TimedDAG/prefill/
unbound serial;six accepted bound cases are a separate series.

| Cell | Client/model/mode/flow | Measured seconds | Actual candidate/resident events |
| --- | --- | ---: | ---: |
| 0 | LibTorch Add inference CPU | 226.376559 | 1188500 |
| 1 | LibTorch Add inference mixed-A11 | 624.907705 | 1188498 |
| 2 | LibTorch Add inference resident11 | 327.613456 | 1188494 |
| 3 | LibTorch Add complete SGD CPU | 970.259169 | 1188205 |
| 8 | LibTorch Attention inference resident11 | 402.400685 | 1190499 |
| 72 | Pure Python Add inference CPU | 625.406563 | 1188500 |
| 74 | Python-owned native Add inference resident11 | 342.512730 | 1188494 |
| 80 | Python-owned native Attention inference resident11 | 421.433023 | 1190499 |

[First CPU](evidence/formal-b512-first-cpu-20261004.md),
[Add NPU flows](evidence/formal-b512-add-inference-npu-20261004.md),
[CPU training/Attention resident](evidence/formal-b512-cpu-training-resident-inference-20261004.md),
[Python inference](evidence/formal-b512-python-inference-20261004.md).
New numa-bound-solo-v1 cases (all Add inference/prefill):cell12 PDG/LibTorch CPU
260.290475s;14 resident329.307822s;48 Settle/LibTorch CPU214.649084s;50 resident
331.282152s;96 Settle/pure Python CPU834.559428s. Counts remain1188500 forCPU and
1188494 forresident. No cross-series pooling;cell98 is still unmeasured. Cell24 PDG/CPU streaming adds570.014357s;
[streaming evidence](evidence/formal-b512-cpu-streaming-20261004.md) also preserves
cell36 completed-over-bound699.995797s outside accepted cases.

Different events and single processes prohibit strict equal-work/recommendation claims.
Historical unbound parents are terminal/empty with released leases.

Plan:TASK/plans/fullsize-continuous-e69b3bd-blas16-02.json,
SHA58e0bbab0a891b3645b3d64d35d788e405837ce5bbbb5a624e7a6a4b23aedc2e.
120 FP32 cells,24 envelopes,11 NPUs primary,60GiB/card or512GiB CPU. All120 budget02
files exist;94 transfer original-width measured envelopes with explicit limitations.
No transfer itself qualifies another family/client/schedule/full-size process.
106 accepted first processes plus recommendation repeats remain;cell36 has a completed consumer retained as a bound failure. Remaining nominal phase
forecasts sum≈204.0h before construction/repeats;budget forecasts with1.15 included
sum234.6h (CPU77.4h/NPU157.3h). These are unchanged transferred forecasts,not
actual remaining wall time or a promised completion date.

CPU/mixed Add physical32,resident Add inference4/training2. Attention training:
CPU16,mixed4,resident1 with qualified explicit11-owner map. No unqualified48-row policy.
Some independently declared bounds:resident Add training4500/9400s(step/child),
resident Attention9000/18400;CPU LibTorch Attention12000/24400;Python CPU
Attention16500/33400. Other transferred bounds are in each file;largest child35800s.
Original3000s refusals remain separate. No automatic retry/increase on a failure.

Use TASK/launchers/summarize_flow_series.py --audit AUDIT ... --output-dir NEW
for upcoming bound results. It preserves unbound/solo-bound/overlap-bound series,
rejects duplicate consumer artifacts and dtype mixing,and does not recommend a
configuration. Current audits/formal-series-review03 aggregates14 accepted cases;review02
retains13 and review01 the eight old cases. Five synthetic guard checks passed in
formal-series-guards01.json. Its first
wrong-cwd preparation failure is retained;no measurement was affected. The older
formal-matrix-review03/summarize_formal_matrix.py cover only the unbound series.
Do not mix bound measurements or qualification screens into old repeat groups.

## Completed scale/calibration/profile/control evidence

Cold complete-SGD feasibility (two connected windows,not continued formal throughput):
CPU Add BLAS1 1458.897s;mixed-A11 FP32 Add1287.284s/Attention2655.242s;
resident9 Add2170.550s/resident11 Attention5695.490s;FP16 mixed-A8
Add1254.205s/Attention2334.247s. [FP16 evidence](evidence/original-b512-eager-fp16-20261004.md)
uses FP16 payload gradients,FP32 loss/masters/slots,static scale128. Counts differ;
no cross-card/dtype speed ratio. Cold feasibility never substitutes for formal continuation.

All calibration envelopes are established:CPU BLAS16 four model/mode cases;
accelerator ten;remaining Python ten. These original-width/reduced-batch gates
validated persistence/owners/memory and yielded finite allowances. No more broad
pilots. Earlier PythonB64/physical32 timeout900s remains failed. CPU ATen16 alone
had left OpenBLAS1;formal CPU explicitly starts OpenMP/OpenBLAS16,MKL1,limit32.
[CPU calibration](evidence/original-width-cpu-blas16-20261004.md),
[Accelerators](evidence/original-width-accelerator-calibration-20261004.md),
[Python](evidence/original-width-python-calibration-20261004.md).

- qualify-flow-isolation01 failed after successful NPU consumer exit:old monitor
  rejected remaining descendants before reaping. Cleanup/leases empty;overlap
  unstarted. [Retained failure](evidence/measurement-isolation-20261004.md).
- Fix103f5b6 is pushed;17 real-process/control checks passed on its immutable
  checkout (8.048s). Reap exited descendants/check their statuses;≤2s natural
  teardown stays inside original lane timeout;live/nonzero helpers still fail.
  Old-controller synthetic fork reproducer is retained;it does not identify
  that old NPU helper. [Control gate](evidence/measurement-lifecycle-20261004.md).
- qualify-flow-isolation02 passed all four consumers/three monitors and terminal
  audit,exit0 at02:42:16.833543Z,empty cgroup,leases released. Each NPU process
  reaped two exit0 descendants with zero teardown delay. Same-backend outputs/
  losses/counters matched. CPU194.385205solo→233.628268overlap;NPU325.315243→329.014670.
  **Screen rejected** at CPU1.201883/NPU1.011372 against≤1.05 each. No concurrent
  recommendation or other class qualification. [Audit](evidence/measurement-isolation-result-20261004.md).
  TASK/audits/qualify-flow-isolation02.json;keep all sources/plans/raw records.
- profile-formal-resident-add01 passed/audited,exit0 at02:04:27.114218Z,empty/released.
  Original B512 resident FP32 Add cold inference,two windows,same11 owners/chunks.
  Requested5s trace:actual5.72370525s span,168322 tasks/all11cards,444 hashed files,
  no observed AiCPU. Vector87.18% of summed task durations,not wall-time share.
  Instrumented340.721409s excluded from formal matrix.
  [Profile](evidence/fullsize-resident-add-profile-20261004.md).
  TASK/audits/profile-formal-resident-add01.json;auditor audit_resident_profile.py exercised.
- profile-fullsize-mixed01 remains failed4500s,no complete consumer result;
  Attention unstarted. Valid independently audited5.029450s/all11cards/137841-task
  slice,mostly small vector ops,no observed AiCPU. It does not identify timeout
  cause or certify resident/full-update behavior. No blind retry.
  [Retained profile](evidence/fullsize-mixed-profile-slice-20261004.md).

## Next independent work

1. Monitor/audit live Add training01 with v4;both inference companions
   are audited-passed. Uses plan02,v3 helpers,ACLNN_CACHE_LIMIT=0 and unchanged9640s
   whole budget. Attention training remains11cards. Inspect terminal evidence
   between heavy jobs;do not chain unrelated work to a passing prerequisite.
2. Continue untouched FP32 cells with existing11-owner maps/capacities and finite
   allowances when devices are available. CPU60/84/108 are still unstarted.
   Keep cell36's completed-but-over-bound result separate;do not retry blindly.
3. FP32 Attention profile profile_formal_resident.py --cell8 remains unstarted:
   delay220s,duration5s,execution1800s,profile/export3360s,queue120s,11cards,
   minimum24GiB free disk. Use a fresh numbered name and audit_resident_profile_v2.py.
   Add FP32 profile is already audited;do not repeat it.
4. FP16 profile_resident_fp16.py/audit_resident_fp16_profile.py accept the actual
   audited8/11-card inference configuration and require matching ACLNN_CACHE_LIMIT. --inspect derives a finite trace
   interval/budget from --reference-name COMPLETED_JOB --reference-audit AUDIT;
   retain packet,owners,chunks,dtype and exclude profiling from formal timings.
5. Complete same-series fresh repetitions before recommending configurations;
   finish support/portability/evidence audit and close all current goal jobs.
   Actual NVIDIA/x86_64 execution stays target-pending under the user's plan.

## Environment and protected history

Authorized public module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
This authorization overrides stale private-guide paths;shared/usr/local driver
is untouched. Preserve module PYTHONPATH and prepend workload source/python.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0,PYTHONDONTWRITEBYTECODE=1,
standalone ACL_OP_INIT_MODE=0 and output cwd. CPU correctness1 thread,build2.
CPU timings OpenMP/OpenBLAS16,MKL1,OMP_THREAD_LIMIT32;NPU startup BLAS1,ATen8 resident.
Host320 physical cores,8 NUMA nodes,16 NPUs;NPU PCI NUMA=-1,actual locality unknown.
Latest free disk data165GiB/root11GiB;recheck before large writes/profiles.

**Never resume,stop,signal or clean historical-cpu-attention-01.** Worker2686919
(~123.47GiB RSS) is deliberately stopped;its running receipt is not active
computation. Its timing lock is distinct. Other superseded dispatchers stay
cancelled. wide-eager-cpu-attention-b512-extended01 was separately cancelled143
following the BLAS finding;worker2801147 gone,empty cgroup,no completed B512 result.
Preserve old23391s forecast/27000s bound/3000s refusal and cancellation rationale.

scripts/status.py exits1 for retained malformed build-reverse-gather-python-dev01;
do not rewrite that history. Prior navigation/schema audit passed834 links and
10-target schema;this is not new hardware verification. Retain active/cited
artifacts and reproducers. Do not clean reference repositories.

Latest implementation103f5b6;workload e69b3bd. This increment records two audited
original-size FP16 inference companions and the active complete-training job.
No graph/model/core change or uncommitted implementation;reviewed evidence only. Task-local helpers are
retained by hashes. A goal is active for autonomous completion;check get_goal on
re-entry. Continue after commits/pushes.
