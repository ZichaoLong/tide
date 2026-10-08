# Current handoff

Updated 2026-10-08 (Asia/Shanghai). **FIRST-PROCESS-ONLY HANDOFF RUNNING; return to the user-directed wake-up workflow.**
The latest user explicitly requested unattended remaining experiments and a later
status/result review. Commit/push remain authorized. No subagents. Follow
[execution-flows](execution-flows.md);current alignment outranks experiment skill
overhead. Reference repositories and ObsidianVault stay read-only.

Repo `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`,branch
`graph-execution-foundation`. Re-entry:git status,python scripts/status.py,this
file,and [ROADMAP F1–F7](ROADMAP.md). `TASK=/mi/data2T/zlong/tide-execution-flows`.
All handoff writes use scripts.durable_records.replace_text atomically.

**Implementation and current local correctness gates are complete within the
declared CPU/NPU profiles(F1–F5). F6/F7 remain open for full-size measurements,
profiling reconciliation and final evidence review. Automatic repetitions are cancelled by the latest user alignment.** Actual NVIDIA/
x86_64 execution remains target-pending. Preserve the strict near-tie limitation.

The user authorized simplifying the remaining experiments on 2026-10-08:
retain first-process coverage; cancel all automatic FP32/FP16 repetitions;
accept shared-server observations without contention-triggered retries. Keep
warmup, measured work, strict correctness/acceptance, admission and finite limits.
At most four justified targeted follow-ups may be considered during review;
none are automatically scheduled. Selector/new-semantics work stays deferred.

Running `unattended-matrix03` replaces matrix02 after its current child drains.
Manifest: `TASK/plans/unattended-matrix03.json`; preflight:
`TASK/plans/unattended-matrix03-validation.json`. It retains the existing active
cell113 as a barrier and only31 new first-process jobs. All248 conditional repeat
slots are removed (up to160 FP32 and8 FP16 executions). Source/builds, child
launchers, warmup/measurement,120s device admission, single-heavy-job timing and
24GiB data/8GiB root reserves are unchanged. Whole allowance807265s is a protective
upper bound, not an ETA; first-process work was estimated at about160h before
this handoff. No failed or completed slot is retried.

The handoff has suspended only matrix02's identified Python coordinator(PID2262346); the
current Settle/Python/streaming/Add resident FP32 complete-training child keeps
running on its11 leased devices. Once its terminal receipt and empty cgroup are
verified, the handoff retires matrix02 and starts the successor. Parent
cancellation then means authorized queue replacement, not a failed measurement.
Original manifests, workload code, completed raw results and audits stay unchanged.
Two resource-free real-service guards passed natural drain and cancellation
cleanup. The single-process summarizer preserves every existing case metric;
repeat jobs/limits are rejected before any execution. No model/NPU gate was rerun.

New unit: `tide-execution-flows-unattended-matrix03.service`; cwd:
`TASK/sources/control-isolation-clean02`; workload e69b3bd,controller103f5b6.
Records: `TASK/runs/unattended-matrix03/{status.json,task.log,handoff.json,dispatch/result.json}`.
During drain, `handoff.json` is current; dispatch starts after the child finishes.
Verified2026-10-08: matrix03 MainPID2559079 is active/transient in background.slice;
handoff state `draining-current-child`, current child MainPID2262372 remains active.
The coordinator is stopped(T); the child is not stopped. No result is claimed yet.
Submission: `TASK/plans/unattended-matrix03-submission.json`.
[Single-process policy and handoff evidence](evidence/unattended-single-process-20261008.md).
[Queue operation and exact commands](unattended-measurements.md).

At the matrix01 disk stop:58/120accepted bound FP32 first processes(52from that queue
plus6earlier);30failed(28queue plus2earlier);32unstarted;no repeats yet.
The queue added11passes and1failure after the October6handoff. Nineteen failed
cells never obtained devices and did not execute the model. Other failures
remain explicit. FP32 Attention inference profiling passed;all four FP16
companions have first-process evidence. No formal recommendation is available.

Original-B512 FP16 Attention complete training passed on11cards,terminal exit0
at2026-10-04T10:52:49.692924Z:measured6190.644420s,warmup5804.552517s,
construction393.357138s,1.984931input tokens/s. Audit:
`TASK/runs/unattended-matrix01/dispatch/audits/formal-resident-fp16-attention-training01.json`.
This completes first-process evidence for all four separate FP16 companions;
automatic repetition is no longer required under the latest single-process policy. The protected historical
stopped worker is excluded and remains untouched.

Original-B512 FP16 Add complete training passed terminal audit on8cards:
measured2027.354117s,warmup2035.162127s,construction83.335077s,6.061102input tokens/s,
12288outputs/cut816/1187990events. It ended exit0 at06:49:59.658769Z,empty/released.
Both FP16 inference traces passed on all8cards;their five-second sampled intervals
showed no AiCPU tasks,without claiming that unsampled phases contain none.
[Training/profile evidence](evidence/formal-b512-fp16-training-profiles-20261004.md).
Earlier inference samples remain Add194.098070s and Attention222.541188s,
one process each in numa-bound-resident-fp16-cacheoff-v1-8devices. Their raw
failures,audits,cache policy and independent-series limitations remain unchanged.

## Contract and acceptance

Independent CPU,mixed and NPU-resident flows;general online greedy prefill and
streaming,PDG positive-delay feedback,packed operations,bounded capacities,
continued windows,complete backward/optimizer/checkpoint training. No CPU
precomputed routes/events/gradients feed candidates. Preserve int64,stable ties,
physical duplicate-edge identity,missing/zero and None/zero semantics and declared
VJPs. Five presets and fine switches remain configurable.

Performance scope:PDG LibTorch;TimedDAG/Settle LibTorch and Python;both schedules,
Add/Attention,inference/training,CPU/screened mixed/resident. FP32 primary,FP16
separate. One first-process attempt per full-size configuration; no automatic
repetitions. Recommendations must retain the uncertainty of single shared-server
observations; profiling and independent references are separate.
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
outputs12288/cut816,no diagnostics/profiler. **58/120 accepted bound first processes;
automatic repetitions are cancelled.** The eight historical table rows below are
TimedDAG/prefill/unbound serial;they are not pooled with the current bound series.

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
The initial numa-bound-solo-v1 cases (all Add inference/prefill):cell12 PDG/LibTorch CPU
260.290475s;14 resident329.307822s;48 Settle/LibTorch CPU214.649084s;50 resident
331.282152s;96 Settle/pure Python CPU834.559428s. Counts in these cases are1188500 forCPU and
1188494 forresident. No cross-series pooling. Cell24 PDG/CPU streaming adds570.014357s;
[streaming evidence](evidence/formal-b512-cpu-streaming-20261004.md) also preserves
cell36 completed-over-bound699.995797s outside accepted cases.

Different events prevent strict equal-work claims; single-process engineering choices retain shared-server uncertainty.
Historical unbound parents are terminal/empty with released leases.

Plan:TASK/plans/fullsize-continuous-e69b3bd-blas16-02.json,
SHA58e0bbab0a891b3645b3d64d35d788e405837ce5bbbb5a624e7a6a4b23aedc2e.
120 FP32 cells,24 envelopes,11 NPUs primary,60GiB/card or512GiB CPU. All120 budget02
files exist;94 transfer original-width measured envelopes with explicit limitations.
No transfer itself qualifies another family/client/schedule/full-size process.
62bound cells lack accepted first-process evidence:30failed(including the two
prior CPU streaming cases),one running,and31not started at the policy change.
Automatic recommendation repeats are cancelled. The frozen per-cell forecasts transfer
calibration assumptions and are not promises of actual remaining wall time.
Current counts and failures are in the linked queue/storage record above.

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

## Next wake-up

1. Read matrix03's unit, `TASK/runs/unattended-matrix03/status.json` and
   `handoff.json`. During `draining-current-child`, inspect cell113's own receipt;
   matrix02's coordinator is deliberately stopped and must not be resumed.
   After `successor-running`, inspect `dispatch/result.json` and phase summaries.
   Report passed, failed, skipped, unstarted and waiting separately.
2. Do not restart failed jobs, raise limits or duplicate live work. The new frozen
   plan has31 first-process jobs and zero repeats; the already-running child is
   audited as a barrier. All profiling and FP16 first processes already exist.
   A global stop retains its reason. Manager exit0 does not certify all cells.
3. Review single-process same-series timings, FP16 companions, actual events,
   shared-server uncertainty and trace limitations. Missing/failed competitors
   and small timing differences prevent a universal fastest-flow conclusion.
   Only a concrete unresolved decision/anomaly justifies a bounded targeted rerun.
4. Final F6/F7 reconciliation and support/target-machine handoff follow the results.
   No continuous agent polling or automatic callback is required.

## Environment and protected history

Authorized public module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
This authorization overrides stale private-guide paths;shared/usr/local driver
is untouched. Preserve module PYTHONPATH and prepend workload source/python.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0,PYTHONDONTWRITEBYTECODE=1,
standalone ACL_OP_INIT_MODE=0 and output cwd. CPU correctness1 thread,build2.
CPU timings OpenMP/OpenBLAS16,MKL1,OMP_THREAD_LIMIT32;NPU startup BLAS1,ATen8 resident.
Host320 physical cores,8 NUMA nodes,16 NPUs;NPU PCI NUMA=-1,actual locality unknown.
At continuation submission2026-10-08T12:10:08Z,free disk data134.76GiB/root13.22GiB;
recheck before large writes/profiles. Authorized2026-10-06maintenance freed
32.337883GiB by removing6416reviewed old profile/SQLite/object files and
hash-deduplicating5774raw pairs.32separately dated zero-byte markers remain.
40inactive root artifact/qualification directories(2.714954GiB)now live under
TASK/retired-root-artifacts-20261006,with atomic original-path symlinks and verified
content.644frozen queue inputs and109retained records still match. Dry runs,
logs,receipts and postcheck:TASK/plans/storage-cleanup-20261006-01.
[Maintenance details](evidence/storage-maintenance-20261006.md). The prior0.865349GiB
CMake-rule cleanup remains separately recorded under storage-cleanup-20261004-01.
Retired objects require recompilation;removed profiler SQLite databases require
re-export from preserved raw data when needed. Deduplicated raw paths are hardlinks:
keep them immutable and copy to a fresh output before any raw-writing operation.
Current/qualified builds,formal/failure records,protected history,SDKs,environments
and reference repositories remain. Maintenance services are terminal;the
measurement manager continues independently.

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

Latest core/controller implementation103f5b6;workload e69b3bd. This increment
implements the authorized single-process-only queue with a verified durable drain
and handoff. Graph/model code, measurement workloads and per-child limits are
unchanged; new task-local control/report helpers have frozen hashes. Automatic
repetitions are cancelled, shared-server uncertainty is explicit, and current
cell113 continues. Return to the wake-up workflow after commit/push.
Final F6/F7 evidence reconciliation remains pending.
