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

**Only current heavy job:formal-bound-next01**,running six new solo FP32 B512
Add inference cells. The corrected CPU/NPU isolation screen completed its
execution/cleanup checks,but rejected concurrent performance qualification:
CPU measured time+20.2%,NPU+1.1%. Keep formal heavy measurements solo. No competing
heavy work. The deliberately stopped historical worker below is separate.

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

## Active group:formal-bound-next01

Unit `tide-execution-flows-formal-bound-next01`,verified transient/background.slice/
Nice10,MainPID158470 at launch,started2026-10-04T02:23:31.244486Z. Its finite
isolation dependency wait has ended and dispatch/isolation-audit.json passed.
The slowdown screen was rejected,so this group correctly stayed **solo**.

Controller103f5b6c8e9b2e6e46a7f2733185433de2cb5f3f at
TASK/sources/control-isolation-clean02;workload remains frozen
**e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8** at TASK/sources/compare-discrete-clean01.
Six fresh repeat1 cells12,14,48,50,96,98:PDG/LibTorch,Settle/LibTorch,Settle/Python;
CPU and resident11 per submatrix. Cell12 CPU PDG is running;later cells are not
accepted yet. New series `numa-bound-solo-v1`,separate from prior unbound results.

- CPU lane80 cores across NUMA0–3;NPU-host lane78 across4–7;two controller slots.
  Sample every live thread mask/private anonymous NUMA pages every5s,RSS/time every1s.
  CPU RSS cap=static estimate+8GiB;NPU resident280GiB,mixed64GiB;shared reserve136GiB.
  Admission is repeated per actual group;external load/shared pages remain uncontrolled.
- Original budget02 step/child limits unchanged. Group bounds1840,3040,1840,
  3040,4840,3040s include queue/controller allowance. NPU queue120s per cell;
  group whole17760s,outer24420s including≤6600s dependency wait. First failure stops.
- Records:TASK/runs/formal-bound-next01/{status.json,task.log,dispatch/result.json,
  dispatch/isolation-audit.json,assessment/result.json}. assessment/group-N holds
  monitor/result.json and cell-I-repeat-1/{case.json,consumer/result.json}.
- Inspect:`systemctl --user show tide-execution-flows-formal-bound-next01`.
  Stop only if justified:`systemctl --user stop tide-execution-flows-formal-bound-next01`.
- Launch:TASK/launchers/formal-bound-next01.sh. **Do not edit live inputs:**
  after_isolation_bound_matrix.py,dispatch_bound_matrix.py,audit_bound_matrix.py,
  audit_flow_isolation_v2.py,isolated_flow_case.py,fullsize_configs.py,audit_flow_runtime.py,
  plans/formal-bound-next01-inspection02.json,plan02,budgets,controller/workload source,
  existing builds and hashed dependencies.
- Terminal audit:TASK/launchers/audit_bound_matrix.py --name formal-bound-next01
  --output NEW_AUDIT_JSON (parent must exist;no overwrite). It is prepared,not yet
  exercised on this actual group. Never call a running parent passed.
- Six negative dispatcher checks passed:audits/formal-bound-dispatch-rejections01.
  An earlier read-only preparation refused NPU-node memory while the old screen
  was live;plans/formal-bound-next01-inspection-attempt01.json preserves it.
  No workload started during those preparations.

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
outputs12288/cut816,no diagnostics/profiler. **8/120 audited first processes;
0 cells have three-process evidence.** All eight are TimedDAG/prefill/unbound serial.

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
Different events and single processes prohibit strict equal-work/recommendation claims.
Historical unbound parents are terminal/empty with released leases.

Plan:TASK/plans/fullsize-continuous-e69b3bd-blas16-02.json,
SHA58e0bbab0a891b3645b3d64d35d788e405837ce5bbbb5a624e7a6a4b23aedc2e.
120 FP32 cells,24 envelopes,11 NPUs primary,60GiB/card or512GiB CPU. All120 budget02
files exist;94 transfer original-width measured envelopes with explicit limitations.
No transfer itself qualifies another family/client/schedule/full-size process.
112 first processes (including six in the live group) plus recommendation repeats
remain. Remaining nominal phase forecasts sum≈205.2h before construction/repeats,
CPU68.0h/NPU137.2h;cross-family transfer makes this uncertain,not promised completion.

CPU/mixed Add physical32,resident Add inference4/training2. Attention training:
CPU16,mixed4,resident1 with qualified explicit11-owner map. No unqualified48-row policy.
Some independently declared bounds:resident Add training4500/9400s(step/child),
resident Attention9000/18400;CPU LibTorch Attention12000/24400;Python CPU
Attention16500/33400. Other transferred bounds are in each file;largest child35800s.
Original3000s refusals remain separate. No automatic retry/increase on a failure.

Use TASK/launchers/summarize_flow_series.py --audit AUDIT ... --output-dir NEW
for upcoming bound results. It preserves unbound/solo-bound/overlap-bound series,
rejects duplicate consumer artifacts and dtype mixing,and does not recommend a
configuration. It reproduced all eight old cases in audits/formal-series-review01
and passed five synthetic guard checks in formal-series-guards01.json. Its first
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

1. Monitor/audit the live six-cell group. After terminal audit,commit/push its
   actual results and update separate series coverage;continue remaining FP32
   cells with existing finite budgets. Prefer uncovered/shorter cases before the
   longest Python CPU Attention training. Do not launch old serial scripts beside it.
2. Complete required fresh repetitions before recommending configurations.
   Bound/unbound and overlapping/solo conditions stay distinct. Do not treat the
   rejected screen as authorization for concurrent formal timings.
3. Resident Attention FP32 partial profile is prepared via profile_formal_resident.py
   --cell8:delay220s,duration5s,execution1800s,whole3360s,queue120s,11NPUs,24GiB
   free-data-disk prerequisite. Never launched. Existing Add profile is complete;
   do not rerun it. A profile is not formal timing or full-training coverage.
4. FP16 preparation:plans/resident-fp16-companions01.json statically admits four
   LibTorch/TimedDAG/prefill resident companions,Add/Attention inference/training,
   same original packets/11-owner maps/physical rows/capacities. Peak estimates
   20.616/39.769/48.812/52.822GiB/card. Step/child allowances900/2200,900/2200,
   4500/9400,9000/18400s,transferred as operating limits from FP32,not FP16 forecasts.
   run_resident_fp16_companion.py is syntax-checked,**not launched**. Actual
   execution,terminal audit and separate FP16 profile remain. This is a limited
   companion set,not a second Cartesian matrix or a qualification claim.
5. Finish support/portability/evidence audit and close all current goal jobs once
   actual remaining work is complete. NVIDIA/x86_64 device execution remains
   explicitly target-pending under the user's cross-machine validation plan.

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

Latest pushed implementation103f5b6 and evidence2920f24. This checkpoint records
the rejected isolation screen and ongoing matrix dispatch;no graph/model/core
change. Continue after commits/pushes. Uncommitted work is documentation/evidence
until the next stated implementation change;task-local helpers are retained by hashes.
