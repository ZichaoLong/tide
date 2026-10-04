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

**Only current heavy job:formal-bound-cpu-streaming01**,running.
The previous six-cell group ended failed only because cell98 could not lease11
NPUs within its queue allowance. Five completed consumers passed terminal audit.
Its four dependent successor services stopped before starting any consumer.
The historical stopped worker is separate and protected below.

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

## Current CPU-only group:formal-bound-cpu-streaming01

Submitted and verified transient/background.slice/Nice10,MainPID473708 at launch,
from controller103f5b6c8e9b2e6e46a7f2733185433de2cb5f3f
at TASK/sources/control-isolation-clean02. Workload remains frozen
**e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8** at TASK/sources/compare-discrete-clean01.
Five fresh repeat1 streaming Add CPU cells24,36,60,84,108 cover all required
family/client submatrices. Series numa-bound-solo-v1;no earlier process is repeated.

- Same80-core NUMA0–3 lane,static CPU estimate+8GiB RSS allowance,reserve136GiB;
  admission/placement/time/RSS monitor is unchanged. No competing heavy work.
- Existing child1600s for LibTorch and4600s for Python;group limits1840,1840,1840,
  4840,4840s and whole15320s. First failure stops;no retries or larger budgets.
- Shell:TASK/launchers/formal-bound-cpu-streaming01.sh;static/runtime preparation:
  TASK/plans/formal-bound-cpu-streaming01-inspection01.json.
- Records:TASK/runs/formal-bound-cpu-streaming01/{status.json,task.log,assessment/result.json};
  assessment/group-N/{result.json,monitor/result.json,cell-I-repeat-1/case.json}.
- Inspect:systemctl --user show tide-execution-flows-formal-bound-cpu-streaming01.
  Stop only if justified:systemctl --user stop that exact unit.
- Terminal auditor:audit_bound_matrix_v2.py --name formal-bound-cpu-streaming01
  --output NEW_AUDIT_JSON. Parent directory must exist;no overwrite.
- Do not edit active dispatch_bound_matrix.py,isolated_flow_case.py,fullsize_configs.py,
  audit_flow_runtime.py,plan02,budgets,inspection,shell,frozen sources or hashed builds.

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
outputs12288/cut816,no diagnostics/profiler. **13/120 audited first processes;
0 cells have three-process evidence.** The first eight below are TimedDAG/prefill/
unbound serial;the five newly audited bound cases are a separate series.

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
1188494 forresident. No cross-series pooling;cell98 is still unmeasured.

Different events and single processes prohibit strict equal-work/recommendation claims.
Historical unbound parents are terminal/empty with released leases.

Plan:TASK/plans/fullsize-continuous-e69b3bd-blas16-02.json,
SHA58e0bbab0a891b3645b3d64d35d788e405837ce5bbbb5a624e7a6a4b23aedc2e.
120 FP32 cells,24 envelopes,11 NPUs primary,60GiB/card or512GiB CPU. All120 budget02
files exist;94 transfer original-width measured envelopes with explicit limitations.
No transfer itself qualifies another family/client/schedule/full-size process.
107 first processes plus recommendation repeats remain. Remaining nominal phase
forecasts sum≈204.2h before construction/repeats;budget forecasts with1.15 included
sum234.8h (CPU77.5h/NPU157.3h). These are unchanged transferred forecasts,not
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
configuration. It aggregates13 cases in audits/formal-series-review02;review01
reproduced the eight old cases. Five synthetic guard checks passed in
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

1. Monitor/audit the active CPU-only group. The five newly accepted bound cases
   and dependency failures are audited in
   [the current report](evidence/formal-b512-bound-add-prefill-20261004.md).
   Series coverage is TASK/audits/formal-series-review02. Continue independent CPU work while11 NPUs are unavailable;do not
   leave CPU cases behind an unavailable-NPU dependency chain again.
2. At changed NPU availability,continue unstarted cell98 and other untouched FP32
   cells using the existing11-owner maps/capacities/finite budgets. Preserve
   failed receipt names and allocate fresh directories. No heavy measurement overlap.
3. FP16 companion plan resident-fp16-companions01.json covers four LibTorch/
   TimedDAG/prefill resident cases:2/8 inference and5/11 complete training. No
   companion has actually run. Step/child budgets900/2200,900/2200,4500/9400,
   9000/18400s stay;transferred FP32 allowances are not FP16 forecasts. Reuse
   run_resident_fp16_companion.py and audit_resident_fp16_companion.py (latter
   syntax-checked,not yet exercised on a terminal consumer). No broad second matrix.
4. The FP32 Attention profile helper profile_formal_resident.py --cell8 is ready;
   its dependent service failed before starting. Delay220s,duration5s,execution1800s,
   profile/export3360s,queue120s,11cards,24GiB free-disk prerequisite. Add profile
   is already audited;do not repeat it. Use audit_resident_profile.py at terminal.
5. New profile_resident_fp16.py / audit_resident_fp16_profile.py are syntax-checked
   only,not submitted. The runner requires --memory add|attention --reference-audit
   AUDIT --out NEW;--inspect derives a finite trace interval/budget from a passed
   unprofiled FP16 companion. Preserve original packet/owners/chunks/dtype.
6. Complete same-series fresh repetitions before recommending configurations;
   finish support/portability/evidence audit and close all current goal jobs.
   Profiling,FP16 and qualification screens remain separate. Actual NVIDIA/x86_64
   execution stays target-pending under the user's cross-machine validation plan.

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

Latest implementation103f5b6;previous handoff checkpoint8fe7e74. This checkpoint
records the terminal partial matrix and independent CPU continuation. No
graph/model/core change or uncommitted implementation. This increment contains
reviewed documentation/evidence only. Task-local helpers are
retained by hashes. A goal is active for autonomous completion;check get_goal on
re-entry. Continue after commits/pushes.
