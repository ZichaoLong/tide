# Current handoff

Updated 2026-10-04 (Asia/Shanghai). **ACTIVE**: user authorizes continuing under
[execution-flows](execution-flows.md) until the overall goal is complete,including
commits/pushes. No per-commit pause. No subagents. A later explicit pause overrides.
Repo `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`,branch
`graph-execution-foundation`. Re-entry:`git status --short --branch`,
`python scripts/status.py`,this file,and relevant [ROADMAP F1–F7](ROADMAP.md).
Reference repos and ObsidianVault remain read-only.

## Contract and acceptance

Independent CPU,mixed and NPU-resident online flows;streaming/general greedy
prefill,PDG positive-delay feedback,packed computation/transport,bounded capacity,
complete training and continuation. No CPU precomputed routes/events/gradients
feed candidates. Preserve int64/stable ordering,edge identities,missing/zero,
None/zero and declared VJPs. Five presets/fine switches stay configurable.
Performance:PDG LibTorch;TimedDAG/Settle LibTorch and Python;both schedules,
inference/training,CPU/screened mixed/resident. FP32 primary,FP16 separate.
Three fresh processes per formal recommendation;independent profiling.

Latest user decision **“接受，保留严格失败并单列限制，不阻塞”** accepts the
located original-scale numerical near-tie route failure as a separately listed
limitation,without blocking delivery. Exact comparisons,tie rules,fixtures and
dtype contracts stay unchanged. Never call that pair strictly equivalent;
disclose differing actual work counts in performance reports. Do not ask again.

Keep3000s/1.15 and all historical capacity/time refusals. Measurements may justify
a separately declared longer budget. Normally≤2 measured improvement rounds/~90min
active diagnosis per issue;no indefinite queues or blind retries. Formal heavy
timings serial;current overlapping jobs are feasibility/calibration only.
Implementation commit→affected immutable clean qualification→separate evidence
commit/push. Do not repeat unrelated passed gates. User contract outranks the
experiment skill;reuse minimal records,no new tracking infrastructure. Update
this handoff atomically with scripts.durable_records.replace_text.

## Qualified baseline

- All ten representative family/client/schedule submatrices complete. **Do not
  repeat these screens.** Their selected mixed presets feed the full-size plan.
- Full CPU integration on78e9df6:9458 passed,654 scoped optional skips;12/12
  CTests. [CPU integration](evidence/integrated-cpu-20261004.md). Original120-failure
  incomplete-build gate and test-assertion failures remain retained.
- Current core C++ hash
  `ca3597e96eb7a09dc68542b7b1c0904ec39f93375fe358524bf9b6273508868c`.
  Eager owners,packed transport,admission,FP16 masters and public resident
  sharded training/continuation are qualified on their cited fixed sources.
- e69b3bd corrects equivalent() integer/bool tensor comparisons to exact zero
  tolerance,including int64>2**53. CPU121/NPU49 clean gates passed;floating
  tolerance/algorithms unchanged. [Comparator](evidence/exact-discrete-comparison-20261004.md).
- c6ef224 extends6.25% CPU allocation allowance to learned+master+gradient/slots
  during training. CPU108 passed;first original-width recheck passed425.6376s
  within estimate. Second workers16 policy hit900s and remains failed;no further
  worker sweep. [RSS correction](evidence/cpu-training-rss-20261004.md).
- Fresh aarch64 CUDA-linked client passed187 CPU plus7 relocation checks,and
  updated eager FP16/master/RSS client passed108 CPU checks. Real NVIDIA/x86_64
  remains target-pending. [Target recipes](eager-target-validation.md).

## New integrated NPU qualification

Clean e69b3bd/compare-discrete-clean01. Old resident embedded d412541 core;
new backends use verified matching a785d43 core bytes and fresh install metadata.
This is core package assembly,not core recompilation. Resident C++/Ascend C and
installed combined eager+resident C++ consumer were freshly built.

| Job | Terminal result |
| --- | --- |
| build-integration-resident-standalone-clean02 | passed/exit0;current core+fresh resident,sharded-session check build |
| build-integration-resident-python-clean03 | passed/exit0;current Python core+fresh resident binding/shared library |
| build-integration-online-clean01 | passed/exit0;installed public-header-only combined consumer,verified loader |
| integration-eager-npu-clean01 | passed49/212.35s,no skips;two NPUs6,8,released |
| integration-resident-npu-clean01 | passed93/423.79s,no skips;three NPUs6,8,12,released |

All units inactive/empty cgroups;source/binary/installed/library hashes audited.
Resident gate covers all families/both schedules,FP32/FP16,SGD/AdamW,head splitting,
int64 boundaries,complete continued inference/training and fresh-process2→3-owner
checkpoint restore. [Integrated report](evidence/integrated-npu-consumers-20261004.md).
First standalone/Python clean01 failed before compile:incremental core lacked
cmake_install.cmake. Python clean02 failed before compile:bindings must be ON.
All three failures remain retained. Helpers build_integrated_resident.py,
build_integrated_resident_python.py,build_online_integrated.py and
 audit_integrated_resident.py under TASK/launchers contain exact commands/audits.

Current builds under TASK/builds:
`integration-core-standalone-clean02`,`integration-core-python-clean03`,
`integration-resident-standalone-clean02`,`integration-resident-python-clean03`,
`integration-online-clean01/consumer/tidegraph-online-bench`.
Python core artifacts are byte-verified copies of packed-transfer-npu-python-clean01.
CPU consumer `eager-rss-training-cpu-clean01` is c6ef224,source-matching current
consumer code. CUDA host consumer `eager-half-cuda-clean01` is c6ef224.

## Original-scale results and numerical limit

All below are cold feasibility/phase diagnostics,not formal recommendations.
Original packets:D2048/B512/T12/V50304,480 body nodes/2208 edges,
Add9,468,053,696 and Attention17,521,117,376 learned parameters.
Two connected windows/one complete SGD update unless stated otherwise.

| Model/flow | Cards;physical samples | Actual update seconds |
| --- | --- | ---: |
| Add CPU FP32 | CPU;32×16 | 1458.897225208 |
| Add mixed-A FP32 | 11;32×16 | 1287.284313840 |
| Attention mixed-A FP32 | 11;8×64 | 2655.242050484 |
| Add resident FP32 | 9;2×256 | 2170.549862239 |
| Attention resident FP32 | 11;1×512 | 5695.490452595 |
| Add mixed-A FP16 | 8;32×16 | 1254.205146587 |
| Attention mixed-A FP16 | 8;8×64 | 2334.246752751 |

Resident original FP32 inference also passed both models. Resident Attention
training used a separately measured9000s allowance;old3000s refusal remains.
FP16 source7b1fae5,eager-half-clean01,qualified eager-half-npu-clean01 consumer;
payload autograd gradients,FP32 loss/masters/slots,static scale128. Actual B512
job wide-eager-half-b512-01 passed/exit0,both updates and memory checks under3000s;
eight-card lease released. [FP16 B512 audit](evidence/original-b512-eager-fp16-20261004.md).
Prior FP16 pilots:Attention rows4 cost forecast3418s refused3000s;rows16 memory
63.887GiB refused;rows8 admitted51.312GiB with2488s forecast. These refusals remain.

FP32 eager Add CPU/mixed share1,183,427 events. Earlier resident Add has1,183,429;
FP16 eager Add/Attention have1,183,449/1,184,548 versus FP32 mixed Attention1,184,436.
Full-size strict parity is not established by feasibility success. Historical
route-witness02 diagnostic passed,but strict CPU/resident comparison failed:
sample17/window1/time280/region7,CPU246 beats245 by one FP32 ULP;resident rounds
both scores to the same FP32 value and correctly chooses lower-ID245. FP64 norms
of resident proposals still round to that same FP32 score. Window events2325/2327,
max pre-split proposal error7.703900337219238e-6. No altered tie policy or tolerance.
[Witness](evidence/original-add-route-witness-20261004.md);user accepts this listed
limitation without blocking other delivery. route-witness01 capacity-copy error
remains failed;do not conflate forensic completion with equivalence.

## Active jobs and exact next actions

`TASK=/mi/data2T/zlong/tide-execution-flows`;all units are
`tide-execution-flows-NAME` in background.slice,Nice10. Durable records:
`TASK/runs/NAME/{status.json,task.log}`,plus assessment/result.json. Frozen sources
TASK/sources,builds TASK/builds. Do not edit live helpers or frozen source.

1. **wide-eager-cpu-attention-b512-extended01 running** since2026-10-03T16:03:04Z.
   Sourcec6ef224/eager-rss-training-clean01;current CPU client,ATen16/workers1,
   original FP32 B512/physicalB32×16. Measured phase forecast×1.15=23391.025825s;
   separate27000s update,27500s child,27600s outer bound.429.436GiB estimate under
   512GiB cap. Original3000s refusal retained. Helper
   wide_eager_cpu_attention_b512_extended.py immutable. Audit actual terminal
   result/failure;no blind retry. It does not hold online-measurement.lock,so
   every formal runner must explicitly verify this job terminal/empty first.
2. **wide-mixed-continued-pilots01 running**,frozen e69b3bd/compare-discrete-clean01.
   Acquired11 NPUs1,2,3,4,5,6,7,8,9,11,12 (logical0..10). Eight original-width
   TimedDAG training pilots:LibTorch/Python × prefill/streaming × Add/Attention;
   mixed presets reused from representative screen. AddB64/physicalB32,
   AttentionB8/physicalB4,one continued warmup+one measured FP32 SGD step,two
   windows each. Purpose:continuous cost/peak calibration,not formal throughput.
   May overlap CPU feasibility. First case LibTorch/prefill/Add started;no result
   yet. Helpers wide_mixed_continued.py and fullsize_configs.py **immutable**.
   queue120s,child900s/case,outer7320s,first failure stops. Aggregate reservation
   430+128+64+8GiB passed dynamic half-memory. Preserve each case and failure.
   Inspect assessment/result.json and case/consumer/result.json. Forecasts do not
   substitute for actual originalB512 timing.

Inspect known jobs with `systemctl --user show tide-execution-flows-NAME` and
its task.log;stop only an identified current task if a real failure demands it.
No current task requests an unbounded NPU wait.

## Full-size timing plan and remaining acceptance

TASK/plans/fullsize-continuous-e69b3bd-01.json fixes120 FP32 cells:
ten family/client/schedule submatrices ×2 models ×2 modes ×3 flows.
All24 unique static geometry/client envelopes admit at11 NPUs or one CPU,
60GiB/card or512GiB CPU with unchanged margins. One continued warmup+one measured
SGD step,two windows each;every recommended configuration needs three independent
processes. FP16 separate. This plan is not execution evidence.

CPU Attention trainingB16,mixedB4,residentB1. Resident retains the qualified
explicit11-owner Attention map and unchanged queue/trace/KV capacities.
Resident AddB2,inferenceB4;CPU/mixed AddB32. Actual continuous peak/cost calibration
is still required. Task-local fullsize_formal_cell.py is an **unexecuted draft**:
requires hashed per-cell measured budgets,passed integrated gates,terminal old
cold jobs and exclusive online-measurement.lock. No formal job submitted yet.
Review the first admitted actual cell before multiplying cases.

Remaining:finish active CPU/continued mixed results;qualify necessary continuous
resident/CPU original-width memory/cost;execute full-size matrix,repeat formal
recommendations,separate full-size profiles/FP16 comparisons,then final support/
portability/evidence audit and terminal job closure. Current functionality and
integrated semantics do not close F6. No speedup threshold;no failure-only or
compile-only substitute for locally available complete runs. Existing strict
numerical failure is separately accepted,not a blocker to these actions.

## Environment and protected history

Authorized public module `libtorch-npu/2.10.0-cann9.0.0`,Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
Public stack authorization overrides stale private-stack guide paths;/usr/local
shared driver untouched. Preserve module PYTHONPATH and prepend source/python;
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0,PYTHONDONTWRITEBYTECODE=1.
Standalone NPU uses ACL_OP_INIT_MODE=0. NPU cwd must be output dir to avoid vendor
fusion_result.json in frozen source. CPU correctness one ATen/BLAS thread;build2.
Latest free disk:data167GiB/root11GiB;recheck before large writes.

**Never resume,stop,signal or clean historical-cpu-attention-01.** Deliberately
SIGSTOPped worker2686919 (~123.47GiB RSS) is protected. Its stale running receipt
is not active computation;its old timing.lock is distinct from current lock.
status.py exits1 for retained malformed build-reverse-gather-python-dev01 metadata;
do not rewrite it. finite_ranked_horizon.py and wide_attention_horizon_pilot.py
remain unqualified drafts;no48-row bound or changed original capacity is accepted.

Read-only navigation/schema audit checked834 local links with no missing targets;
portability schema is valid (10 targets,4 verified/6 implemented),not new hardware
qualification. This checkpoint records reviewed integration/FP16 evidence and current plans;
there is no uncommitted production code. Continue active jobs after commit/push,
without pausing.
