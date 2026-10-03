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
disclose differing actual work counts. Do not ask again. The sample17/window1/
time280/region7 witness has CPU246 ahead by one FP32 ULP,while resident245/246
round to a tie and correctly select245;proposal error7.7039e-6,events2325/2327.
This witness does not explain every future discrepancy.
[Strict failure retained](evidence/original-add-route-witness-20261004.md).

Keep3000s/1.15 and all historical capacity/time refusals. Measurements may justify
a separately declared longer budget. Normally≤2 measured improvement rounds/~90min
active diagnosis per issue;no indefinite queues or blind retries. Formal heavy
timings serial. Implementation commit→affected immutable clean qualification→
separate evidence commit/push. Do not repeat unrelated passed gates. User contract
outranks experiment-skill overhead;reuse minimal records. Update this handoff
atomically with scripts.durable_records.replace_text.

## Qualified baseline — do not repeat

- **All ten representative family/client/schedule submatrices complete.** Their
  selected mixed presets feed the full-size matrix. Full-size F6 remains open.
- Complete CPU integration on78e9df6:9458 passed,654 scoped optional skips;12/12
  CTests. [CPU integration](evidence/integrated-cpu-20261004.md).
- e69b3bd exact discrete comparator:CPU121/NPU49 passed;includes int64>2**53.
  Floating tolerances and algorithm semantics unchanged.
  [Comparator](evidence/exact-discrete-comparison-20261004.md).
- Current-core NPU integration on e69b3bd:49 eager and93 resident checks passed,
  no skips. All families/both schedules,FP32/FP16,SGD/AdamW,complete continuation
  and fresh-process2→3-owner restore. Fresh resident C++/Ascend C libraries and
  combined public-header consumer;core archives were byte-verified reuse plus
  new install metadata,**not core recompilation**. Three failed packaging attempts
  remain retained. [Integration](evidence/integrated-npu-consumers-20261004.md).
- CPU training allocation allowance on c6ef224:108 passed;original-width first
  RSS recheck passed425.6376s. Second workers16 policy timed out900s and stays
  failed. No further worker sweep. [RSS](evidence/cpu-training-rss-20261004.md).
- CUDA-linked aarch64 installed client:187 CPU checks and7 relocation checks;
  updated eager FP16/RSS client108 CPU checks. **Actual NVIDIA/x86_64 target
  execution remains pending.** [Recipes](eager-target-validation.md).

Current core C++ hash:
`ca3597e96eb7a09dc68542b7b1c0904ec39f93375fe358524bf9b6273508868c`.
TASK/builds:integration-core-{standalone-clean02,python-clean03},
integration-resident-{standalone-clean02,python-clean03},
integration-online-clean01/consumer/tidegraph-online-bench.
CPU consumer:eager-rss-training-cpu-clean01 (c6ef224,matching current consumer
source);CUDA host client:eager-half-cuda-clean01. Frozen source for current work:
TASK/sources/compare-discrete-clean01 at e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8.

## Completed scale and calibration evidence

Original packets retain D2048/B512/T12/V50304,480 body nodes/2208 edges,
Add9,468,053,696 and Attention17,521,117,376 parameters. Cold complete SGD
feasibility,two connected windows,**not formal continued throughput**:
CPU Add BLAS1 1458.897s;mixed-A11 FP32 Add1287.284s/Attention2655.242s;
resident9 Add2170.550s/resident11 Attention5695.490s;FP16 mixed-A8
Add1254.205s/Attention2334.247s. FP16 payload gradients,FP32 loss/masters/slots,
static scale128. Actual work differs;no equal-work or dtype speed ratio.
[FP16](evidence/original-b512-eager-fp16-20261004.md),
[FP32 mixed](evidence/original-b512-eager-mixed-20261004.md),
[resident Attention](evidence/original-b512-attention-training-20261003.md).

Completed original-width/reduced-batch continued pilots on frozen e69b3bd:

- [CPU BLAS16](evidence/original-width-cpu-blas16-20261004.md):four LibTorch
  model/mode cases passed;ATen16/workers1/startupBLAS16. The separate
  [BLAS diagnosis](evidence/cpu-blas-policy-20261004.md) found ATen16 had left
  OpenBLAS at1;its3.432× reduced-topology diagnostic is not a full-size ratio.
  No further worker sweep.
- [Accelerator calibration](evidence/original-width-accelerator-calibration-20261004.md):
  ten passed;four LibTorch resident cases,two LibTorch mixed inference and four
  Python mixed training cases. All owners/continuation/allocator checks passed.
- [Python calibration](evidence/original-width-python-calibration-20261004.md):
  **remaining-python-calibration01 passed10/10**,exit0 at2026-10-03T22:42:51.857555Z,
  empty cgroup and all leases released. Four pure-Python CPU cases,four
  Python-owned native resident cases,two mixed inference. Actual package/core/
  resident binary/placement/owner identities checked,including the distinct
  manifest schemas. Phase sums and1.15 forecasts independently recomputed.
  CPU/resident use two physical groups;mixed inference one. A single group
  does not certify B512 multi-group persistence. Its audit helper
  audit_python_calibration.py plus audit_flow_runtime.py are now exercised.
- The older [mixed continued batch](evidence/original-width-continued-mixed-20261004.md)
  retains four LibTorch passes and a PythonB64/physical32 timeout900s. Failed
  parent and unstarted cases remain history;new narrower pilots do not erase it.

**First actual formal B512 processes:cells0,1,2,repeat1 all passed/audited.**
LibTorch/TimedDAG/prefill/FP32 Add inference:

| Flow | Measured s | Input tokens/s | Actual candidate/resident events | Physical rows×groups |
| --- | ---: | ---: | ---: | --- |
| CPU | 226.376559 | 54.281238 | 1188500 | 32×16 |
| mixed-A,11 NPUs | 624.907705 | 19.663704 | 1188498 | 32×16 |
| resident,11 NPUs | 327.613456 | 37.507617 | 1188494 | 4×128 |

One continued warmup and measured step,two windows each;outputs12288,cut816.
No phase instrumentation,diagnostics or profiler. Bounds/memory/owner checks
passed. Resident and mixed took1.447× and2.760× the CPU time in these single
processes. **Not strict equal-work comparisons or three-process recommendations.**
[CPU audit](evidence/formal-b512-first-cpu-20261004.md),
[three-path audit](evidence/formal-b512-add-inference-npu-20261004.md).
formal-first-cpu-blas01 and formal-add-inference-npu01 are terminal/empty.
Evidence commit8ef3229 is pushed. Do not rewrite either audit JSON while the
following formal dispatcher reads their hashes.

[Full-size mixed profile failure/slice](evidence/fullsize-mixed-profile-slice-20261004.md):
profile-fullsize-mixed01 failed Add at4500s without a complete consumer result;
Attention unstarted. Parent empty,leases released. profile-slice-inspect01 passed:
441 copied files hash-verified,5.029450s/all11cards/137841tasks,mostly small vector
operations;no AiCPU observed in that slice. It neither explains the timeout nor
certifies full-update or resident placement. No automatic rerun or larger timeout.

## Active job and immutable inputs

`TASK=/mi/data2T/zlong/tide-execution-flows`;unit names
`tide-execution-flows-NAME`,background.slice,Nice10. Current own heavy work is
**formal-short-next01**,same frozen e69b3bd. The dependency wait ended after
Python calibration passed/empty;**cell8 resident Attention inference is running**,
then cell3 CPU Add training follows. Repeat1,strictly serial,fresh11-card lease
only for cell8;no lease during CPU work. One continued warmup and one measured
step,two connected windows,SGD for training;no instrumentation/profiling.

- Cell8:900s/update,2200s child. Cell3:3000s/update,6300s child.
- Formal group9100s;whole29620s including the already-ended≤20400s dependency
  wait. Queue≤120s. First failure stops;no automatic retry.
- Records:TASK/runs/formal-short-next01/{status.json,task.log,dispatch/result.json,
  assessment/result.json,assessment/cell-N-repeat-1/consumer/result.json}.
- Immutable while live:defer_formal_group.py,dispatch_formal_blas_group.py,
  fullsize_formal_cell_blas.py,fullsize_configs.py,plan02,budgets8/3 and both
  first-CPU/Add-comparison audit JSONs. Never start competing heavy work.
- Inspect:`systemctl --user show tide-execution-flows-formal-short-next01`.
  Cancellation,only if justified:`systemctl --user stop tide-execution-flows-formal-short-next01`.
- After terminal/empty,run the exercised schema-aware audit:
  `python TASK/launchers/audit_formal_blas_group_v2.py --name formal-short-next01 --output NEW_AUDIT_JSON`.
  Create its output parent first. Preserve partial failures and unstarted cells.

Old audit_formal_blas_group.py remains unchanged for the published Add report.
The v2 audit also rechecked that pair into
TASK/audits/formal-add-runtime-schema-v2.json. Its first output-directory
creation failure is retained under formal-add-runtime-schema-v2-attempt-01;
unchanged assertions passed after directory creation. It was an audit-output
error,not a consumer failure. Runtime-schema checks now include actual native
inference and training records,not only standalone or pure Python manifests.

## Next actions and full-size scope

Active plan:TASK/plans/fullsize-continuous-e69b3bd-blas16-02.json,
SHA58e0bbab0a891b3645b3d64d35d788e405837ce5bbbb5a624e7a6a4b23aedc2e.
120 FP32 cells,24 static configuration envelopes,11 NPUs for primary comparisons,
60GiB/card or512GiB CPU. Plan01 remains history. All geometry/capacity unchanged;
plan02 adds explicit CPU BLAS startup16. Static admission and reduced-batch
pilots do not qualify the120 full-size family/client/schedule/model/mode cells.

CPU Attention trainingphysical16,mixed4,resident1;CPU/mixed Add32,resident Add
training2/inference4. Resident Attention training uses the qualified explicit
11-owner map. No unqualified48-row limit. Formal cells0–2/repeat1 are complete;
117 other first processes and recommendation repeats remain open. Live cells8/3
are not passed merely because they started.

Budget02 files exist for cells0–11,72–83,88,94. Each hashes actual pilot basis.
The ten new Python budgets are now backed by the terminal-parent audit;none has
been executed. Original3000s/1.15 refusals stay separate from longer bounds:

| Cells | Separately declared step/child seconds |
| --- | --- |
| 5,77 resident Add training | 4500/9400 |
| 11,83 resident Attention training | 9000/18400 |
| 9 CPU LibTorch Attention training | 12000/24400 |
| 72 Python CPU Add inference | 1800/4000 |
| 75 Python CPU Add training | 3000/6400 |
| 78 Python CPU Attention inference | 4500/9400 |
| 81 Python CPU Attention training | 16500/33400 |
| 74,80 Python resident inference | 900/2200 |
| 73 Python mixed Add inference | 4500/9400 |
| 79 Python mixed Attention inference | 9000/18400 |
| 76,88 Python mixed Add training | 7500/15400 |
| 82,94 Python mixed Attention training | 16500/33400 |

These are operating limits,not B512 timings. In particular the Python CPU
Attention training forecasts10340.528306/12475.200981s imply hours of actual
measurement;prioritize shorter cells before the longest CPU Attention work.
Do not assume one family/schedule or a C++ run certifies a Python counterpart.

1. Commit/push the completed Python calibration evidence,then audit the terminal
   formal cells8/3,write their actual work/throughput/memory
   report and commit/push. Continue after commits;no approval/pause needed.
2. Run the remaining actual full-size matrix,using the measured budget basis and
   explicit transfer limits where a different family/schedule/preset is involved.
   Three fresh processes before formal recommendations. Keep strict failures
   and differing actual work visible;do not infer all discrepancies from one witness.
3. Separate full-size resident/FP16 profiling and comparisons remain open;do not
   repeat the completed ten representative submatrices or unrelated passed gates.
4. Finish support/portability/evidence audit and terminal task-job closure. NVIDIA/
   x86_64 execution stays target-pending. Keep the protected historical task below
   untouched. No production code changed during this continuation.

## Environment and protected history

Authorized public module `libtorch-npu/2.10.0-cann9.0.0`,Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
Public stack authorization overrides stale private-guide paths;shared/usr/local
driver untouched. Preserve module PYTHONPATH,prepend source/python;
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0,PYTHONDONTWRITEBYTECODE=1.
Standalone NPU uses ACL_OP_INIT_MODE=0 and output cwd. CPU correctness1 thread,
build2. Launcher defaultBLAS1;CPU BLAS helpers explicitly override child env16.
Latest free disk:data165GiB/root11GiB;recheck before large writes.

**Never resume,stop,signal or clean historical-cpu-attention-01.** Protected
worker2686919 (~123.47GiB RSS) remains deliberately stopped;its stale running
receipt is not active computation. Its old timing.lock differs from current lock.

wide-eager-cpu-attention-b512-extended01 was separately cancelled/exit143 after
the measured BLAS finding;worker2801147 gone,cgroup empty,no completedB512 result.
Its cancellation.json retains rationale/hashes,old23391s forecast,27000s budget
and3000s refusal. It is neither a passed timing nor a failed mathematical update.
Waiting-only formal-mixed-prefill-01,formal-mixed-prefill-02 and
remaining-continuous-calibration01 were cancelled before measurement or leases;
all cancelled/exit143/empty. **Do not resume those superseded dispatchers.**
status.py exits1 for retained malformed build-reverse-gather-python-dev01;do not
rewrite history. finite_ranked_horizon.py and wide_attention_horizon_pilot.py
remain unqualified drafts. Prior navigation/schema audit passed834 links and
10-target support schema;not new hardware verification.

Evidence commits:1d32008 CPU pilots,f6e49f0 retained profile failure/slice,
d416e3c ten accelerator pilots,b45c2b8 first formal CPU,8ef3229 first three-path
formal comparison;all pushed. This checkpoint adds the audited Python calibration
(JSON/Markdown),its ROADMAP link and this consolidated handoff. No unrelated
uncommitted work. Task-local helpers/budgets remain outside the repository;
active versions above are immutable. Continue after commits.
