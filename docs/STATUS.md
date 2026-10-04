# Current handoff

Updated 2026-10-04 (Asia/Shanghai). **ACTIVE.** The user explicitly resumed work
and authorized completing the overall goal,including the proposed bounded
CPU/NPU resource-isolation qualification. Continue autonomous implementation,
measurement,audit,commit and push;no automatic per-commit pause. A later user
pause overrides. Follow [execution-flows](execution-flows.md). No subagents.
qualify-flow-isolation01 is terminal/failed and audited:the NPU consumer completed,
but its monitor rejected remaining group processes at immediate exit. No overlap
started. Correct/retest lifecycle handling;no concurrent formal timings are qualified.
Next independent heavy job is profile-formal-resident-add01 on frozen e69b3bd.
No competing heavy work during this profile.
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
timings remain serial until the newly authorized resource-isolated overlap
is implemented and qualified. No silent mixing of bound/concurrent and old
unbound/serial measurement series. Implementation commit→
affected immutable clean qualification→
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
Evidence commit8ef3229 is pushed. Preserve the audit JSONs and their hashes as
recorded inputs to the completed dispatchers.

[Full-size mixed profile failure/slice](evidence/fullsize-mixed-profile-slice-20261004.md):
profile-fullsize-mixed01 failed Add at4500s without a complete consumer result;
Attention unstarted. Parent empty,leases released. profile-slice-inspect01 passed:
441 copied files hash-verified,5.029450s/all11cards/137841tasks,mostly small vector
operations;no AiCPU observed in that slice. It neither explains the timeout nor
certifies full-update or resident placement. No automatic rerun or larger timeout.

## Current work and active-job boundary

**TERMINAL FAILED** `qualify-flow-isolation01` / unit
`tide-execution-flows-qualify-flow-isolation01`, exit1, MainPID0, empty cgroup,
all leases released. CPU solo passed;NPU consumer passed but its monitor failed
`child exited with remaining group processes: npu`;overlap never started.
[Failure audit](evidence/measurement-isolation-20261004.md),raw audit
TASK/audits/qualify-flow-isolation01.json. Preserve all raw records and frozen
controller a3e7270/workload e69b3bd. No parallel timing has been approved.

Lifecycle fix implemented:reap exited adopted grandchildren before classifying
a leak;allow≤2s natural teardown inside the original lane bound,record remaining
states,still fail/clean live leaks and nonzero adopted exits. Directed17 checks
passed in8.050s. The original controller reproduced the exited-zombie failure
using a real fork,retained at TASK/audits/measurement-zombie-reproducer-a3e7270.
This does not identify the exact old NPU helper. Commit the tested fix,create a
frozen controller,rerun its directed checks,then qualify after the profile ends.
No graph/model change,process/step/queue/whole budget increase or blind retry.

**RUNNING** `profile-formal-resident-add01`,unit
`tide-execution-flows-profile-formal-resident-add01`,source e69b3bd at
TASK/sources/compare-discrete-clean01. One cold Add inference update/two windows,
11NPUs,unchanged full B512 physical4/128groups;five-second msprof slice only.
Verified transient/background.slice/Nice10,MainPID84252 at launch,started
2026-10-04T01:56:52.733403Z;lease physical1–9,11,12 remapped logically0–10.
Execution1200s,export≤120s/session,whole2760s,outer2940s,queue120s;24GiB free
storage prerequisite. It cannot certify whole-update profiling or formal speed.

- Launch shell:TASK/launchers/profile-formal-resident-add01.sh.
- Records:TASK/runs/profile-formal-resident-add01/{status.json,task.log,queue.json,
  assessment/result.json,assessment/consumer/result.json,assessment/raw/}.
- Inspect:`systemctl --user show tide-execution-flows-profile-formal-resident-add01`.
  Stop only if justified:`systemctl --user stop tide-execution-flows-profile-formal-resident-add01`.
- Terminal audit prepared:TASK/launchers/audit_resident_profile.py --name
  profile-formal-resident-add01 --output NEW_AUDIT_JSON. Not yet exercised.
- Frozen while active:profile_formal_resident.py,source/build/input dependencies,
  and its shell. Do not launch the old serial matrix scripts beside it.

Bounded remaining-matrix dispatcher is prepared externally as
TASK/launchers/dispatch_bound_matrix.py;syntax checked,not yet qualified. It is
solo by default and permits only exact audited screen pairs for overlap,with
matching source/configuration/binding/environment/cards. It keeps new bound
series separate from historical unbound results. A read-only preparation while
the failed qualification was still live refused the local NPU-node memory
estimate;no matrix process started. Receipt:plans/formal-bound-next01-inspection-attempt01.json.
Update its controller revision after the lifecycle fix;then prepare a new finite
first-process group for PDG/Settle rather than repeating unrelated passed gates.

## Closed current group and evidence

`TASK=/mi/data2T/zlong/tide-execution-flows`;unit names
`tide-execution-flows-NAME`,background.slice,Nice10. **The previous formal group is closed.** The new isolation group above is the
only authorized current heavy stage. The historical stopped process is separate.
`formal-python-short01` passed all3 cells/exit0 at2026-10-04T00:20:12.114767Z;
MainPID0,inactive,empty cgroup,both NPU leases completed/released. Its dependency
wait for formal-short-next01 ended before measurement. Same frozen e69b3bd.
[Terminal-parent audit](evidence/formal-b512-python-inference-20261004.md):

| Cell | Python-owned inference flow | Measured s | Input tokens/s | Actual events | Physical rows×groups |
| --- | --- | ---: | ---: | ---: | --- |
| 72 | pure Python CPU Add | 625.406563 | 19.648019 | 1188500 | 32×16 |
| 74 | native resident Add,11 NPUs | 342.512730 | 35.876039 | 1188494 | 4×128 |
| 80 | native resident Attention,11 NPUs | 421.433023 | 29.157658 | 1190499 | 4×128 |

Each:repeat1,originalB512,one continued warmup and one measured step,two windows
per step,outputs12288/cut816. Source/package/native binding/resident binary,
placement/owner/capacity/work/timeout checks passed. CPU RSS growth72.635326GiB;
max NPU allocator growth7.282186/13.528637GiB. No strict equivalence or binding
cost claim follows from matching resident event counts. Own timings stayed serial.
CPU live environment also confirmed OpenMP/OpenBLAS16,MKL1,thread limit32,
autoload0;affinity remained all320CPUs/all8memory nodes.

- Bounds unchanged:74/80 900s/update,2200s child;72 1800s/update,4000s child.
  Group9240s;whole15960s including≤6600s dependency wait;NPU queue≤120s.
- Records:TASK/runs/formal-python-short01/{status.json,task.log,dispatch/result.json,
  assessment/result.json,assessment/cell-N-repeat-1/consumer/result.json}.
- Preserve source,launchers,plan02,budgets74/80/72 and hashed calibration inputs.
  Re-audit into a new output whose parent exists:
  `python TASK/launchers/audit_formal_blas_group_v2.py --name formal-python-short01 --output NEW_AUDIT_JSON`.
- TASK/audits/formal-matrix-review03 summarizes **8/120** actual first processes,
  cells0,1,2,3,8,72,74,80;**0** cells have three fresh processes. Earlier reviews
  remain historical. No representative submatrix or passed gate was rerun.

formal-short-next01 is now passed/exit0 at2026-10-03T23:31:02.569252Z,empty,lease
released,**both cells8/3 audited**. [Actual measurements](evidence/formal-b512-cpu-training-resident-inference-20261004.md):
CPU Add training970.259169s,warmup893.228409s,12.664657input tokens/s,
1188205candidate events,32×16groups,128.458973GiB RSS growth;resident11 Attention
inference402.400685s,warmup401.514133s,30.536727input tokens/s,1190499events,
4×128groups,13.528270GiB max allocator growth. Both originalB512,two windows,
outputs12288/cut816. Different models/modes;not a CPU/NPU ratio. One process each.

Old formal audit remains unchanged for the first Add report. V2 also rechecked
that pair into TASK/audits/formal-add-runtime-schema-v2.json. Its first output-dir
creation failure is retained under formal-add-runtime-schema-v2-attempt-01;
unchanged assertions passed after directory creation. It was an audit-output
error,not a consumer failure. V2 checks actual Python/native runtime identities.

## Next actions and full-size scope

Active plan:TASK/plans/fullsize-continuous-e69b3bd-blas16-02.json,
SHA58e0bbab0a891b3645b3d64d35d788e405837ce5bbbb5a624e7a6a4b23aedc2e.
120 FP32 cells,24 static configuration envelopes,11 NPUs for primary comparisons,
60GiB/card or512GiB CPU. Plan01 remains history. All geometry/capacity unchanged;
plan02 adds explicit CPU BLAS startup16. Static admission and reduced-batch
pilots do not qualify the120 full-size family/client/schedule/model/mode cells.

CPU Attention trainingphysical16,mixed4,resident1;CPU/mixed Add32,resident Add
training2/inference4. Resident Attention training uses the qualified explicit
11-owner map. No unqualified48-row limit. Formal cells0,1,2,3,8,72,74,80/repeat1
are audited;
112 other first processes and recommendation repeats remain open. Static plans
and reduced-batch pilots never substitute for these actual full-size processes.

Budget02 files now exist for all120 cells. The prior26 files are unchanged;
94 new finite allowances transfer matching original-width model/client/mode
pilot budgets using max(1,representative target/source median ratio),rounding
up to300s. This does not transfer full-size qualification. Different source,
thread/shape/family/schedule/preset limits are explicit;no lower allowance from
a faster representative result. `prepare_transferred_budgets.py` and
TASK/plans/transferred-budget02-receipt.json preserve their derivation. First
failure stops each submitted group;no automatic retry or larger bounds.
Each budget hashes its measured basis;all old3000s/1.15 refusals remain.
The ten Python budgets have the terminal-parent calibration audit basis;
74/80/72 now have audited formal results. Remaining cells are unexecuted.
Original3000s/1.15 refusals stay separate from longer bounds:

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

The120 initial-process nominal phase estimates sum to207.2h before construction
and repeats (budget phase_forecasts_seconds divided by their1.15 guard). After
the eight audited cells,the remaining sum is205.2h:CPU68.0h/NPU137.2h. This is a
rough planning sum with cross-family transfer uncertainty,not measured remaining
time. It makes the matrix a multi-day workload under the serial timing contract.
Budgets are operating limits;forecasts are not B512 timings. The Python CPU
Attention training forecasts10340.528306/12475.200981s indicate potentially
hours-long measurements;prioritize shorter cells before the longest CPU work.
Do not assume one family/schedule or a C++ run certifies a Python counterpart.

**User has authorized resumption and overall completion.** Do not launch old
prepared serial scripts beside the new isolation qualification. Next implement
and qualify bounded aggregate resource admission;then advance the matrix.

1. Implement and qualify [parallel resource review](evidence/parallel-resource-review-20261004.md).
   Suggested first overlap:one CPU lane plus one11-NPU lane,disjoint CPU masks/
   memory nodes,combined resource reservation and a finite solo/overlap control
   under identical binding. Installed numactl/taskset and320cores/8NUMA/2.01TiB
   make this plausible,but NPU PCI NUMA=-1 leaves locality unverified. Two11-card
   cases cannot fit16cards. Current global lock/admission is serial-only;do not
   bypass it without implementing and qualifying the aggregate policy. Existing
   conservative estimates permit some pairs,not CPU Attention training+resident
   under the observed half-memory budget. No concurrent speedup is measured.
2. Complete remaining actual full-size FP32 cells and recommendation repeats,
   using declared finite bounds and actual work disclosure. Do not repeat the
   ten completed representative submatrices or unrelated correctness gates.
   TASK/launchers/formal-attention-inference-next01.sh (cells6/7) is prepared,
   bash-n valid,group13300s/outer13360s,**never submitted**. It complements
   measured resident cell8. Strict failures remain separate;one near-tie witness
   does not explain every future difference. No further CPU worker sweep.
3. Separate full-size resident/FP16 profiling and comparisons remain open.
   TASK/launchers/profile_formal_resident.py is inspection-only:passed cell2/8
   binary/config/owners,one cold inference step/two windows,five-second trace,
   24GiB free disk prerequisite. Add delay120s/execution1200s/whole2760s;
   Attention delay220s/execution1800s/whole3360s. Inspection records are
   TASK/plans/profile-formal-resident-{add,attention}01-inspection.json.
   TASK/launchers/profile-formal-resident-add01.sh passes bash-n,outer2940s,
   queue120s,11NPUs,**never submitted**. No profile audit helper qualified yet.
   No automatic retry or increase of the retained mixed-profile timeout.
4. TASK/launchers/summarize_formal_matrix.py has been exercised through review03.
   It checks unique cell/repeat identity and preserves unmeasured/failed cells.
   Three-process evidence is necessary,not automatically sufficient,for a
   recommendation. Binding/overlap qualification must be a separately recorded
   series rather than silently merged with these unbound serial measurements.
5. Finish support/portability/evidence audit and final job closure after the
   remaining matrix. NVIDIA/x86_64 execution stays target-pending. Historical
   stopped work below remains untouched. No production code changed this round.

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

Recent pushed evidence:af1289b Python calibration,f52bbec public numerical
limitation,2485d39 CPU training/resident Attention inference. This checkpoint
adds the three audited Python formal cells,resource-isolation inspection/proposal,
ROADMAP coverage and this paused handoff. All changes are documentation/evidence;
there were no production changes or unrelated edits at517c48f,pushed. Task-local
helpers/budgets remain retained by the evidence. The user has now resumed work.
This implementation checkpoint includes the active handoff,measurement-control
modules,13 directed checks and the authorized isolation contract. Full-size
isolation qualification is being launched as above. Continue after commits/pushes.
