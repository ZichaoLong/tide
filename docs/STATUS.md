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

## Completed scale/calibration evidence

Original packets:D2048/B512/T12/V50304,480 body nodes/2208 edges,
Add9,468,053,696 and Attention17,521,117,376 parameters. Existing cold complete
SGD updates,two connected windows,**not formal continued throughput**:
CPU Add BLAS1 1458.897s;mixed-A11 FP32 Add1287.284s/Attention2655.242s;
resident9 Add2170.550s/resident11 Attention5695.490s;FP16 mixed-A8
Add1254.205s/Attention2334.247s. FP16 payload gradients,FP32 loss/masters/slots,
static scale128. Actual work differs;no equal-work or dtype speed ratio.
[FP16](evidence/original-b512-eager-fp16-20261004.md),
[FP32 mixed](evidence/original-b512-eager-mixed-20261004.md),
[resident Attention](evidence/original-b512-attention-training-20261003.md).

[Mixed continued calibration](evidence/original-width-continued-mixed-20261004.md):
wide-mixed-continued-pilots01 failed/exit1 after four LibTorch passes and one
Python AddB64 timeout900s;three Python cases never started. All devices released.
AddB64/physical32 and AttentionB8/physical4,one continued warmup+one measured
step,two windows each. Current narrower Python follow-up is listed below.

[CPU BLAS diagnosis](evidence/cpu-blas-policy-20261004.md):ATen16 left
OpenBLAS0.3.30 USE_OPENMP at1. Explicit startupBLAS16 improved a reduced-topology
complete-training diagnostic3.432× across three processes/policy;reported losses/
work counters match. This is not a full-size ratio. Both probe jobs passed/empty.

[CPU original-width BLAS16 pilots](evidence/original-width-cpu-blas16-20261004.md):
cpu-blas-continued01 passed/exit0,empty cgroup. Original model width/parameters,
reduced batch,two physical chunks,phase instrumentation,overlapped profiling.
ATen16/workers1,OMP16/OPENBLAS16,MKL1,OMP_THREAD_LIMIT32.

| Cell | CPU FP32 case | Batch/physical | Warmup/measured s | B512 forecast warmup/measured s,including1.15 |
| --- | --- | --- | --- | --- |
| 3 | Add training | 64/32 | 105.561763/115.645236 | 939.407377/1039.361385 |
| 9 | Attention training | 32/16 | 536.410864/535.885729 | 9544.565449/9692.312292 |
| 0 | Add inference | 64/32 | 26.999169/28.504013 | 248.392358/262.236922 |
| 6 | Attention inference | 64/32 | 163.179488/210.936170 | 1501.251293/1940.612768 |

[Full-size profile failure and valid slice](evidence/fullsize-mixed-profile-slice-20261004.md):
profile-fullsize-mixed01 failed/exit1 at2026-10-03T19:43:32Z;Add timeout4500s,
no complete consumer result,Attention unstarted. Unit/cgroup empty,11 cards
released. profile-slice-inspect01 passed/exit0:441 hash-verified copied files;
5.029450s/all11cards/137841tasks,mostly small vector ops,zero AiCPU observed in
this slice. It does not prove full-update placement,timeout cause or resident
behavior. Never call the parent passed or infer timing from its snapshot.

## Active jobs and immutable inputs

`TASK=/mi/data2T/zlong/tide-execution-flows`;units
`tide-execution-flows-NAME` in background.slice,Nice10. Records
TASK/runs/NAME/{status.json,task.log,assessment/result.json}. Inspect with
`systemctl --user show tide-execution-flows-NAME`;no unbounded NPU waits.

1. **remaining-accelerator-calibration02 passed/exit0**,all ten cases,terminal
   2026-10-03T20:45:55.803368Z,empty cgroup and released11-card leases. Audited:
   [original-width accelerator calibration](evidence/original-width-accelerator-calibration-20261004.md).
   Raw cell-N/result.json and queue-cell-N.json retained. Audit helper
   audit_accelerator_calibration02.py initially used the wrong owner field for
   resident records;failed audit-only copy retained in audit-attempt-01. Corrected
   audit strictly checks resident Full/state owners and eager node owners;no
   runtime record or production code changed. Old PythonB64 timeout stays failed.

2. **formal-first-cpu-blas01 passed/exit0**,empty cgroup. Actual originalB512
   CPU/Add inference,LibTorch/TimedDAG/prefill,BLAS16/ATen16/workers1,
   physical32×16,continued warmup207.571919s,measured226.376559s,
   construction46.319800s,54.281238input tokens/s. Outputs12288,cut816,
   candidate events1188500,body candidates1163924;RSS42.767GiB inside estimate.
   [First formal audit](evidence/formal-b512-first-cpu-20261004.md) passed exact
   source/binary/packet/budget/owner/phase/cgroup checks. One process only;
   no recommendation or CPU/NPU ratio. Its actual result has now been reviewed.
3. Submit **formal-add-inference-npu01**,same frozen e69b3bd:exact cells1,2,
   repeat1,mixed-A then resident on11NPUs,serial,fresh lease per cell,queue120s.
   Helper dispatch_formal_blas_group.py requires the first audited result.
   OriginalB512,continued warmup1+measured1,two windows,no phase instrumentation.
   Budgets:cell1=1200s/update+2800s child;cell2=900s/update+2200s child;
   whole5600s. First failure stops;no automatic retry. Lease-free CPU gaps.
   Helpers dispatch_formal_blas_group.py,fullsize_formal_cell_blas.py,
   fullsize_configs.py,plan02,budgets1/2 and the first formal JSON report are
   immutable while this group is live. Review both actual outputs,then complete
   missing Python calibration;long CPU Attention remains deferred.

Completed accelerator pilots (all terminal-parent audited):

| Cell | Case | Batch/physical | Warmup/measured s | B512 forecast warmup/measured s,including1.15 |
| --- | --- | --- | --- | --- |
| 2 | resident add inference | 8/4 | 5.265367/5.147500 | 387.530991/378.856023 |
| 5 | resident add training | 4/2 | 22.728816/19.825407 | 3117.543127/2701.048755 |
| 8 | resident attention inference | 8/4 | 6.458343/6.310399 | 475.334045/464.445368 |
| 11 | resident attention training | 2/1 | 27.637039/24.794583 | 7398.533431/6550.711632 |
| 1 | mixed-a add inference | 64/32 | 51.433151/62.055762 | 473.184986/570.913014 |
| 7 | mixed-a attention inference | 32/16 | 61.855470/81.838889 | 1138.140655/1505.835549 |
| 76 | Python mixed-a Add training | 32/32 | 279.218476/336.290254 | 5101.002177/6167.096223 |
| 82 | Python mixed-b Attention training | 8/4 | 147.516218/181.253649 | 10706.075535/13258.712427 |

Python streaming Add/cell88 also passed:B32/physical32,warmup292.447311s,
measured347.113487s;B512 forecasts5345.784679/6365.153116s. Budget02 separately
permits7500s/update,15400s process,retaining the original3000s refusal.

Forecasts are not actualB512 timing. Audit helper
TASK/launchers/audit_accelerator_calibration02.py passed on the terminal parent.
Final cell94 also passed:measured176.356590s,B512 forecasts10350.964783/12894.607619s.
Its budget02 separately declares16500s/update,33400s process,retaining3000s refusal. First formal audit helper
TASK/launchers/audit_formal_first_cpu.py passed on the terminal first actual result.
The serial group helper now has the required reviewed prerequisite.

## Next actions and full-size scope

Active plan: TASK/plans/fullsize-continuous-e69b3bd-blas16-02.json,
SHA58e0bbab0a891b3645b3d64d35d788e405837ce5bbbb5a624e7a6a4b23aedc2e.
120 FP32 cells,24 static configuration envelopes,11 NPUs for primary comparisons,
60GiB/card or512GiB CPU. Old plan01 is retained history. All geometry/capacity
unchanged;plan02 adds explicit CPU BLAS startup16.

CPU Attention trainingphysical16,mixed4,resident1;CPU/mixed Add32,resident Add
training2/inference4. Resident Attention uses the qualified explicit11-owner map.
No unqualified48-row limit. One continued warmup+one measured step,two windows,
SGD training. Formal cell0/repeat1 is complete;all other full-size matrix
cells and recommendation repeats remain pending.

Budget02 files exist for cells0–11 (unsubmitted except cell0). Original3000s
forecast refusals remain:resident Add training3117.543s→separate4500s/update;
resident Attention training7398.533s→9000s/update;CPU Attention training9692.312s
→12000s/update,24400s process. Smaller inference cells use tighter limits.
Python cell76 also has a separate7500s/update,15400s process budget after
its5101.002177/6167.096223s forecasts refused3000s. Each file hashes its actual
pilot basis;none certifiesB512 runtime. Python cell82 has16500s/update,33400s
process after10706.075535/13258.712427s forecasts refused3000s.

1. Accelerator calibration completed and audited10/10;do not repeat it.
   Evidence commit d416e3c is pushed. First formal cell0 is also audited.
2. Execute/audit the short Add inference NPU pair (cells1,2). Keep original
   work counts and strict-equivalence limitation beside the CPU comparison.
   No recommendation before three fresh processes. Defer longest CPU Attention.
3. Finish remaining necessary Python original-width calibration:CPU four model/
   mode envelopes,mixed inference and Python-owned resident paths have no new
   original-width continued pilot yet. Do not assume representative/small gates
   or a C++ timing certify these Python timings. Unsubmitted helper
   remaining_python_calibration.py prepares these10 serial cases after the first
   actual formal audit;whole20300s,first failure stops. It is parsed,not qualified.
4. Execute remaining full-size family/client/schedule/model/mode flows,three fresh
   processes before formal recommendations,separate full-size resident/FP16
   profiling/comparisons,then final support/portability/evidence audit and task-job
   closure. All ten representative screens and unrelated passed gates stay closed.

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
d416e3c ten accelerator pilots;first formal CPU evidence is being committed with
this handoff. No production code changed in this continuation. Task-local audit/budget helpers are outside
the source repository;live versions above are immutable. Continue after commits.
