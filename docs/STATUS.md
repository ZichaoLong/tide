# Current handoff — 2026-10-11

The authorized G1–G5 stage is active after the earlier F1–F7 closure. No pause or
permission request is pending. Occupied NPUs are explicitly authorized; no empty
card admission or foreign process operations. No subagents. Repository
`/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`,branch
`graph-execution-foundation`;always use the explicit repository cwd.
`TASK=/mi/data2T/zlong/tide-execution-flows` below. Contract:
[execution-flows §11](execution-flows.md);sole backlog:[ROADMAP G1–G5](ROADMAP.md).

## Active work and exact next action

**Passed `tide-g4-npu-resident-completion01.service`** at unchanged clean
9c3ffd3acad1a6be31f37e349e6505a14ffc4e1b,cwd
`TASK/sources/g-stage-profile-raw-clean01`. It passed exactly98 previously unstarted
component invocations,then the installed client and547 public cases with zero skips.
It retains33 accepted resident10 invocations;the2 corrected State/Read dtype
invocations passed separately at clean e3c86d7. Production resident libraries
are unchanged. This is a composite qualification,not a new whole-matrix rerun.

Physical3,7,9 map to logical0,1,2. Budget8CPU/24GiB/6000s service,5955s guard,
5400s workload;first failure stops. Separate background.slice user service.
Plan:`TASK/plans/NAME.{json,py,sh}`. Outputs:`TASK/runs/NAME/{status.json,
bounded.json,sharing.json,gate.json,components/result.json,public.xml,task.log}`.
Inspect unit with `systemctl --user show tide-NAME.service -p ActiveState
-p Result -p MainPID -p ControlGroup`,then inspect stage logs/receipts. Never
call a live stage passed. Stop command for this project-owned task if requested:
`systemctl --user stop tide-g4-npu-resident-completion01.service`.

Also passed CPU-only `tide-g4-scale-control-cuda-clean02.service`,clean00b1775,
cwd `TASK/sources/g4-scale-control-clean01`:fresh CUDA-linked scale consumer
and5 CTests after the control repair.6CPU/8GiB/900s service,855s guard,2 build jobs.
plans/NAME.{json,sh};runs/NAME/{status.json,bounded.json,task.log};builds/NAME.
CUDA core is builds/g3-cuda-core-clean02 (one core for both runtime consumers).
Attempt01 failed before build because its site launcher was not created after
an incorrect core-path assertion;failed receipt retained,no survivors.
Terminal zero exit,empty cgroup,no survivors;all5 CTests passed.

**Failed `tide-g-stage-device-batch09.service`:**`TASK/plans/g-stage-device-batch09.{json,py}` at
clean00b1775,cwd `TASK/sources/g4-scale-control-clean01`. Its full
resident/scale/module/CPU prerequisites have passed and been audited. It
contains only `g2-npu-components10`→`g2-npu-complete-profiles10`:6 component cases
and4 complete traces,only Read replay timing completed before export failure. Child service ceilings1500s+2700s;
controller4500s. Existing inner/workload budgets unchanged. No second tranche.

All correctness/source prerequisites passed. Performance batch09 and components10
failed at first Read replay profile export:aggregate RSS exceeded the unchanged
8GiB inner cap. Both units are terminal,MainPID0/empty cgroups;guard has no survivors.
The independent timed measurement completed (0.010990634560585022s),with parity
checked,then profiler export failed. Preserve measurement.json,raw collection and
all failed/running nested receipts. No timing repeat or budget enlargement.
Remaining5 components and all4 complete traces are unstarted,not a live queue.
Failed tide-g2-read-replay-export01.service before analysis (wrong API entry);
terminal,MainPID0/empty cgroup,no guarded survivors. Retained unchanged.
Failed export02:cap1 prematurely exited with pending tasks (NON_BLOCKING helper);
no trace produced. Terminal,empty cgroup,no survivors.
Passed tide-g2-read-replay-export03.service:explicit text output and
blocking timeline after export,one parser worker. Offline raw recovery found235
actual kernel events;measurement0.010990634560585022s and original raw unchanged.
4CPU/8GiB/180s service,135s guard;11.1167s,peak2,892,038,144B;terminal zero exit,
empty cgroup,no survivors. Original batch09/components10/export01/export02
remain failed. Plans/runs retain exact hashes and logs.

Implemented deferred NPU capture and standalone offline exporter;23 directed CPU
lifecycle tests passed. No graph/model/backend implementation changed. Implementation committed and pushed as 1bba49799fe9a2549a58fa9c55f668219f69ed89.
Passed tide-g2-profile-export-cpu-clean01.service at1bba497:49 focused CPU
checks,zero skips;terminal success,empty cgroup,no survivors. Separate evidence
commit will audit XML/receipts alongside the new actual profile lifecycle.

Failed tide-g-stage-device-batch10 and g2-npu-components11:Read batched
capture passed,measurement0.0019429884850978851s. All18 offline parser tasks
succeeded and trace exists;summary failed because TorchNPU text timestamps
are decimal strings. Original failed receipts retained;no remeasurement.
Four remaining components and4 native traces unstarted;no active device queue.

Summary normalizes profiler timestamp/duration numbers only;model int64 clocks
unchanged.29 directed CPU tests passed and the retained actual trace summary
now parses. Repair committed/pushed9fd5263. Passed g2-profile-summary-cpu-clean01:
55 CPU checks plus summary-only recovery of both existing Read traces;terminal
success/empty cgroup/no survivors. Read replay10.99063456ms/235kernels;
batched1.94298849ms/21kernels. No repeated model execution or export.

Terminal failed tide-g-stage-device-batch11.service at clean9fd5263,
cwd TASK/sources/g2-profile-export-clean02;controller4500s. Dependency chain:
1. tide-g2-npu-components12.service:4 unmeasured Add/Attention replay/batch
   cases at9fd5263;8CPU/12GiB/1500s service/1200s workload;each capture+export
   shares180s/8GiB RSS/2GiB output. Read cases explicitly excluded.
2. tide-g2-npu-complete-profiles12.service:4 never-started native traces,
   clean00b1775,cwd TASK/sources/g4-scale-control-clean01;8CPU/16GiB,
   2700s service/2600s workload;600s/12GiB/2GiB per case. Profile/native
   computation unchanged;build identities checked against that exact source.
First failure stops;no retries,larger limits,or full-size queue. Shared cards
allowed;two snapshots and >=10GiB headroom,no operations on other users/jobs.
Inspect systemctl --user show tide-g-stage-device-batch11.service -p ActiveState
-p Result -p MainPID -p ControlGroup,then current child assessment/log.
TASK/plans/NAME.{json,py,sh};TASK/runs/NAME/{status,bounded,sharing,assessment}.json,
task.log;controller uses batch.json. Stop only if newly requested via
systemctl --user stop on these exact new task units. Old protected work untouched.
Components12 passed all4 cases;the six-case composite is audited in
docs/evidence/npu-batched-components-20261011.{md,json}. Complete profiles12
passed Add mixed-c (125063 ops;observed allocator growth38,956,544/38,110,720B),
then failed Add resident before valid timing:compact reverse packet subbudget.
All task units terminal,MainPID0/empty cgroups/no survivors. Attention cases unstarted.
HBM admission is distinct from per-window/per-owner reverse tensor subbudgets.
Next:derive a safe physical sample cap from existing static traffic bounds,then
declare a finite new split configuration plus the still-unstarted cases. Do not
repeat original B8 resident or mixed Add. GPU hardware validation remains pending.
Controller: TASK/runs/g-stage-device-batch09/{status.json,batch.json,task.log};
case: TASK/runs/g2-npu-components10/{assessment.json,read-replay.log,
read-replay/measurement.json,read-replay/raw};plan and input hashes remain sealed.

Latest pushed implementation:e3c86d70795c594eafbc587388d9e984c3f92612 (test-only
State/Read stream boundary). Repair evidence is committed;complete resident evidence/support claims were committed as ffe4e49;active jobs read immutable snapshots,not this worktree.
Audit writer:`TASK/plans/audit-g4-repair-evidence01.py`.

## Implemented and qualified scope

| Area | Evidence and boundary |
| --- | --- |
| State/Read | Built-in batch first-order VJPs,grouped finite checks and exact Norm FP32 VJP order;Full/Aggregate batching already consumed. Custom/nondefault replay,causal depth,per-slot connectivity and small tensors remain |
| Add assembly | Reuses full content batches and removes private structural-graph row clones;public storage/version behavior and per-tick rounding retained. No fusion/new speedup claim |
| CPU | Clean e565f96 composite accepts9650 unique FP64/FP32 cases:9648 original+2 exact resource-admission repairs;654 optional skips. Also22 topology,10 installed and12 matching-core CTests |
| NPU modules | Clean4f23993:236 positive public cases+2 explicit CSR refusals across Python/native FP32/FP16. Independent standalone2-graph/3-step/checkpoint/nondefault-stream checker passed |
| Scale repair | Clean00b1775:fresh standalone consumer5 CTests,8 CPU and8 NPU complete training cells. Region control metadata fixed;payload ownership unchanged. Separate scale FP16 historical0.002/0.02 policy retained |
| State/Read checker | Clean e3c86d7:NPU FP32/FP16 each48 configurations,336 windows,18 refusals with default/nondefault streams;fresh checker linked unchanged hashed production libraries |
| NPU builds | Clean a926703 dual cores,resident owners,installed/online/scale clients;7 resident host+5 scale CTests. Matching e565f96 imports and4 real profile preparation calls passed |
| CUDA local | Clean a926703 core/resident/clients,102 semantic sources/1 CPU CTest,7 host CTests,271 CPU tests/84 optional skips. e565 imports/absent-CUDA refusal and e3c86d7 changed checker compile/link/help passed;no GPU execution |
| Profiles/GPU | Complete mixed/resident CANN entry implemented/prepared;Read replay timing retained;5 components and4 traces unstarted. Full GPU target commands ready;correctness,residency,multicard,memory/performance await hardware |

Evidence:[initial](evidence/batched-vjp-cuda-local-20261010.md),
[Add follow-up](evidence/add-batch-followup-20261010.md),
[NPU modules](evidence/npu-module-batching-20261011.md),
[environment](evidence/qualification-environment-20261011.md),
[control precision/stream gates](evidence/precision-and-stream-gates-20261011.md),
[resident composite](evidence/resident-composite-20261011.md).
CPU/NPU declared correctness profiles are verified with exact cited scopes;CUDA
actual execution remains pending. NPU profiling/performance remain separate. Incidental correctness timings are not benchmarks.

## Immutable inputs and terminal receipts

| Snapshot under TASK/sources | Commit / use |
| --- | --- |
| g-stage-followup-clean03 | e565f962da63a7e5ceb5ffdbd0ec6699db95c296,CPU composite/imports/preparation |
| g-stage-followup-clean01 | a926703e9a41ec6cd41bcf8f2f83665f36bdb999,NPU/CUDA full builds |
| g-stage-vendor-env-clean01 | dd5e22060d6521ed291938858f40efa54b5c18bd,vendor PYTHONPATH correction,46 CPU checks |
| g-stage-artifact-cwd-clean01 | 4f23993315554f68be3352d7d15533a0e445b260,artifact cwd/relative libs,50 CPU checks,module composite |
| g-stage-component-cwd-clean01 | 7da524268cc8b1eb677cbd99d2847f6690778ea4,component cwd,30 CPU checks |
| g-stage-profile-raw-clean01 | 9c3ffd3acad1a6be31f37e349e6505a14ffc4e1b,raw-retaining profiler callback,31 CPU checks |
| g4-scale-control-clean01 | 00b1775ce15afa20fb0552d51f6afdf4392ccc1c,scale repair,NPU training09 and planned profiles |
| g4-state-read-clean01 | e3c86d70795c594eafbc587388d9e984c3f92612,checker repair,NPU clean01/CUDA host01 |

Core digest:`0e2d0baa8fb2e3db1aed807c20c4ef78e53838922295dd4f60fb7578313266f0`.
Original resident all-source digest:`53dfba264427b829ead61e59524e28d490488f1ce854c767be43b00b1ccf058b`;
e3 changes only its test file,not production libraries. Source bridges:
`plans/g-stage-{profile,scale}-source-bridge01.json`. Backend builds:
`builds/g3-{npu,cuda}-*-clean02`;new scale:`builds/g4-scale-control-clean01`.
CPU matching core:`builds/g1-add-assembly-dev01` retains dirty construction provenance.
Do not claim an old test ran on a later evidence commit.

All gates cited as passed have terminal zero exit,clean snapshots and no guarded
survivors. Important retained failures:CPU full06 (2 resource-admission repairs);
modules04 vendor-path,modules05 RSS,modules06 site cwd,modules07 timeout after168
positives (completion01 supplied68+2 refusals);training08 FP16 control dtype;
resident09 unsorted physical visibility;resident10 State/Read repeat-bound.
Diagnostic01 missed a quoted-include directory before testing;diagnostic02 passed
both no-sync/sync variants,so resident10 was not deterministically reproduced.
The checker nonetheless omitted its required input boundary;production already
had it. No original failure is relabeled. Detailed receipts are under their
unchanged TASK/runs/NAME directories. Nested interrupted records may still say
running;use the enclosing terminal guard/unit,not stale nested states.

## Finite performance scope and operating controls

Components:one NPU,Read/Add/Attention replay/batch FP32,32×128,180s/case,
1200s workload/1500s service,8CPU/12GiB,inner8GiB RSS/2GiB total output.
Complete traces:two NPUs,Add/Attention×mixed-c/resident,600s/case,
2600s workload/2700s service,8CPU/16GiB,inner12GiB RSS/2GiB output per case.
Packets:`packets/g2-complete-{add,attention}-profile02/workload.json`:
D64/B8/T4/V257,32nodes/112edges,TimedDAG prefill,2 connected windows,
1 warmup+1 collected training step;Add SGD/Attention AdamW,8GiB/card,
aggressive safe sample chunks. Offline0.68–2.27GB/card estimates are not observed
calibration. Profiles diagnose execution,not unprofiled throughput.

Resident Full selects its existing explicit conditioned softmax comparison,
[resident-full-vjp](resident-full-vjp.md);its original width257 strict failure
is retained. Scale FP16 uses [its separate historical policy](accelerator-scale.md).
Other VJP/tensor/update tolerances and exact discrete routes are unchanged.

Occupied-card launcher:`plans/g-stage-shared-launch02.py`,two occupancy snapshots,
healthy cards with10GiB observed headroom,least-load selected set sorted for RTS
visibility. Records external contention;no exclusive lease or empty-card rule.
Public module:`libtorch-npu/2.10.0-cann9.0.0`;Python:
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
CANN compiler pools TE_PARALLEL_COMPILER,TBE_PARALLEL_COMPILER,
CANN_KNOWLEDGE_BANK_PROCESS_NUM,MAX_COMPILE_CORE_NUMBER are1;Torch pools1,
TASK_QUEUE_ENABLE=0,ACL_OP_INIT_MODE=0;resident ACLNN_CACHE_LIMIT=0.
No vendor installation edits. Device probes/caches run in artifact directories.

All long jobs use separate background.slice services and fixed source. Legacy
cgroup v1 does not prove requested CPU/memory/task kernel enforcement.
`plans/g-stage-run-bounded01.py` supplies affinity,sampled summed RSS and deadline;
shared mappings count multiple times. No automatic retries/timeout enlargement,
old matrix restart or undefined second tranche. Heavy timing is serial;Trackio
is off. Data disk last observed97GiB free;recheck before large writes.

## Selection limits and protected history

[Earlier CPU evidence](evidence/batched-selection-cpu-20261010.md) retains its
source:component ratios Read3.16×,Add1.28×,Attention2.96×;complete standalone Add
3.810711s→4.090873s,RSS701222912B→814317568B. It predates Add assembly and shows
no complete-graph gain under external load. Attention was refused before
allocation. No extrapolation to new NPU results;retain CPU,mixed-a/b/c,resident
and fine controls. Custom replay,causal depth,public clones,per-slot connectivity
remain. CUDA initial one-thread semantic adapter,serial reduction/scatter and
uncalibrated peaks require target evidence. GPU toolchains:
`/mi/data2T/zlong/gpu-toolchains`;private module torch-cuda/2.10.0-cu128;
[target commands](device-control.md).

Old F1–F7 and61/120 timings keep original source/config;missing cells are not a
queue. [Old terminal report](evidence/selection-terminal-20261010.md).
Preserve [strict near-tie witness](evidence/original-add-route-witness-20261004.md):
CPU246 one FP32 ULP ahead;resident245/246 tie selects245;events2325/2327.
No relaxed discrete check or uniform strict cross-device equivalence claim.

**Never resume,stop,signal or clean historical-cpu-attention-01 / worker2686919.**
Malformed historical build-reverse-gather-python-dev01 may make status.py exit1;
preserve it. No project writes in ObsidianVault;reference repos read-only.
Selector/new upstream semantics remain deferred until the user updates the canon.
