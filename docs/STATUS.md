# Current handoff — 2026-10-11

The authorized G1–G5 stage continues after the previous F1–F7 local closure.
State/Read batching, the Add assembly follow-up and full declared CUDA resident
source are implemented; current CPU qualification and backend builds retain the
exact source scopes below. Occupied NPU devices are explicitly authorized.
The new clean4f23993 CPU gate passed50 cases; the complete mixed-pdg NPU
resume calibration passed with peak aggregate RSS10379898880B (9.67GiB).
The two earlier device attempts exposed
qualification-environment defects/resource costs, not a passed device suite:
batch03 lost CANN PYTHONPATH in its resume child; batch04 reached five first-case
checks then exceeded the original12GiB aggregate RSS cap. Performance has not run. Batch05/modules06 also failed before device selection:
the shell changed cwd before the site launcher imported its source helpers.
New batch06/child07 keeps launcher cwd fixed; numerical children still use OUT.
Its import/preflight checks passed; all prior receipts remain unchanged.

The vendor-path correction is committed at dd5e220 and passed46 clean CPU tests.
This next implementation keeps vendor-generated files in output directories,
resolves relative native libraries before changing cwd and restores cwd on
failure.32 directed CPU tests passed (`runs/g4-artifact-cwd-dev01.xml`). Next:
inspect running batch06 and child07 below. The correction is committed and pushed
at4f23993315554f68be3352d7d15533a0e445b260; CANN compiler pools are limited to1. The calibration is passed and a fresh finite device batch is running with unchanged
work/time/memory bounds. Do not resubmit failed identities or old matrices.
Selector/upstream semantic extensions and actual GPU execution remain deferred.

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. Use that explicit cwd. The initial c0ce4ee5 handoff
has advanced through authorized pushed work; no rollback is required.
`TASK=/mi/data2T/zlong/tide-execution-flows` below. Contracts:
[execution-flows §11](execution-flows.md). Sole backlog: [ROADMAP G1–G5](ROADMAP.md).

## Implemented and verified scope

| Area | Current result and boundary |
| --- | --- |
| State/Read | Built-in batch first-order VJPs and grouped finite checks; exact Norm FP32 operation order. Full/Aggregate batching already existed. Custom/nondefault replay, causal depth, per-slot connectivity and small tensors remain costs |
| Add follow-up | a926703 reuses full content batches and removes private structural-graph row clones; public states retain independent storage/version behavior and per-tick rounding. No new speedup or kernel-fusion claim |
| CPU qualification | Clean e565f96 accepts9650 unique FP64/FP32 cases:9648 full-run passes plus exactly2 same-source resource-admission repairs;654 optional skips. Also22 topology cells,10 installed consumers and12 matching-core CTests passed |
| NPU local | Clean a926703 fresh Python/standalone cores,resident owners,installed/online/scale clients;7 resident host and5 scale CTests. Clean e565f96 dynamic imports without device initialization and4 actual complete-profile preparation commands passed. New-source device correctness/performance pending |
| CUDA local | Clean a926703 fresh core,standalone/Python resident,installed/online consumers;102 semantic sources/1CPU CTest,7 host CTests,271 CPU tests passed/84 optional device skips. Matching e565f96 imports and absent-CUDA refusal passed. No GPU execution |
| Profiling and selection | Complete native mixed-a/b/c/resident CANN entry implemented; synthetic lifecycle and host preparation passed. Prior finite CPU selection retains its original scope. New6 component measurements and4 complete traces are prepared, with no new timings |
| GPU target | Full strict3-device zero-skip target gate and trace commands ready. Real correctness,residency,multicard,memory calibration and performance require a GPU host |

[Current follow-up evidence](evidence/add-batch-followup-20261010.md) and its JSON
record the source/build identities, composite CPU acceptance and unsubmitted
batch. [Initial stage evidence](evidence/batched-vjp-cuda-local-20261010.md) remains
scoped to a02c18c/bda25de/afbc075. The CPU support cells remain verified; NPU/CUDA
remain implemented for current source, with old verified attempts preserved.

CPU/profile/host source: `e565f962da63a7e5ceb5ffdbd0ec6699db95c296`, frozen at
`sources/g-stage-followup-clean03`. Backend build source:
`a926703e9a41ec6cd41bcf8f2f83665f36bdb999`, at `sources/g-stage-followup-clean01`.
The exact difference is docs/execution-flows.md, profile_execution_flow.py and
three test files; all compiled sources match. Current core digest:
`0e2d0baa8fb2e3db1aed807c20c4ef78e53838922295dd4f60fb7578313266f0`.
Resident digest: `53dfba264427b829ead61e59524e28d490488f1ce854c767be43b00b1ccf058b`.
The CPU build `builds/g1-add-assembly-dev01` retains its frozen dirty development
construction provenance; exact source/binary matching precedes clean qualification.

## Terminal records and retained failures

Each job has `runs/NAME/{status.json,task.log}`, `plans/NAME.sh` and unit
`tide-NAME.service`. Ignored `artifacts/execution-flows-g*` links expose these
records to status.py; prepared plans without run directories are not submissions.

| Job | Terminal result |
| --- | --- |
| g1-cpu-completion01 | passed, clean e565f96; audited composite inventory then22 topology/10 installed/12CTest; gate/result.json |
| g1-cpu-resource-repair01 | passed, same e565f96; exactly2 resource-admission cases,8 visible CPUs,48.37s; tests.xml |
| g3-cuda-clean02 | passed, clean a926703; fresh CUDA builds and host checks; builds/g3-cuda-*-clean02 |
| g3-npu-clean02 | passed, clean a926703; fresh dual-owner NPU builds and host checks; builds/g3-npu-*-clean02 |
| g3-followup-host01 | passed, clean e565f96; fresh imports,expected CUDA refusal,4 prepared profile commands; preparation.json |
| g2-profile-entry-clean01 | passed15 synthetic lifecycle tests, clean adbd51e; subsequent18 directed storage/refusal checks in runs/g2-profile-cann-bound-dev01.log, covered by the new full CPU inventory |
| g1-cpu-qualification04 | cancelled/exit143 after obsolete owner-map stub diagnosis; reproducer retained, test corrected in2d45fee |
| g1-cpu-qualification05 | failed:9647 passed/654 skipped/1 stale backend-name assertion; reproduced and fixed in e565f96 |
| g1-cpu-qualification06 | failed:9648 passed/654 skipped/2 pre-execution resource-admission refusals; exact same-source repair above, original receipt unchanged |

Qualification06's6-CPU affinity exposed the CLI suite's half-visible-CPU budget:
2 workers plus2 coordinator slots need8 visible CPUs. Both failures had empty
workload lists. Completion01 accepts only that exact failure inventory and the
same-source two-case repair; it does not relabel06. Future complete CPU suite
launches need at least8 visible CPUs. No numerical tolerance or performance
budget changed. Directed Add development had477 passing cases; dirty snapshot
digest `17ecec8fc01ecccd00af2855d1ccd734de29a31fdba51978505989e9cdbb8555`
and build provenance remain in runs/g1-add-assembly-dev01.

Audit `runs/g1-g3-followup-audit01.json` verifies hashes,clean source snapshots,
terminal passed services with MainPID0/empty cgroups and no CPU guard survivors.
Its writer `plans/audit-g1-g3-followup01.py` refuses overwrite. Old audits and
failed/interrupted records remain intact; nested records of cancelled jobs may
still say running and are not accepted. Batch03 and modules04 are terminal failed with MainPID0,empty cgroups and no remaining guarded child groups; original records remain unchanged.

## Current qualification environment and retained device failures

| Job | Result and scope |
| --- | --- |
| g-stage-device-batch03 / g4-npu-modules04 | failed at e565f96: real NPU doctor passed on physical9→logical0; first mixed-pdg fresh resume could not import CANN tbe. Later children unstarted |
| g4-vendor-env-cpu-clean01 | passed46 tests at clean dd5e220 in92.94s;28 directed entry/profile plus18 CPU library qualification cases; peak aggregate RSS814571520B |
| g-stage-device-batch04 / g4-npu-modules05 | failed at dd5e220: doctor and first mixed-pdg's five observable/VJP/update checks passed; fresh resume triggered sampled RSS12909670400B above12884901888B cap. Later children unstarted |
| g4-npu-resume-calibration01 | failed before workload: guard rejected untracked CANN kernel_meta in the frozen dd5e220 checkout; compiler-limited calibration never ran |

Every listed service has MainPID0 and an empty cgroup. Outer job/guard receipts
are terminal; interrupted nested gate/sharing records can still say running and
are not accepted as live or passed. Preserve every original receipt.
`runs/g4-npu-modules05/artifact-relocation.json` hashes the9 generated vendor
files moved to that job's `generated-kernel-meta/`. Tracked snapshot bytes never
changed and `sources/g-stage-vendor-env-clean01` is clean again. The failed
calibration01 record remains unchanged.

Installed CANN sources establish four independent compiler-pool controls:
TE_PARALLEL_COMPILER, TBE_PARALLEL_COMPILER,
CANN_KNOWLEDGE_BANK_PROCESS_NUM and MAX_COMPILE_CORE_NUMBER. The new calibration
will set all four to1 in launch configuration, without modifying installations
or changing numerical semantics. One exact Python FP32 mixed-pdg case includes
fresh-process resume;500s workload,600s service,555s guard,6CPU,12GiB. This is
correctness/resource calibration, not a performance measurement or budget increase.
Vendor-source hashes remain in `plans/g4-npu-resume-calibration01.json`.

Passed qualification identities: read-only clean snapshot `sources/g-stage-artifact-cwd-clean01`,
CPU job `g4-artifact-cwd-cpu-clean01` and NPU calibration
`g4-npu-resume-calibration02`. Both started2026-10-11T08:34:01Z in separate
background.slice services at4f23993 and are terminal passed with MainPID0 and
empty cgroups. NPU physical9 maps to logical0. CPU50 cases passed in94.31s,
peak RSS814440448B; NPU case passed all7 checks with peak RSS10379898880B. No
guarded children remain; the snapshot stayed clean. The CPU gate uses
32 directed tests plus18 library qualification cases,600s/4CPU/8GiB.

Each unit is `tide-NAME.service`; output is `runs/NAME/{status.json,task.log,bounded.json}`,
plus CPU tests.xml or NPU calibration.json/gate/. Inspect with
`systemctl --user show tide-NAME.service -p ActiveState -p Result -p MainPID -p ControlGroup`.
Exact launch commands are recorded in status.json; each uses source-root job.py,
`plans/g-stage-run-bounded01.py` and `plans/NAME.sh`. Do not resubmit either.
Prepared successor `plans/g-stage-device-batch06.{py,json}` / child07 uses the
same4f23993, matched clean02 builds, artifact cwd and compiler limits1. It is
running as `tide-g-stage-device-batch06.service`; the module child is
`tide-g4-npu-modules07.service`. Python FP32 passed all57 named cases;
Python FP16 is running. Native FP32/FP16,standalone,training,resident and
performance remain unaccepted. Both prerequisites and preflight passed. Receipt:
`runs/g-stage-device-batch06-preflight01.json`. Original check-only command (now correctly refuses used outputs):
`python /mi/data2T/zlong/tide-execution-flows/plans/g-stage-device-batch06.py --check-only`.

[Environment qualification evidence](evidence/qualification-environment-20261011.md)
records the completed checks separately from the live batch.

The public `profile_batched_components.py` entry now also changes to its output
cwd before device resolution/model creation, rather than only before trace
export. Resolution failures get explicit failed records and no fabricated device.
Correction7da5242 is committed and pushed.30 directed development tests and30
clean-source CPU tests passed at `g2-component-cwd-cpu-clean01` (0.60s,no skips);
unit terminal passed,MainPID0,empty cgroup,no guard survivors. Immutable source:
`sources/g-stage-component-cwd-clean01`; receipts/JUnit/guard in runs/NAME.
Workload-generation and operation function hashes match4f23993 exactly.
The active batch remains frozen at4f23993 and already launches numerical children
from artifact directories; no running source/plan or timing policy changed.

## Shared NPU execution and finite successor scope

User authorization on2026-10-11 permits occupied NPUs and overrides the skill's
empty-device rule. No permission remains outstanding. Health,observed memory
headroom and declared resource/time bounds still apply; no other process or
protected historical worker is stopped,signalled or changed.

`plans/g-stage-shared-launch01.py` takes two snapshots, allows occupied/high-load
cards, prefers lower load and requires10GiB observed HBM headroom. It records
physical→logical mapping and before/after occupancy in `runs/NAME/sharing.json`.
It neither claims exclusive ownership nor changes account queue/lock records.
Shared-device timings must disclose external contention. Earlier inventory:
`runs/g-stage-shared-resource01.json`; selection is refreshed at each launch.

Calibration and clean CPU qualification passed. The immutable batch06 plan now runs
run these dependencies, each in its own background.slice service:

| Stage | Scope | Service ceiling |
| --- | --- | --- |
| modules | 1NPU; named Python/native FP32/FP16 modules and standalone checker;2700s workload | 50min,6CPU,12GiB,192tasks |
| training | 1NPU;8 positive-delay PDG Add/Attention × SGD/AdamW × FP32/FP16 cells;2500s | 45min,6CPU,12GiB,192tasks |
| resident | 3NPUs; complete zero-skip target qualification;5400s | 100min,8CPU,24GiB,256tasks |
| components | 1NPU;6 Read/Add/Attention replay/batch FP32 cases32×128;180s/case,1200s total | 25min,8CPU,12GiB,192tasks |
| complete profiles | 2NPUs;4 Add/Attention × mixed-c/resident traces;600s/case,2600s total | 45min,8CPU,16GiB,256tasks |

First failure stops successors; total child ceilings15900s/controller16200s.
Resident Full trajectories explicitly select the existing score-conditioned softmax
comparison from [resident-full-vjp](resident-full-vjp.md), whose width257 strict
failure predates this stage. Only that checker uses the named policy; scores,
other values/VJPs/updates retain original tolerances and all discrete routes stay
exact. Original strict failures and the Add near-tie witness remain preserved;
this is not a uniform strict-equivalence claim or a newly relaxed threshold.
No automatic retries,timeout expansion,full-size cases or second tranche.
Heavy measurements are serial; Trackio stays off. Original batch02/child03 plans
were never submitted; batch03/child04 and batch04/child05 retain failed receipts.
A successful calibration must not relabel any of those attempts.

Complete-profile packets `packets/g2-complete-{add,attention}-profile02/workload.json`
use D64/B8/T4/V257,32 nodes/112 edges,TimedDAG prefill,two connected windows,
one warmup and one collected training step; Add SGD/Attention AdamW. Offline
capacity estimates0.68–2.27GB/card fit the8GiB allowance; observed memory is still
uncalibrated. Physical chunks preserve logical batch,Attention normalization,
gradient and optimizer boundaries. These traces diagnose execution,not throughput.
GPU rebuild/full target/residency commands: [device-control](device-control.md).
Actual GPU correctness,residency,multicard,memory and performance remain unverified.

## Resource controls, remaining costs and performance

All long tasks use frozen source and separate background.slice services.
Legacy cgroup v1 leaves cpu/memory/pids controllers at parent slices, so requested
CPUQuota/MemoryMax/TasksMax are not proven kernel-enforced. Snapshot:
`runs/g-stage-resource-controls01.json`. Builds use two workers and thread pools1.
Later CPU gates and every pending device child use existing foundation_lifecycle
through `plans/g-stage-run-bounded01.py` for affinity,sampled aggregate RSS and
an effective deadline45s below the service ceiling. RSS sampling is not an
instantaneous allocation cap. `bounded.json` records peaks and group cleanup.
Launch/affinity and deliberate timeout/reap checks are retained in
`runs/g-stage-bounded-{launch,timeout}-check01.json`; the intentional timeout
remains failed. Component groups cap RSS8GiB/output2GiB; complete profiles
cap RSS12GiB/output2GiB per case. No workload budget was enlarged.

[Prior finite CPU selection](evidence/batched-selection-cpu-20261010.md): Read3.16×,
Add1.28×,Attention2.96× component ratios; complete standalone Add
3.810711s→4.090873s,peak RSS701,222,912B→814,317,568B. No whole-graph gain or
stable regression size was established under external load. Attention was
refused before allocation and has no complete timing. These precede the Add
assembly follow-up. Retain CPU/mixed-a/b/c/resident and fine controls; no universal
route winner. State/Read batching is not proven fusion. Custom replay,causal
depth,public clones and per-slot connectivity remain costs. CUDA's initial
one-thread semantic adapter,serial reductions/scatter and uncalibrated peaks
need target evidence before further optimization.

Public NPU module `libtorch-npu/2.10.0-cann9.0.0`; Python under
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311`.
Standalone SDK and wheel owners remain separate. Resident launch retains
`ACLNN_CACHE_LIMIT=0` and `ACL_OP_INIT_MODE=0`. Private CUDA module
`~/privatemodules/torch-cuda/2.10.0-cu128`,tools under
`/mi/data2T/zlong/gpu-toolchains`. No GPU hardware is present; shared drivers are unchanged.
Last disk check root64GiB/data102GiB free; recheck before large writes.

## Protected history

Old F1–F7 qualification and61/120 selection timings retain their original
sources/configurations; missing cells are not a queue. [Old terminal report](evidence/selection-terminal-20261010.md).
Preserve the [strict near-tie witness](evidence/original-add-route-witness-20261004.md):
CPU246 oneFP32ULP ahead; resident245/246 tie selects245; proposal error7.7039e-6,
events2325/2327. No relaxed discrete comparison or universal equivalence claim.

**Never resume,stop,signal or clean historical-cpu-attention-01 / worker2686919.**
Malformed historical build-reverse-gather-python-dev01 can make status.py exit1;
preserve it. No project writes in ObsidianVault; reference repositories remain read-only.
