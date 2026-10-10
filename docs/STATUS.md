# Current handoff

Updated 2026-10-10 (Asia/Shanghai). The G1–G5 stage is active and **not globally
closed**: CPU correctness, local CUDA/NPU builds, CPU selection evidence and GPU
handoff are ready; new NPU device qualification/profiling is resource-blocked.
The user has authorized the follow-up implementation and two-stage experiment plan.
Current work: reduce Add internal batch assembly/storage overhead with CPU regressions;
prepare complete standalone mixed/resident profiling and dependency-ordered finite
submission entries. Performance execution may wait for resources. GPU hotspot
parallelization follows real-device qualification/profiling.
Implementation a926703, profile-entry tests adbd51e and owner-map test correction
2d45fee are committed/pushed. Current HEAD e565f96 also corrects the CANN profile storage validator and
a backend-neutral refusal test; compiled core/resident/consumer source remains
identical to a926703. Qualification/build work is running below;
no device experiment is submitted. This is not a user pause. Continue authorized
work; commits never impose a pause. No subagents. Selector/new upstream semantics
remain deferred. Contracts: [execution-flows §11](execution-flows.md).
Sole backlog: [ROADMAP G1–G5](ROADMAP.md).

## Current follow-up implementation and qualification

Add reuses whole content batches and omits private structural-graph row clones;
public states still own storage, with first-order/version/None rules preserved.
The C++ virtual interface changed: matching rebuilt backend cores are required.
`g1-add-assembly-dev01` passed477 directed FP64/FP32 Add/storage/clock/State/Read
tests from a frozen dirty snapshot of8523353. Its development tree digest is
`17ecec8fc01ecccd00af2855d1ccd734de29a31fdba51978505989e9cdbb8555`.
The exact new core digest is
`0e2d0baa8fb2e3db1aed807c20c4ef78e53838922295dd4f60fb7578313266f0`.
Raw archive/log/status/development records remain in `runs/g1-add-assembly-dev01`.

The complete standalone profiling entry has28 directed identity/result/CSV
checks and22 CLI/failure checks passed;2 optional binary skips were followed by
both standalone CLI cases passing with the historical byte-fixed CPU consumer
(launcher compatibility only). Clean `g2-profile-entry-clean01` at adbd51e passed15
lifecycle/acceptance tests without vendor initialization.

`g1-cpu-qualification04` is **cancelled**, exit143, MainPID0/empty cgroup. Its old
test stub had index but no type for the already backend-neutral owner-map adapter;
a directed reproduction failed before tensor work. The corrected test covers
NPU/CUDA names and all18 owner-map checks passed. Reproducer and interrupted
nested records stay in `runs/g1-cpu-qualification04`; they are not passing evidence.

Qualification05 has one diagnosed stale assertion so far: sample-chunk refusal
expected `native NPU`, while the shared resident entry now reports
`native CUDA/NPU`. Its fixed-source reproduction and corrected one-case check
are retained in `runs/g1-cpu-qualification05/sample-chunk-{reproducer,fix}.log`.
The full suite continues to collect all failures before another clean gate;
The sample-chunk assertion fix and CANN profiler minimum-storage correction
(200MB, matching local msprof help) passed18 directed checks, retained in
`runs/g2-profile-cann-bound-dev01.log`. These are committed/pushed as e565f96 separately
from the ongoing gate; compiled core/resident/consumer sources are unchanged.
Qualification06 is being launched from clean e565f96 in
`sources/g-stage-followup-clean03`, with the unchanged90min/6CPU/12GiB budget and
independent process-group guard. Qualification05 continues solely to collect its
full failure inventory; no performance timing is running. Batch02 now requires
qualification06 and admits only the exact tests/docs/profile-validator delta
from a926703, plus unchanged compiled core/resident/consumer identities.

| Active job / service `tide-NAME.service` | Fixed source / scope | Limits |
| --- | --- | --- |
| `g1-cpu-qualification05` | clean2d45fee, read-only `sources/g-stage-followup-clean02`; collecting full regression failures; diagnosed obsolete refusal assertion | 90min,6CPUs,12GiB,256tasks |
| `g1-cpu-qualification06` | clean e565f96, read-only `sources/g-stage-followup-clean03`; full CPU,22 topology cells,installed consumers,CTest; exact reused `builds/g1-add-assembly-dev01` with dirty construction provenance retained; independent RSS/affinity/deadline guard | 90min,6CPUs,12GiB,256tasks |
| `g3-npu-clean02` | cleana926703, read-only `sources/g-stage-followup-clean01`; fresh dual-runtime cores/resident,installed/online/scale clients,host gates | 3h,4CPUs,16GiB,512tasks |
| `g3-cuda-clean02` | same cleana926703; fresh core/dual-runtime resident/installed/online,102 CPU semantic sources and directed CPU tests | 3h,4CPUs,16GiB,512tasks |

All use separate background.slice services, job.py records and two build workers;
no timing jobs overlap. Legacy cgroup v1 keeps cpu/memory/pids controllers at
parent slices: the table records requested systemd controls, not verified kernel
enforcement. Build workers2, thread pools1 and wall deadlines are explicit;
profiling separately guards process-group RSS/storage/affinity. The pending
device batch now wraps every child with the existing foundation_lifecycle RSS/
affinity/deadline guard, and its launch/timeout/reap checks passed. Retained snapshot:
`runs/g-stage-resource-controls01.json`. Logs/receipts: `runs/NAME/{task.log,status.json}`. No running
job is accepted as passed. After builds finish, validate actual profile commands
with `--prepare-only` and new native/resident imports without device initialization.

## Source and reviewed delivery

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. Initial clean HEAD matched c0ce4ee5. Previous implementation
afbc07556c693e9809bb005154a4b9b806d16c51 and reviewed delivery8523353 are committed/pushed.
The follow-up above changes source; old qualification/build evidence below stays
tied to its original identities until the new gates complete.

G1 State/Read132cada, accounting fixes8a4d60f and exact Norm VJP fixa02c18c;
G3 CUDA/shared boundary bda25de; component profiler d1df03c; target capacity/
control-policy ecafdf4, logical0 peer entry5018133, compiled-consumer identity
afbc075. Previous core digest
`dbc370539ed261846524c427b7b9a37a38f6857ba6a6ccb78ad365168140a150` belongs to those receipts. Frozen qualification sources remain intact.

| Area | Delivered / exact evidence boundary |
| --- | --- |
| State/Read | Built-in batch first-order VJPs and grouped finite checks; exact Norm FP32 operation order restored. Full/Aggregate batching already existed. Custom/nondefault replay, causal/structural limits and small tensors remain |
| CPU correctness | Clean a02c18c:9595 FP64/FP32 tests,654 optional skips,22 full topology cells including positive-delay feedback,10 installed checks;12 matching-core standalone CTests also passed |
| CUDA local | Clean bda25de: standalone/Python/core/installed/combined builds;102 semantic sources,7 host CTests plus1 semantic CTest,229 CPU tests/84 hardware skips. Clean afbc075:6 entry identity tests and absent-CUDA refusal. Dynamic imports passed without device initialization |
| NPU local | Clean bda25de public-stack dual-runtime/core/resident/installed/combined/scale builds;7 resident host and5 scale CTests. Python imports passed without device initialization. No current-source device correctness or performance claim |
| Limited CPU selection | Six PyTorch component measurements and two standalone Add measurements accepted; Attention baseline refused before allocation, candidate never started. Batch terminal failed as declared; no retries/full-size expansion |
| GPU target | Complete3-device zero-skip gate and tracing commands ready. Real correctness, residency, multicard, memory calibration and performance all pending |

[Qualification/build report](evidence/batched-vjp-cuda-local-20261010.md) and
[finite CPU selection review](evidence/batched-selection-cpu-20261010.md) have
reviewed JSON companions with source/record hashes. Support matrix NPU status
is now implemented for current source; historical verified entries are retained as passed source-scoped attempts,
with their original commits/scopes and contents. CPU qualified; CUDA implemented/build-tested only.

## Terminal jobs and retained artifacts

`TASK=/mi/data2T/zlong/tide-execution-flows`. Paths below use this root.
Ignored `artifacts/execution-flows-g*` links expose these same receipts to
`scripts/status.py`. Current active jobs are listed above. Dormant prepared-plan
links have no run directories and are not submissions. Historical records and the known malformed record are unchanged.
Previously passed services below have MainPID0 and empty cgroups; inspect with `systemctl --user show UNIT -p ActiveState
-p Result -p MainPID -p ControlGroup` and `runs/NAME/{status.json,task.log}`.

- `g1-cpu-qualification03`: passed clean a02c18c; gate in `runs/NAME/gate`,
  source `sources/NAME`, byte-matched reused core `builds/g1-norm-dev06`.
- `g4-cpu-standalone01`: passed12 CTests; clean afbc075 launch source
  `sources/g3-target-identity-clean01`, same core. `runs/NAME/ctest.xml`.
- `g3-cuda-clean01`, `g3-npu-clean01`: passed clean bda25de from
  `sources/g3-backends-clean01`; builds `g3-{cuda,npu}-*-clean01`.
- `g3-target-identity-clean01`: passed6 tests and correct absent-CUDA refusal
  on clean afbc075. `g3-plugin-import01`: both CUDA/NPU imports passed on the
  same source and matching bda25de binaries; no device initialized.
- `g2-cpu-assessment01`: failed/exit1 at case9 as its stop policy required.
  Component source clean ecafdf4 (`sources/g2-components-ecafdf4`), complete
  baseline c0ce4ee (`sources/g2-baseline-c0ce4ee`) and candidate a02c18c.
  Eight valid measurements retained. Explicit physicalB32 Attention estimated
  25,662,653,552B against6GiB admission; no allocation, OOM or timing result.

Reviewed audits: `runs/g1-g3-reviewed-audit01.json` and
`runs/g2-cpu-assessment-reviewed01.json`; auditor scripts of corresponding
names in `plans/`. They refuse overwrite and never rerun workloads. Original
CPU gate01 remains73 failed/9451 passed/723 skipped, gate02 cancelled after
Norm diagnosis. NPU qualification01/02 remain allocation timeouts with no code
execution. CUDA unavailable01 wrapper remains failed on an obsolete expected
error substring; `runs/g3-cuda-refusal-audit01.json` separately accepts the
retained preflight refusal. Earlier failed development attempts remain retained.

## Resource blocker and exact next commands

Latest retained read-only inspection: 2026-10-10T11:41:18.704642+00:00; all16 chips
were unavailable and no free physical IDs were returned.
`runs/g4-npu-resource-inspection03.json` retains the observation. Public module
libtorch-npu/2.10.0-cann9.0.0 and msprof were rechecked; no device workload ran.
No process was touched. Recheck healthy/no-process/HBM/utilization before a
new allocation; use the helper's two-snapshot and advisory-lock checks. Do not
mechanically retry an unchanged timed-out lease. The resource-information
question has no answer yet; authorization to execute is already present.

Prepared batch `plans/g-stage-device-batch02.{py,json}` is **not submitted**.
It runs clean e565f96 and hashes all plans, packets and offline capacity records.
Prerequisites: qualification06 at e565f96 and both backend clean02 gates at a926703;
the controller verifies the exact docs/profile-validator/test differences, unchanged core/
resident/compiled-consumer identities, passing receipts and terminal services.
Batch01 remains a superseded unsubmitted plan because qualification04 was cancelled.
The new controller's initial check-only correctly refused the still-running
qualification05. Plan Python/shell syntax passed; complete preflight is pending.

| New plan in `plans/` | Dependency / workload | Service resources |
| --- | --- | --- |
| `g4-npu-modules03.{sh,py}` | CPU/backend predecessors passed;1NPU, all named Python/native FP32/FP16 modules and standalone checker;2700s workload | 50min,6CPUs,12GiB,192tasks |
| `g4-npu-training03.{sh,py}` | modules03 passed;1NPU,8 positive-delay PDG Add/Attention × SGD/AdamW × FP32/FP16 training cells;2500s | 45min,6CPUs,12GiB,192tasks |
| `g4-npu-resident03.sh` | training03 passed;3NPUs, complete strict target gate;5400s | 100min,8CPUs,24GiB,256tasks |
| `g2-npu-components03.{sh,py}` | all correctness passed;1NPU,6 independent FP32 Read/Add/Attention replay/batch cases32×128;180s/case,1200s total | 25min,8CPUs,12GiB,192tasks |
| `g2-npu-complete-profiles03.{sh,py}` | components03 passed;2NPUs,4 complete Add/Attention × mixed-c/resident traces;600s collection/export per case,2600s workload | 45min,8CPUs,16GiB,256tasks |

Every allocation wait is capped at60s. `plans/g-stage-run-bounded01.py` guards
every child with its declared RSS and CPU affinity plus a deadline45s below the
service ceiling; this does not enlarge workload budgets. `bounded.json` records
peak sampled RSS and remaining child groups. RSS sampling is not an instantaneous
kernel allocation cap. Launch/affinity and deliberate timeout/reap checks are
retained in `runs/g-stage-bounded-{launch,timeout}-check01.json` (the intentional
timeout stays failed). Components cap process-group RSS at8GiB
and outputs at2GiB; complete profiles cap each group at12GiB and outputs at2GiB.
Packets `packets/g2-complete-{add,attention}-profile02/workload.json` use
D64/B8/T4/V257,32 body nodes/112 edges,TimedDAG prefill,two connected windows,
one warmup and one collected training step; Add SGD/Attention AdamW. Offline
8GiB/card admission estimated0.68–2.27GB/card and full physical batch8; **observed
memory calibration is still required**. Physical chunks preserve batch/gradient/
optimizer boundaries. Four profiles diagnose execution, not throughput.

First failure stops the dependency chain and leaves successors unstarted. Child
service ceilings total15900s; controller16200s. No retries,timeout extension or
full-size/second-tranche cases. Further implementation and selection cases must
follow concrete profile evidence. Old plans/results/failures stay unchanged.

After qualification06 and both backend services pass, validate without submitting:

```bash
/usr/bin/python3 /mi/data2T/zlong/tide-execution-flows/plans/g-stage-device-batch02.py --check-only
```

Only after successful preflight and a fresh safe resource opportunity, submit:

```bash
TASK=/mi/data2T/zlong/tide-execution-flows
systemd-run --user --unit=tide-g-stage-device-batch02 --service-type=exec --slice=background.slice \
  --working-directory="$TASK/sources/g-stage-followup-clean03" \
  --property=KillMode=control-group --property=Nice=10 --property=CPUQuota=100% \
  --property=MemoryMax=512M --property=TasksMax=128 --property=RuntimeMaxSec=16200 \
  --setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin --setenv=PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python3 scripts/job.py --output-dir "$TASK/runs/g-stage-device-batch02" \
  -- /usr/bin/python3 "$TASK/plans/g-stage-device-batch02.py"
```

`runs/g-stage-device-batch02/batch.json` records all rows, including unstarted
successors. Inspect current jobs with `systemctl --user show tide-NAME.service
-p ActiveState -p Result -p MainPID -p ControlGroup` and `runs/NAME/{status.json,task.log}`.
GPU target commands remain in [device-control](device-control.md); local builds
and CPU checks do not replace real-device correctness,residency,multi-card,memory
or performance. No device allocation has occurred in this follow-up.

## Performance interpretation and remaining implementation costs

Single CPU component ratios: Read3.16×, Add1.28×, Attention2.96×. Complete
standalone Add3.810711s→4.090873s with equal loss/output/event counts and
peak RSS701,222,912B→814,317,568B. No whole-graph gain or stable regression size
is established under external load. Attention has no complete measurement.
Keep CPU/mixed-a/b/c/resident and fine controls; no new universal route winner.
Read/State composition is not kernel fusion. Add's clone/view/structural work,
custom replay, causal depth and per-slot connectivity groups remain costs.
Resident already has explicit device VJPs; CUDA's initial one-thread semantic
adapter, serial reductions/scatter and uncalibrated memory remain target limits.

Public NPU module `libtorch-npu/2.10.0-cann9.0.0`; Python explicitly under
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311`. Standalone SDK
and wheel owners stay separate. Resident launch sets `ACLNN_CACHE_LIMIT=0` and
`ACL_OP_INIT_MODE=0` as previously evidenced. Private CUDA module
`~/privatemodules/torch-cuda/2.10.0-cu128`, tools under
`/mi/data2T/zlong/gpu-toolchains`. No GPU here; no shared drivers changed.
Last disk check root64GiB/data101GiB free; recheck before large writes.

## Protected history

Old F1–F7 qualification and61/120 selection timings retain only their original
sources/configurations; missing cells are not a queue. [Old terminal report](evidence/selection-terminal-20261010.md).
Preserve the [original strict near-tie witness](evidence/original-add-route-witness-20261004.md):
CPU246 oneFP32ULP ahead; resident245/246 tie selects245; proposal error7.7039e-6,
events2325/2327. No relaxed discrete comparison or universal equivalence claim.

**Never resume, stop, signal or clean historical-cpu-attention-01 / worker2686919.**
Malformed historical build-reverse-gather-python-dev01 can make status.py exit1;
preserve it. No project writes in ObsidianVault; all reference repositories stay read-only.
