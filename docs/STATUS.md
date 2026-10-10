# Current handoff

Updated 2026-10-10 (Asia/Shanghai). The G1–G5 stage is active and **not globally
closed**: CPU correctness, local CUDA/NPU builds, CPU selection evidence and GPU
handoff are ready; new NPU device qualification/profiling is resource-blocked.
All current-stage submitted jobs are terminal; nothing is queued or automatically
scheduled. This is not a user pause. Continue authorized work when resources
permit; commits never impose a pause. No subagents. Selector/new upstream
semantics remain deferred. Contracts: [execution-flows §11](execution-flows.md).
Sole backlog: [ROADMAP G1–G5](ROADMAP.md).

## Source and reviewed delivery

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. Initial clean HEAD matched c0ce4ee5. Latest
implementation is afbc07556c693e9809bb005154a4b9b806d16c51, committed/pushed;
this boundary adds reviewed evidence, support scope and handoff only.

G1 State/Read132cada, accounting fixes8a4d60f and exact Norm VJP fixa02c18c;
G3 CUDA/shared boundary bda25de; component profiler d1df03c; target capacity/
control-policy ecafdf4, logical0 peer entry5018133, compiled-consumer identity
afbc075. Core digest
`dbc370539ed261846524c427b7b9a37a38f6857ba6a6ccb78ad365168140a150` is unchanged
through the latest entry/doc follow-ups. Frozen qualification sources remain intact.

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
`scripts/status.py`;30 submitted stage jobs are terminal. Four prepared-plan
links are intentionally dormant until their run directories are created; they
are not submissions. Historical records and the known malformed record are unchanged.
Every named service is `tide-NAME.service`, background.slice, terminal MainPID0
and empty cgroup; inspect with `systemctl --user show UNIT -p ActiveState
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

Last retained read-only inventory: 2026-10-10T10:54:31.504388+00:00, all16 chips had external
processes. `runs/g4-npu-resource-inspection02.{json,txt}` retains the observation.
No process was touched. Recheck healthy/no-process/HBM/utilization before a
new allocation; use the helper's two-snapshot and advisory-lock checks. Do not
mechanically retry an unchanged timed-out lease. The resource-information
question has no answer yet; authorization to execute is already present.

Prepared new-source plans are **not submitted**. Run them serially; the six
component timing cases start after module correctness and after other current
heavy jobs terminate. This is a new declared scope, not the old experiment queue.

| Plan in `plans/` | Fixed source / acceptance | Service resources |
| --- | --- | --- |
| `g4-npu-modules01.{sh,py}` | `sources/g2-components-ecafdf4`;1NPU, all named Python/native FP32/FP16 State/fiber/Read gates, independent CPU references, standalone FP32 checker;2700s workload | 50min,6CPUs,12GiB,192tasks |
| `g4-npu-training01.{sh,py}` | `sources/g3-target-identity-clean01`; modules01 must pass;1NPU,8 standalone positive-delay PDG Add/Attention × SGD/AdamW × FP32/FP16 complete training cells;2500s total | 45min,6CPUs,12GiB,192tasks |
| `g4-npu-resident01.sh` | same afbc075 source;3NPUs, complete target entry/default strict comparisons,5400s workload; explicit resident cache configuration retained | 100min,8CPUs,24GiB,256tasks |
| `g2-npu-components01.{sh,py}` | same afbc075 source; modules01 and six CPU component cases must pass;1NPU,6 independent FP32 Read/Add/Attention replay/batch cases at rows32,width128;180s/case,1200s total | 25min,8CPUs,12GiB,192tasks |

All four have unique run directories and60s maximum allocation wait. Components
cap each process group at8GiB and total outputs at2GiB; stop on the first failure
with no repeat or budget increase. The whole CPU tranche remains failed, while
its six passed components satisfy this independent NPU component prerequisite.
No complete/resident NPU timing or new full-size case is declared yet: first
inspect actual profiles and memory, then choose only a decision-changing case.

After a safe allocation opportunity, the first launch is exactly:

```bash
TASK=/mi/data2T/zlong/tide-execution-flows
systemd-run --user --unit=tide-g4-npu-modules01 --service-type=exec --slice=background.slice \
  --working-directory="$TASK/sources/g2-components-ecafdf4" \
  --property=KillMode=control-group --property=Nice=10 --property=CPUQuota=600% \
  --property=MemoryMax=12G --property=TasksMax=192 --property=RuntimeMaxSec=3000 \
  --setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin --setenv=PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python3 scripts/job.py --output-dir "$TASK/runs/g4-npu-modules01" \
  -- /bin/bash "$TASK/plans/g4-npu-modules01.sh"
```

Use the same wrapper with the exact plan/source/resource row for later gates.
Verify the unit, queue receipt, physical-to-logical mapping, logs and terminal
acceptance before reporting a pass. Never edit a snapshot used by a live job.
GPU build/gate/trace commands and required3-device/P2P capabilities are in
[device-control](device-control.md); the current local source/build/import
results do not replace any target-machine gate.

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
