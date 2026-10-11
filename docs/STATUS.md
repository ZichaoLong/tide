# Current handoff — 2026-10-11

The authorized G1–G5 stage continues after the previous F1–F7 closure. No pause
or permission request is pending. Occupied NPU devices are explicitly authorized.
State/Read batching, Add assembly/storage and full declared CUDA resident source
are implemented. CPU qualification/build evidence retains its exact source scope.
New NPU public module qualification is complete:236 positive cases plus2 explicit
CSR refusals. The separate standalone module checker also passed.

**Current blocker to the successor batch:** `g-stage-device-batch07` and
`g4-npu-training08` are terminal failed at clean9c3ffd3. Four complete FP32
training cases passed; the first FP16 case failed `consumer control changed FP32
policy` before numerical comparison. All later stages are unstarted. Both units
have MainPID0/empty cgroups; the guard has no survivors. No performance measurement
has run. Preserve all original receipts and budgets.

Next: commit the tested legacy scale control-metadata correction, create a clean
fixed-source snapshot and fresh standalone scale build, and run the matching CPU
and NPU training qualifications. Only tools/accelerator_scale changed; shared
core/resident/online sources stay identical. Retain all original numerical policies.

Development `g4-scale-control-dev01` passed: fresh standalone scale build,5 CTests
and8 CPU complete training cells (Add/Attention × SGD/AdamW × FP32/FP16).
The frozen dirty snapshot records base0eb34c7 plus exactly resident.cpp and its
dtype regression; client digest cf465934a167bd499791a10a9ba06def550ac757992e199f81e9e43568e3b282.
Source and resource record: `runs/NAME/development.json`;6CPU/8GiB/900s service.
MainPID0,empty cgroup,no guard survivors. New test linked to the retained old
adapter fails as expected (`runs/g4-scale-control-red01/result.json`). This is
development evidence,not clean qualification. FP16 scale training explicitly
uses its pre-existing0.002/0.02 policy; public module tolerances are unchanged.

Running: `tide-g4-npu-resident10.service` at unchanged clean9c3ffd3, original
5400s workload/6000s service,8CPU/24GiB/3NPU. Preflight passed and actual component
checks are running; no full pass yet. Physical8,9,14 map to logical0,1,2.
Plan `plans/NAME.{json,sh}`; output
`runs/NAME/{status.json,task.log,bounded.json,sharing.json,gate/result.json}`.
The scale adapter is outside this gate's device_online/online_bench consumers;
module qualification and exact source bridge passed. CPU build may overlap
correctness only; heavy profiling/timing remains serial after qualification.

Resident09 failed before operator execution: visibility9,8,14 violated the
installed TorchNPU/RTS ascending-order requirement and returned zero devices.
Its failed receipt remains. Site launcher `g-stage-shared-launch02.py` sorts
only the selected set;7 synthetic selection/vendor-parser checks passed in
`runs/g-stage-shared-order-check01.json`. No occupancy restriction was added.
Old batch07 remains failed and all child08 successors remain unsubmitted.

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. Always use that explicit cwd. The initial c0ce4ee5
handoff advanced through authorized pushed work; preserve it without rollback.
`TASK=/mi/data2T/zlong/tide-execution-flows`. Contract:
[execution-flows §11](execution-flows.md). Sole backlog: [ROADMAP G1–G5](ROADMAP.md).

## Implemented and verified scope

| Area | Current evidence and boundary |
| --- | --- |
| State/Read | Built-in batch first-order VJPs, grouped finite checks and exact Norm FP32 VJP order; Full/Aggregate batching already used. Custom/nondefault replay, causal depth, per-slot connectivity and small tensors remain |
| Add assembly | Full content batches reused, private structural-graph row clones removed; public storage/version behavior and per-tick rounding retained. No fusion or new speedup claim |
| CPU | Clean e565f96 composite accepts9650 unique FP64/FP32 cases:9648 original passes plus2 exact same-source resource-admission repairs;654 optional skips. Also22 topology cells,10 installed checks and12 matching-core CTests |
| NPU modules | Clean4f23993 accepts236 positive public cases plus2 explicit CSR refusals across Python/native FP32/FP16. Independent standalone2 graph/3-step/checkpoint/nondefault-stream checker passed. Scale complete training and resident gates remain separate |
| NPU builds | Clean a926703 dual Python/standalone cores,resident owners,installed/online/scale clients;7 resident host and5 scale CTests. Matching e565f96 imports and4 actual profile preparation calls passed |
| CUDA local | Clean a926703 core/resident/clients,102 semantic sources/1 CPU CTest,7 host CTests,271 CPU tests/84 optional device skips. Matching e565f96 imports and absent-CUDA refusal passed; no GPU execution |
| Profiles | Complete standalone mixed-a/b/c/resident CANN entry implemented and CPU/preparation checked.6 component measurements and4 complete traces remain unstarted |
| GPU target | Full strict3-device zero-skip target gate and trace commands ready; correctness,residency,multicard,memory/performance await actual hardware |

Evidence: [Add follow-up](evidence/add-batch-followup-20261010.md),
[initial local qualification](evidence/batched-vjp-cuda-local-20261010.md),
[NPU module composite](evidence/npu-module-batching-20261011.md),
[environment corrections](evidence/qualification-environment-20261011.md).
CPU support remains verified; current NPU/CUDA full profiles remain implemented
until their missing target gates pass. Historical evidence keeps its source scope.

## Fixed inputs and terminal gates

| Source / snapshot under TASK/sources | Scope |
| --- | --- |
| e565f962da63a7e5ceb5ffdbd0ec6699db95c296 / g-stage-followup-clean03 | Full CPU composite, imports/profile preparation |
| a926703e9a41ec6cd41bcf8f2f83665f36bdb999 / g-stage-followup-clean01 | Fresh NPU/CUDA backend builds |
| dd5e22060d6521ed291938858f40efa54b5c18bd / g-stage-vendor-env-clean01 | Preserve vendor PYTHONPATH;46 CPU checks passed |
| 4f23993315554f68be3352d7d15533a0e445b260 / g-stage-artifact-cwd-clean01 | Artifact cwd/relative library fix;50 CPU checks,complete NPU resume calibration and module composite passed |
| 7da524268cc8b1eb677cbd99d2847f6690778ea4 / g-stage-component-cwd-clean01 | Component initialization in artifact cwd;30 CPU lifecycle checks passed |
| 9c3ffd3acad1a6be31f37e349e6505a14ffc4e1b / g-stage-profile-raw-clean01 | Connected raw-retaining TorchNPU callback;31 CPU lifecycle checks passed; no actual profile yet |

Core digest `0e2d0baa8fb2e3db1aed807c20c4ef78e53838922295dd4f60fb7578313266f0`;
resident digest `53dfba264427b829ead61e59524e28d490488f1ce854c767be43b00b1ccf058b`.
Exact4f→9c source/workload/timing bridge: `plans/g-stage-profile-source-bridge01.json`.
Backend paths: `builds/g3-{npu,cuda}-*-clean02`; CPU matching core:
`builds/g1-add-assembly-dev01`, whose dirty construction provenance is retained.
Do not claim an old test ran on a later evidence commit.

Audits: `runs/g1-g3-followup-audit01.json`, module evidence JSON and
`plans/audit-g4-modules01.py`. All accepted gates have terminal zero exits,
clean snapshots and no surviving guarded children. The CPU full run06 remains
failed; its2 admission repairs required8 visible CPUs for the CLI half-CPU rule.

## Retained device failures and exact inspection

| Attempt | Retained failure |
| --- | --- |
| batch03/modules04 | Fresh checkpoint child lost CANN tbe path; fixed in dd5e220 |
| batch04/modules05 | Sampled aggregate RSS12909670400B exceeded unchanged12GiB |
| calibration01 | Frozen dd5 source rejected generated kernel_meta before workload;9 hashed files relocated to modules05/generated-kernel-meta,tracked bytes unchanged |
| batch05/modules06 | Site launcher imported source helpers after incorrect artifact chdir; no device selection occurred |
| batch06/modules07 | Original2700s timeout/exit124 after168 positives; exact same-source completion01 supplied68 positives,2 refusals and standalone check |
| batch07/training08 | Four FP32 cells passed; FP16 control metadata assertion failed. Resident08/components08/profiles08 never started |

Each `NAME` has unit `tide-NAME.service` and
`runs/NAME/{status.json,task.log,bounded.json,sharing.json}` as applicable;
exact argv,cwd,source and exit code live there. Training cases:
`runs/g4-npu-training08/{gate.json,float32/gates.json,float16/gates.json}`.
Module composite: `runs/g4-module-completion01/gate.json`.

Inspect with `systemctl --user show tide-NAME.service -p ActiveState -p Result
-p MainPID -p ControlGroup` and the corresponding receipts. All listed device
attempts are terminal. Nested records interrupted by outer deadlines can remain
running; do not accept them as passed or live. Never overwrite/resubmit outputs.

## Remaining finite scope and controls

Previously declared, unstarted stages:3NPUs full resident zero-skip gate
(5400s workload/6000s service,8CPU/24GiB);1NPU six Read/Add/Attention replay/batch
FP32 component cases32×128 (180s/case,1200s workload/1500s service,8CPU/12GiB);
2NPUs four Add/Attention × mixed-c/resident traces (600s/case,2600s workload/
2700s service,8CPU/16GiB). These require a new reviewed successor plan after the
training correction. Original batch07 is closed failed,not a queue.

Resident Full uses the existing explicit score-conditioned softmax policy from
[resident-full-vjp](resident-full-vjp.md); its width257 strict failure predates
this stage. Other tensor/VJP/update tolerances and exact routes remain unchanged.
Standalone scale FP16 has its separately declared historical D8 tolerance policy
in [accelerator-scale](accelerator-scale.md); no public module tolerance changes.

Complete packets `packets/g2-complete-{add,attention}-profile02/workload.json`:
D64/B8/T4/V257,32 nodes/112 edges,TimedDAG prefill,two connected windows,one
warmup and one collected training step; Add SGD/Attention AdamW. Offline estimates
0.68–2.27GB/card fit8GiB/card but are not observed calibration. Physical chunks
preserve logical batch,Attention normalization,gradient and optimizer boundaries.
These profiles diagnose execution; they are not unprofiled throughput results.

User-authorized occupied-card launcher `plans/g-stage-shared-launch02.py` records
two occupancy snapshots and physical/logical mapping,prefers lower load and
requires healthy devices with10GiB observed HBM headroom. No exclusive lease,
empty-card admission or foreign process operations. External contention disclosed.
Public module `libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
TE_PARALLEL_COMPILER,TBE_PARALLEL_COMPILER,CANN_KNOWLEDGE_BANK_PROCESS_NUM and
MAX_COMPILE_CORE_NUMBER are1 in launch configuration. Resident also uses
ACLNN_CACHE_LIMIT=0 and ACL_OP_INIT_MODE=0. No vendor installation edits.

Every long job has fixed source and a separate background.slice service. Legacy
cgroup v1 does not prove requested quotas/memory/tasks kernel enforcement.
`plans/g-stage-run-bounded01.py` adds affinity,sampled aggregate RSS and deadline;
shared mappings are summed. Component inner caps8GiB RSS/2GiB output; complete
profile caps12GiB RSS/2GiB output. No measurement retries,timeout enlargement,
full-size cases or undefined second tranche. Trackio is off; heavy work is serial.
Data disk last observed102GiB free; recheck before large writes.

## Selection limits and protected history

[Earlier finite CPU evidence](evidence/batched-selection-cpu-20261010.md): component
ratios Read3.16×,Add1.28×,Attention2.96×; complete standalone Add
3.810711s→4.090873s with RSS701222912B→814317568B. This predates Add assembly,
shows no whole-graph gain under external load and cannot predict new NPU results.
Attention was refused before allocation. Retain CPU/mixed-a/b/c/resident and
fine controls. Custom replay,causal depth,public clones and per-slot connectivity
remain costs. CUDA one-thread semantic adapter,serial reductions/scatter and
uncalibrated peaks need target evidence. GPU tools live under
`/mi/data2T/zlong/gpu-toolchains`,private module torch-cuda/2.10.0-cu128;
[target commands](device-control.md) remain pending actual hardware.

Old F1–F7 and61/120 timings keep original sources/configurations; missing cells
are not a queue. [Old terminal report](evidence/selection-terminal-20261010.md).
Preserve the [strict near-tie witness](evidence/original-add-route-witness-20261004.md):
CPU246 oneFP32ULP ahead; resident245/246 tie selects245; events2325/2327.
No relaxed discrete check or uniform strict cross-device equivalence claim.

**Never resume,stop,signal or clean historical-cpu-attention-01 / worker2686919.**
Malformed historical build-reverse-gather-python-dev01 can make status.py exit1;
preserve it. No project writes in ObsidianVault; reference repositories read-only.
Selector/new upstream semantics remain deferred until the user updates the canon.
