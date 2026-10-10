# Current handoff — 2026-10-10

The authorized G1–G5 stage continues after the previous F1–F7 local closure.
State/Read batching, the Add assembly follow-up and full declared CUDA resident
source are implemented. Current CPU qualification, fresh backend builds and
host preparation have passed. All current task services are terminal. The
finite NPU batch is **prepared, preflight-passed and not submitted** because
no safe device was available in the latest retained inspection. This is not
closure of device qualification or performance work. Selector/upstream semantic
extensions remain deferred. No historical queue or old pause instruction resumes.

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. Run repository commands with this explicit cwd.
The original handoff matched clean c0ce4ee5; subsequent authorized implementation
commits through e565f96 are pushed. The evidence commit containing this handoff
is documentation/support-matrix only; no implementation is left uncommitted.
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
still say running and are not accepted. No active current-task job remains.

## Prepared finite device batch and next action

Latest retained read-only inventory is `runs/g4-npu-resource-inspection05.json`,
2026-10-10T13:39:53.097319Z:16 devices,zero free IDs,3-device request unsatisfied.
No NPU workload was launched. Recheck health/processes/HBM/utilization at a new
resource opportunity; allocation uses the helper's two snapshots and advisory
locks. Do not retry an unchanged timed-out lease or touch another workload.
Execution is already authorized; further startup permission is unnecessary.

`plans/g-stage-device-batch02.{py,json}` is fixed to clean e565f96. Its complete
`--check-only` passed at13:21:50Z; receipt
`runs/g-stage-device-batch02-preflight01.json`. It checks CPU completion01,
backend clean02 builds and host preparation01, exact allowed source differences,
core/resident/compiled-consumer identities, sealed plan hashes and unused outputs.
The batch and all five children are **unsubmitted**, not queued. Batch01 and older
child plans remain superseded unsubmitted artifacts, not work to resume.

| Planned child in dependency order | Scope | Service ceiling |
| --- | --- | --- |
| g4-npu-modules03 | 1NPU; named Python/native FP32/FP16 modules plus standalone checker;2700s workload | 50min,6CPU,12GiB,192tasks |
| g4-npu-training03 | 1NPU;8 positive-delay PDG Add/Attention × SGD/AdamW × FP32/FP16 training cells;2500s | 45min,6CPU,12GiB,192tasks |
| g4-npu-resident03 | 3NPU; complete strict target qualification;5400s | 100min,8CPU,24GiB,256tasks |
| g2-npu-components03 | 1NPU;6 Read/Add/Attention replay/batch FP32 cases32×128;180s/case,1200s total | 25min,8CPU,12GiB,192tasks |
| g2-npu-complete-profiles03 | 2NPU;4 complete Add/Attention × mixed-c/resident traces;600s collection/export per case,2600s total | 45min,8CPU,16GiB,256tasks |

Each allocation wait is capped at60s. First failure stops the chain and leaves
successors unstarted. Child service ceilings total15900s; controller16200s.
No retries,timeout expansion,full-size cases or second tranche is declared.
Heavy measurements are serial; Trackio stays off. After profiles, choose concrete
implementation changes, requalify them, then declare limited selection cases.

Packets `packets/g2-complete-{add,attention}-profile02/workload.json` use
D64/B8/T4/V257,32 body nodes/112 edges,TimedDAG prefill,two connected windows,
one warmup and one collected training step; Add SGD/Attention AdamW. Offline
capacity plans fit8GiB/card,estimated0.68–2.27GB/card with physicalB8; this is
not observed memory calibration. Physical chunks preserve logical batch,
Attention normalization,gradient and optimizer boundaries. Complete traces
are diagnostic, not throughput measurements.

Validate again without submission at the next resource opportunity:

```bash
/usr/bin/python3 /mi/data2T/zlong/tide-execution-flows/plans/g-stage-device-batch02.py --check-only
```

After a fresh safe resource opportunity and successful preflight, submit once:

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

`runs/g-stage-device-batch02/batch.json` will record every task,including unstarted
successors. Inspect only current-task units with `systemctl --user show UNIT
-p ActiveState -p Result -p MainPID -p ControlGroup` and their durable receipts.
GPU rebuild/full target/residency commands remain in [device-control](device-control.md).
No local build or CPU check certifies actual GPU execution.

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
