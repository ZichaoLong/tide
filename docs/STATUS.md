# Current handoff — 2026-10-11

The authorized G1–G5 **local delivery is complete**:implementation and CPU/NPU
correctness closure,finite selection evidence and GPU target handoff. Actual GPU
validation remains pending hardware. No current task job or performance queue
is live;the protected historical record is excluded. No new pause instruction
exists. User authorization includes commits/pushes and occupied NPU use;no
empty-card admission or foreign-process operations. No subagents.

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`,branch
`graph-execution-foundation`. Always use explicit repository cwd.
`TASK=/mi/data2T/zlong/tide-execution-flows` below. Sole backlog:
[ROADMAP G1–G5](ROADMAP.md);contract:[execution-flows §11](execution-flows.md).
This handoff is part of the final evidence commit;resolve current HEAD with Git.
No production model path changed after the cited qualification sources.

## Delivered implementation and exact qualification

| Area | Delivered / verified scope |
| --- | --- |
| State/Read | Built-in batch first-order VJPs,grouped finite checks,exact Norm FP32 order;Add reuses full content groups/private row views. Public storage,version checks,None/zero and per-tick rounding retained. Full/Aggregate batch VJPs already consumed |
| CPU | Clean e565f96 composite9650 unique FP64/FP32 accepted cases (9648 original+2 same-source admission repairs),654 optional skips;22 topology,10 installed,12 matching-core CTests |
| NPU public modules | Clean4f23993:236 positive Python/native FP32/FP16 cases+2 explicit CSR refusals;independent standalone2-graph/3-step/checkpoint/nondefault-stream checker |
| Scale repair | Clean00b1775:per-region selection control metadata fixed;5 CTests,8 CPU and8 NPU complete training cells,3 connected windows. Separate historical FP16 .002/.02 scale policy unchanged |
| State/Read checker | Test-only e3c86d7 input stream boundary correction;FP32/FP16 each48 configurations,336 windows,18 refusals,default/nondefault streams. Production libraries unchanged;CUDA checker compile/link/help passed |
| NPU resident | Composite133 component invocations:33 original+98 completion at9c,2 corrected State/Read at e3;installed client and547 public cases/zero skips. Complete declared families/schedules/training/continuation/checkpoint/capacity/1–3 owners |
| CUDA local | Independent CUDA conditional/P2P backend,102 semantic sources;fresh a926703 core/resident/clients,7 host CTests,1 semantic CTest,271 CPU checks/84 optional device skips. Matching imports/absent-device refusal;repaired scale CUDA consumer5 host CTests. No GPU execution |
| Profile tooling | 1bba497 separates raw capture from offline export;9fd5263 accepts CANN decimal timestamp strings. Clean49 then55 focused CPU lifecycle checks;actual deferred exports complete18 parser tasks,validate device events and unchanged original hashes |

Evidence:[initial](evidence/batched-vjp-cuda-local-20261010.md),
[Add follow-up](evidence/add-batch-followup-20261010.md),
[NPU modules](evidence/npu-module-batching-20261011.md),
[environment](evidence/qualification-environment-20261011.md),
[precision/stream gates](evidence/precision-and-stream-gates-20261011.md),
[resident composite](evidence/resident-composite-20261011.md).
No old pass is relabeled as a new-source test. Strict discrete comparison and
[original near-tie failure](evidence/original-add-route-witness-20261004.md) remain.
Resident Full uses its existing explicitly selected conditioned softmax policy,
not a relaxed discrete check. CPU/NPU declared correctness is verified;GPU
compiled/source-delivered remains separate from target-verified execution.

## Finite performance outcome

[Six pure PyTorch NPU components](evidence/npu-batched-components-20261011.md):
32×128 FP32,one independent process/variant,1 warmup/1 measured step. Read
10.9906→1.9430ms,Add14.7226→5.6592ms,Attention65.2333→28.6773ms. Actual traces
support lower host attachment/backward work;these are not complete native graphs.

[Complete standalone selection](evidence/native-selection-20261011.md):clean00b,
TimedDAG prefill,D64/logicalB8/T4/V257,32body/112edges,FP32,2 connected windows,
1 warmup/1 measured full training step,AddSGD/AttentionAdamW. One Torch/node
thread;NPU2owners. No profiler in these five independent timed processes.

| Model | CPU seconds | mixed-c seconds | resident seconds |
| --- | ---: | ---: | ---: |
| Add | 0.443972 | 2.987056 | 0.484066 (physicalB4) |
| Attention | 1.396360 | 8.616372 | no accepted timing:cache-adjoint budget refusal |

Prefer CPU at this moderate scale;Add resident is close to CPU and beats mixed-c
here. Shared external contention and single samples prevent a stable universal
ranking. All Add routes report103 outputs/1648 events;Attention CPU/mixed report
99/1590. Aggregate agreement is not proof of complete trajectory equality.
No new full-size measurement or full-matrix restart was justified.

Three complete traces passed: Add/Attention mixed-c and Add residentB4,with
actual two-card operators and observed allocator calibration. Add resident has
212 queue tasks;8 host completion waits each contain24–28 queue tasks without
same-thread host launch/copy/submission/branch changes. CANN Model Id is a
sentinel,so this evidence uses timing/API boundaries plus the implementation.
Mixed paths retain many submissions,device switches and syncs;resident costs
include gather/scatter,journals and packing. Counts alone do not prove a bottleneck.

Original Add residentB8 was refused by its reverse-packet subbudget. Static
all-choice topology bounds gave17,043,456B(B8) versus8,523,776B(B4),under an
11,184,810B pack reservation. A separately declared B4 case passed,with no budget
increase and unchanged logical batch/window/update boundaries. Attention B4
then hit its cache-adjoint reservation;no further split search. Total HBM and
private reverse reservations are distinct. Both refusals remain failed.

Known costs:custom/nondefault replay,selected-clear and causal depth,per-slot
connectivity,semantic Next replay,small tensors/public clones;resident gather,
scatter,scans,communication and budget sizing;CUDA one-thread semantic adapter,
serial reductions/scatter and uncalibrated peaks. These are future optimization
opportunities,not a new local job queue or missing declared first-order semantics.

## Terminal artifacts and source identities

All current-stage task services are terminal,MainPID0/empty cgroups;guarded tasks
have no surviving child groups. `g5-native-selection01` passed5 cases in68.42s,
peak summed RSS5,019,664,384B. `g5-final-audit02` passed source/binary/result/trace,
device coverage and memory review in48.47s,peak95,584,256B,without device work.
Audit01's API-prefix assertion failure is retained.

TASK/plans/NAME.{json,py,sh} hold frozen commands/budgets;TASK/runs/NAME hold
status.json,task.log,bounded.json and workload records:

- `g2-npu-components10`:Read replay timing accepted later from retained raw;
  inline export exceeded8GiB. `g2-read-replay-export01/02` failed before useful
  export;03 recovered235 kernels without remeasurement.
- `g2-npu-components11`:Read batched capture/export completed;summary failed on
  string timestamps. `g2-profile-summary-cpu-clean01/read-composite.json`
  accepts corrected summaries of both original Read traces,without rerunning them.
- `g2-npu-components12`:four previously unmeasured State cases passed;component
  audit `plans/audit-g2-components-20261011.py` and reviewed evidence committed644c1a9.
- `g2-npu-complete-profiles12`:Add mixed passed,Add residentB8 refused.
  `g2-npu-complete-profiles13`:Attention mixed/Add residentB4 passed,Attention
  residentB4 refused. Raw trace and native failed results remain immutable.
- `g5-native-selection01`:five single unprofiled measurements passed.
  `g5-final-audit02/{audit,residency-windows}.json`:final trace/receipt review.
  Original batch09/10/11 and failed child receipts remain failed.

Primary immutable source directories under TASK/sources:
`g-stage-followup-clean03`=e565f962da63a7e5ceb5ffdbd0ec6699db95c296;
`g-stage-followup-clean01`=a926703e9a41ec6cd41bcf8f2f83665f36bdb999;
`g-stage-artifact-cwd-clean01`=4f23993315554f68be3352d7d15533a0e445b260;
`g-stage-profile-raw-clean01`=9c3ffd3acad1a6be31f37e349e6505a14ffc4e1b;
`g4-scale-control-clean01`=00b1775ce15afa20fb0552d51f6afdf4392ccc1c;
`g4-state-read-clean01`=e3c86d70795c594eafbc587388d9e984c3f92612;
`g2-profile-export-clean01`=1bba49799fe9a2549a58fa9c55f668219f69ed89;
`g2-profile-export-clean02`=9fd52630e5154956f489f29256610ee34922d455.

Core digest:`0e2d0baa8fb2e3db1aed807c20c4ef78e53838922295dd4f60fb7578313266f0`.
Original resident digest:`53dfba264427b829ead61e59524e28d490488f1ce854c767be43b00b1ccf058b`;
e3 changed only its checker within device_online. Builds: TASK/builds/g3-npu-*-clean02;
CUDA core **g3-cuda-core-clean02** (one core for both consumers),resident
`g3-cuda-resident-clean02`;scale fixes `g4-scale-control-clean01` and
`g4-scale-control-cuda-clean02`. Reused CPU core `g1-add-assembly-dev01` keeps its
dirty construction provenance. Source bridges and all failure records remain.

## Next action and operating constraints

No local required command remains. On a GPU host,rebuild a clean final commit
and follow [device-control target commands](device-control.md):complete gate,
actual residency trace,multicard/continuation,memory calibration,then a finite
performance plan. Actual GPU correctness/residency/multicard/performance and
other untested host architectures/version tuples remain pending. Selector/new
upstream semantics still wait for the user's updated canon.

For read-only re-entry: `git status --short --branch` and `python scripts/status.py`
from the explicit target cwd. To inspect the last audit:
`systemctl --user show tide-g5-final-audit02.service -p ActiveState -p Result
-p MainPID -p ControlGroup`;read TASK/runs/g5-final-audit02/audit.json.
There is no current task to stop. New experiments need a new finite declaration,
not reused output paths,repeated timings or resurrected historical queues.

Use public module `libtorch-npu/2.10.0-cann9.0.0`;Python is
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
GPU tools are under `/mi/data2T/zlong/gpu-toolchains`,private module
`torch-cuda/2.10.0-cu128`. Verify actual environment before use;no shared driver
or vendor-file edits. Site launcher `plans/g-stage-shared-launch02.py` honors
occupied-card authorization,two snapshots,health/headroom and sorted physical
visibility. Thread/compiler pools1;resident ACLNN_CACHE_LIMIT=0. Raw device
initialization/caches stay in artifact directories. Data disk last observed80GiB
free;recheck before large writes. Trackio remains off.

Long work uses frozen sources,separate background.slice services,affinity,
sampled aggregate RSS/deadlines and process-group cleanup. Legacy cgroup-v1
CPUQuota/MemoryMax/TasksMax are requested,not proven kernel-enforced. Shared
mappings count repeatedly in RSS. No automatic retries or timeout enlargement.

**Never resume,stop,signal or clean historical-cpu-attention-01 / worker2686919.**
Its retained running record is excluded from current task closure. Malformed
historical build-reverse-gather-python-dev01 may make status.py exit1;do not
rewrite it or treat it as a new blocker. ObsidianVault,lh and fractal-latcarf stay
read-only. Old F1–F7 and61/120 timings keep their original scope and closed budgets:
[earlier terminal report](evidence/selection-terminal-20261010.md).
