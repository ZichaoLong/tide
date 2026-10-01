# Current handoff

Updated 2026-10-02. **ACTIVE: user resumed and accepted overhead reduction.**
Continue the overall goal; commits/pushes authorized. No current pause instruction.
No subagents. Reference repositories and ObsidianVault are read-only.
Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository; branch graph-execution-foundation.
[execution-flows.md](execution-flows.md) is the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Overall task remains incomplete.

## Contract and focus

Candidates independently consume common inputs/parameters/initial state,never
CPU reference events/routes/results/gradients. General online greedy accepts legal
topology/input including positive-delay feedback and natural streaming fallback.
Preserve int64,stable ordering,parallel-edge identity,missing/zero messages,None/zero
gradients. Matrix: PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU ×
streaming/prefill × inference/complete training. Python resident is a C++/CANN client,
not an independent PyTorch scheduler. Five presets retain fine switches; FP32 main,
FP16 separate. CUDA execution is target-machine-pending. Training means loss/VJP/
optimizer/continuation/throughput,not downstream convergence. Contract outranks
run-ml-experiments: minimal existing durable records;Trackio does not block.
Public complete consumers and F6 take priority over repeated internal polishing.
Only affected checks,not unchanged8,954 CPU checks. Implementation commit →
immutable qualification → evidence commit;push each.

## Latest qualified work

Actual CPU/mixed Add/Attention consumers: cleanfe2d886,CPU120/NPU18;
[evidence](evidence/online-consumers-20261002.md). Actual resident FP32 consumers:
clean0d61cb9,CPU59/resident19/mixed18;[evidence](evidence/online-resident-consumers-20261002.md).
Public multi-device training and portable repartition: clean0d7c45e;
[evidence](evidence/public-sharded-training-20261002.md). HARD slot-affine adjoints
and publication: clean96c75f8;[evidence](evidence/resident-emission-training-20261002.md).

Reverse budget splitting: clean106cbeb293847e054525fce21f68a8d1c8a13b13;
[evidence](evidence/resident-reverse-budget-20261002.md),evidence commitb538906 pushed.
Eight jobs passed:CPU1,cache83,consumer21,FP32/FP16 each16 retained trajectories,
512 windows/128 updates total. D32 requested reverse16 selects owner12/query12/key64
and matches explicit1-row plus independent CPU. Separate actual two-card training
trace has no observed AiCPU. Local tensor reservation splitting,not total-memory
admission. Public Gradients ABI changed;clients must match/rebuild. Statistics are
physical capacities/estimates,not observed active rows or allocator peaks.

Public multi-device inference: implementation7329c71f5f48267fed23a1656a821ae059364b2f,
committed/pushed. [Report/audit](evidence/public-sharded-inference-20261002.md).
All7 immutable-source jobs PASSED; audit sharded_inference_evidence.py exited0.
CPU74/library47/actual consumers27,no skips or tolerance changes. Library FP32/FP16
covers3 families,both schedules,distinct Full/state maps,save/reset/load,and2→1/3
card continuation. Actual standalone/Python clients:12 FP32 multi-card inference
plus12 complete-training trajectories,96 windows/24 updates. Legacy single-device
C++ constructor and TrainingPlacement binding alias retained. Model freeze keeps
TensorImpl aliases. Placement persists through Python reset/load;manifests report
requested/resolved maps. Consumer timing distinguishes inference/training.
Separate D32 Attention2-card inference profile,4 windows,diagnostics off:
4,790 AI_VECTOR_CORE/170 AI_CORE/16 MIX_AIV,no observed AiCPU,no reverse/VJP/
optimizer/journal operators;12 model executions,332 notification pairs,1,172
switches,2,432 async copies. Includes construction,not formal throughput.

Source sources/sharded-inference-clean01; backend builds sharded-inference-clean01
and sharded-inference-python-clean01; installed client sharded-inference-consumer-clean01.
Backend links reused recursively byte-verified development objects and unchanged
kernels/core. Runs build-sharded-inference-{,python-,consumer-}clean01 and
sharded-inference-{cpu,library,consumer,profile}-clean01. Do not rerun completed jobs.
Development CPU dev01 fixture-name collection failure retained;corrected dev02
PASSED74. All other development jobs passed. No current build/test/profile jobs live.

## Next work

Evidence commit f3ea9cb for7329c71 is pushed. Public multi-device inference is done.
Projection-retention development passed: both backend builds, installed standalone
consumer build, library26, actual consumers27, FP32/FP16 matrix32 trajectories plus
single-device FP16 legacy12 (704 windows/176 updates). Snapshot projection-retention-dev01
contains the exact changed source; every development job is terminal PASSED.
RetainedProjection owns one immutable projection-bank copy per training update,
shared by retained windows; backward/detach/close release it. Three family cases
validate exact two-window capacity, one-byte-short refusal before progress, three
nonzero updates and optimizer restore against independent CPU autograd. No-projection
training and inference consumer regressions also passed. Reservations are not
allocator peak measurements; parameters/physical gradients are still coordinator-owned.

Projection retention implementation3f85852552c607e7e44dc9480d41eafca1027081 is pushed.
All8 clean qualification jobs PASSED; projection_retention_evidence.py audit exited0.
Evidence docs/evidence/resident-projection-retention-20261002.{md,json}. No current
qualification jobs live. Separate trace14,782 AI_VECTOR_CORE/607 AI_CORE/218 MIX_AIV,
no observed AiCPU; no timing claim. Builds/snapshot projection-retention-clean01,
projection-retention-python-clean01,projection-retention-consumer-clean01. Reused
client objects after source/header/compiler-option audit,fresh link/loader.

Current uncommitted next implementation: compact projection banks and shared
ProjectionStage forward/reverse device chunk services; Ascend C actual-owner row
packing,content-flow lifecycle,retained immutable shard copies,canonical sources/
publication and Full placement cost all connected. Test projection_shards3 adds
physical bank distribution/full-state checks plus three nonzero updates and restore.
Consumer records now distinguish dense vs compact projection placement.

Compact projection implementation is ready to commit/push. All development work
terminal: standalone/Python builds dev01/dev02 and installed consumer dev02 PASSED;
library-dev03 PASSED30,lean-dev04 PASSED2(no-journal FP32/FP16 remote-only bank),
matrix-dev02 PASSED32 trajectories,legacy-dev02 PASSED12,placements-dev02 PASSED3,
three-dev02 PASSED1(FP16 3→2 cards),actual consumers-dev02 PASSED27. Total standalone
48 trajectories/768 windows/192 updates. Keep library-dev02 failure(3 passed then
CPU reference fixture omitted packed=False); only test configuration was corrected.
Backend sources are unchanged since frozen projection-shards-dev02. dev03/dev04
change tests only. Latest live test also writes physical-bank reservation JSON for
qualification; no numeric or tolerance changes. No current jobs live other than
intentionally suspended historical CPU run.

Next commit implementation,push,freeze projection-shards-clean01. Build standalone
and Python via build_projection_shards.py --reuse-host corresponding-dev02
--kernel-build corresponding-dev02. Install/fresh-link consumer with
build_projection_retention_client.py --reuse-client projection-shards-consumer-dev02;
client C++/public headers remain byte-identical. Clean library31,standalone matrix32/
legacy12/placements4 via projection_shard_gates.py,actual consumer27,separate
profile_resident_consumer.py --preset resident --reverse-chunk-rows16. Audit using
projection_shards_evidence.py COMMIT,then evidence-only commit/push. No unchanged
CPU full suite. Do not duplicate terminal development jobs.

After projection shard qualification: total per-device memory admission,actual
consumer FP16,representative five-preset
screening and full-size F6. No new formal full-size throughput result. place_full
currently counts only Full parameter bytes. ContentBudget accounts module tensor
reservations,not total per-device allocator peaks; state/KV,tapes,optimizer proposals,
communication,head/loss need total admission. Existing ShardedFull/RemoteFull/
ShardedFullVjp provide device packet/completion patterns for projection partitioning.

## Environment and job bounds

TASK=/mi/data2T/zlong/tide-execution-flows;RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service;RUN/status.json/task.log and queue.json.
Module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Default shell Python cannot run Torch tests. Public /opt stack supersedes dated
personal guide by user authorization. TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;
preserve module PYTHONPATH,prepend snapshot/python. SoC Ascend910_9392;leased logical
NPUs only. Separate standalone/Python runtimes. Core builds placement-cpu-clean01,
placement-npu-clean01(standalone),placement-npu-python-clean01(Python-owned).
freeze_run.py;runtime cwd env -C RUN prevents vendor files polluting snapshots.
background.slice/Nice10,2 build workers,queue120s/run600s/build900s. Last disk:
data222GiB/root14GiB;check before heavy writes. Atomic handoff durable_records.replace_text.

Historical historical-cpu-attention-01 remains intentionally SIGSTOP;pause.json
outweighs running status and retains memory/TASK/timing.lock. Do not resume or kill.
Historical Add CPU78.793172/NPU4 47.932888ms/token = throughput1.6438× faster;
it does not certify resident. Earlier24 dirty files retained on pushed archive/
restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37,sha256 inventory
TASK/restricted-flow-archive.json. Preserve all cited artifacts and failures.
