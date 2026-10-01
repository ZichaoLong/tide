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

## Current work and next commands

Projection retention implementation3f85852552c607e7e44dc9480d41eafca1027081 and
qualification5f8ff8a are pushed. All8 fixed-source jobs passed; see
[evidence](evidence/resident-projection-retention-20261002.md). Do not repeat them.

Compact projection banks and adjoints implementationacb84f30475c03e91cba5c35964565ab1b6316cf
is pushed. Full owners hold physical forward/gradient banks; actual projection rows
and vectors use device chunk services. Immutable retained snapshots, canonical
alias reduction and optimizer publication are connected. No explicit placement
retains the dense path. Reservations are not allocator peak measurements.

Fixed source sources/projection-shards-clean01. Successful backends are
builds/projection-shards-clean02 and projection-shards-python-clean02;
installed client builds/projection-shards-consumer-clean01. All3 builds PASSED.
Library-clean01 PASSED31; matrix-clean01 PASSED32 trajectories,legacy-clean01
PASSED12,placements-clean01 PASSED4. Total48 trajectories/768 windows/192 updates.
Do not repeat these completed jobs. All source is immutable during qualification.

All9 fixed-source qualification jobs PASSED. Actual consumer-clean01 PASSED27;
profile-clean01 PASSED (D32 Attention/two-card,four windows/two AdamW updates).
Audit projection_shards_evidence.py acb84f30475c03e91cba5c35964565ab1b6316cf exited0.
Evidence docs/evidence/resident-projection-shards-20261002.{md,json}; profile reports
15,615 AI_VECTOR_CORE/631 AI_CORE/218 MIX_AIV,no observed AiCPU. No formal timing claim.
Retain failures: library-dev02 CPU oracle fixture missing packed=False; clean01
backend reuse launcher assumed a copied kernel archive. clean02 follows recorded
artifact paths/hashes. No live qualification jobs; do not duplicate completed work.

Next: total per-device memory admission and safe splitting,actual
consumer FP16,representative five-preset screening,full-size F6. No new formal
full-size throughput result. ContentBudget accounts module reservations,not total
per-device peaks. Include parameters,state/KV,tapes,master/optimizer/proposals,
communication,head/loss and construction transients plus headroom. Existing
NPUCachingAllocator statistics expose current/peak allocated/reserved bytes;
they do not include all vendor/driver HBM. Consumer head optimizer has FP32 masters,
but the actual consumer still rejects FP16. Do not duplicate experiment tracking.

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
