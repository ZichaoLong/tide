# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch graph-execution-foundation.

## Authorized objective and boundaries

User requests completion of all remaining work that can close on this machine:
independent NPU node-ranking/event-queue candidates, full training-step performance,
matched full-size model/configuration/multi-card comparisons, local version checks.
External CUDA GPU/x86 hardware execution stays pending. No subagents/pushes;
reference repositories are read-only. Public single-device core remains unchanged.
Use public /opt CANN/SDK modules, account zlong; never alter the driver.

Historical workload: 465 nodes,2208 logical/4418 physical edges,D2048/B512/V50304,
Attention17,269,426,339 and Add9,468,020,899 parameters (historical binary8.8B label).
FP32 payload,CPU seed7/original owner order,2 body ticks/token,clear,all-softmax,
4 heads. Inference/grad-forward:12 growing-context tokens,4 warmup/8 measured.
Do not shrink dimensions or silently change precision to hide capacity failures.

## Current implementation and development results

Implementation b4f26b3; latest prior evidence005280d; new dispatch/training evidence records200 clean configuration cells plus4 analytic runs. New committed
consumer changes add --ranking-device cpu|model and --event-device cpu|model.
Exact stable lexicographic tensor ranking is batched by candidate width. Selected
and affected counts remain host int64 history maps. The independent tensor queue
computes time readiness/order on the first shard; host owns payload handles and
C++ kernel dispatch. Explicit metadata transfers are counted separately.
Int64 sort on this CANN stack uses on-device AiCPU, not host CPU or AiCore;
retained raw warning/probe: runs/dispatch-ops-probe-d0. No approximate float keys.

--training-steps N,--training-warmup N,--optimizer adamw|sgd,--learning-rate X:
full sequence window from empty graph state per update, retained history within
window, mean token CE targets=(input_id+1)%vocab,HARD signaling,backward,optimizer.
Requires grad1/token warmup0. AdamW beta.9/.999,eps1e-5;SGD momentum.9;decay.01.
No checkpoint import or convergence claim. Contract: accelerator-scale.md.

Frozen dispatch-dev01 (005280d+recorded dirty patch) remains read-only.
- build-cpu-dispatch-dev01/build-npu-dispatch-dev01 PASSED all3 CTests.
- check-cpu-dispatch-dev01/check-npu2-dispatch-dev01 PASSED analytic ranking
  ties/int64, queue time/parallel edges/payload VJP and12 cells each (three
  non-default dispatch combinations xAdd/Attention xmemory/locality; grad/no_grad).
Frozen training-dev02 adds training and near-ULP tie test; never edit it.
- build-cpu-training-dev02/build-npu-training-dev02 PASSED all3 CTests.
- train-{cpu,npu}-r{c,m}-e{c,m}-dev02: ALL8 jobs PASSED,4 cells/job. Each compares
  full values/exact routes/state/history/pending,isolated VJPs,three complete
  training windows,all gradients/None,updated weights and optimizer slots against
  independent CPU scalar-slot schedule. NPU2 sets were1,9/5,11/2,8 and queued reuse.
- Six CLI preflight rejections passed; raw training-cli-rejections-dev02/results.json.
New verifier CPU-Read/dtype flags have been added after dev02 for qualification.
No core source/Python-package code changed; do not repeat unchanged-core8636 tests.

Clean perf-a4 is frozen at b4f26b3. build-cpu-a4/build-npu-a4 PASSED all3 CTests.
qual-cpu-a4(full) and qual-npu4-a4(multicard) PASSED all stages. qual-npu2-a4(full)
PASSED all stages; profile-dispatch-a4 PASSED, raw operator trace
retained. qual-npu8-a4(multicard) PASSED on1,2,5,6,7,8,9,11; each has a background.slice unit. Run immutable gates on CPU and2/4/8
NPUs,real465-node small-tensor gates and actual operator trace. Formal performance
must use the clean qualified source. Native timeouts/RSS/queue waits stay bounded.

## Full-size baseline and next experiment design

Original a3 FP32 NPU Read/control two-card pilots: Add no-grad41.6797,
Attention no-grad100.0539,Add grad-forward64.5930 ms/sample-token passed12/8.
Attention2 grad-forward FAILED during eighth token:allocated58.66GiB,reserved61.05,
free1.71MiB; retained OOM. Four-card follow-up pilot-attention-g1-n4-a3 PASSED12/8,
mean136.0246,peak allocation37.535GiB on most-loaded card; preflight also passed.
Add2 with CPU FP32 controls,NPU Read PASSED12/8,mean27.9618. Records validated,
exit0/no children for successful runs. These are concurrent exploratory pilots;
no causal speedup claim. Evidence: accelerator-scoring-pilots-20260928.{md,json}.

Next bounded screen: screen-add-a4 (two devices) and screen-attention-a4 (four),
seven declared scoring/dispatch choices each, concurrent exploration,12 tokens/4
warmup. Plans in task-root/plans; run-plan-a4.py preserves per-cell failures.
Then capacity, placement/scaling and isolated matched finalist repetitions. Compare CPU FP64,CPU FP32,mixed Read/control and independently
selected tensor ranking/queue; compare2/4/8 cards and memory/locality placement.
Use three fresh-process repetitions for final matched implementation choices.
Complete training uses full12-token windows; include backward/optimizer and exact
window/optimizer/warmup definition. Check capacity before a larger comparison.
Independent concurrent runs and controlled matched timing are separate experiments.
Keep all OOM/timeouts and source identities. Never rerun an unchanged capacity failure.

## SDK2.9/CANN8.5 work (public isolated candidate)

Candidate /opt/software/libtorch-npu/2.9.0-cann8.5.0-aarch64-abi1.
Pinned TorchNPU12d689a08941d4a6e45eab16e3ad6fef96a9affd matches installed public
TorchNPU2.9/Torch2.9; upstream compatibility row7.3.0/CANN8.5.0 retrieved/hashed.
fetch-sdk29-direct-l1 FAILED only fetching compatibility table (TLS timeout);
Git source/submodules completed. One proxy retry succeeded; source.json records it.
Do not relabel the failed fetch. Sources,private torchgen and build-env are isolated.
build-sdk29-l1 PASSED. Standalone smoke/build closure: First linker attempt failed on a wheel-bundled libgfortran search
path; build-sdk29-smoke-l2 PASSED with matched torch.libs in LD_LIBRARY_PATH.
smoke-sdk29-l2 PASSED real-device forward/backward/SGD/checkpoint on12.
Public libtorch-npu/2.9.0-cann8.5.0 published; load/unload/conflict checks passed.
smoke-sdk29-c851-l2/c852-l2 PASSED. build-tide-sdk29-a4 FAILED because SDK2.9 omitted source-path ACL/HCCL headers.
Exact-source46-header supplement recorded in package/header-supplement.json.
build-tide-sdk29-a4b resumes the matching standalone core/consumer build with
two workers and3600s bound.8.5.1/.2 smoke PASSED; public versioned modules published.
All module loaders contain no Python/stub/unresolved libraries; prior failures retained.
Do not edit its recipes/source while active. Inspect task.log/status.json before
SDK loader/smoke/public-module publication and local8.5.x qualification. Existing
2.10/CANN9 SDK/public environments untouched. No new SDK support claim yet.

## Operations and reentry commands

Task root /mi/data2T/zlong/tide-npu-performance; queue eligible pool twelve chips
1,2,5,6,7,8,9,11,12,13,14,15. A candidate pool is not a free-device allocation.
Last inventory had7 free chips;8-device work may need to wait. Do not edit locks,
priorities or other workloads. Existing freeze_run.py creates frozen worktrees,
background.slice units and artifact links. Each run owns status/queue/task.log.
Unit pattern tide-npu-performance-NAME.service; stop only a named own task with
systemctl --user stop UNIT. Build roots under task-root/builds; raw runs under runs.

Reentry: git status --short --branch;python scripts/status.py;inspect live job JSON
and bounded log tails. Formal runs: run/{run.json,summary.json,metrics.jsonl,stdout.log,
lifecycle.json}; do not print all RSS samples. Validate terminal records with
python /home/zlong/.agents/skills/run-ml-experiments/scripts/validate_run_record.py RUN_DIR.
Trackio best-effort local project tide-npu-performance,task-root/trackio,storage auto,
viewer /home/zlong/venvs/trackio/bin/python. Raw records authoritative.
TASK_QUEUE_ENABLE=0 is explicit; default async cross-device event waits crashed.
SDK finalizes after tensors/workers; repeated-finalize warning needs normal exit.
Immutable historical gates: accelerator-scale-20260928 and accelerator-scoring-20260928.
Core gate evidence: accelerators-20260928 (CPU8636+22,328 NPU cells/four CANN stacks).
