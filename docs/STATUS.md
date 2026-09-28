# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch graph-execution-foundation.

## Authorized scope

Portable GPU/NPU foundation; full-size LibTorch/NPU historical17B Attention and
"8.8B" Add performance: independent concurrent processes AND one model spanning
2/4/8 cards, with locality-aware placement. Up to12 available chips total.
No subagents/pushes/reference-repo edits. Public core remains unchanged.

Latest accepted extension: configurable CPU/model-device Read and softmax,
FP64/FP32, independent correctness comparisons. NPU node ranking/event scheduling
are separate future candidates; validate independently before choosing based on
inference AND training performance. Current benchmark is no_grad or grad-forward,
without backward/optimizer/history detach; it does not measure a training step.

## Fixed historical workload

465 nodes,2208 logical/4418 physical edges,D2048/B512/V50304. Attention exactly
17,269,426,339 parameters; Add9,468,020,899 (historical8.818*1024^3). FP32 payload,
CPU seed7/original owner-RNG order,2 body ticks/token,clear,all-softmax,4 heads.
Twelve growing-context tokens,4 warmup+8 measured. Default CPU norm-fp64-v1 Read.
Never silently shrink topology/dimensions/precision. Contract: accelerator-scale.md.

## Code and gates

Implementation commit a12ee0c; scoring-qualification handoff2188190; baseline bb0ecc6.
Frozen perf-a2/client-{cpu,npu}-a2 passed immutable CPU/2/8-NPU tiny gates and
real465-node D8/B1 gates. Evidence: evidence/accelerator-scale-20260928.{md,json}.
Default strict VJP remains rtol1e-5/atol1e-6. Wide Add Full squared-norm gradients
suffer cancellation reproduced in one node; explicit basis-conditioned checks
complete Jacobians at original tolerance and FP64 contractions. Prior strict
failures remain failed. Core baseline1b0cb48 was qualified: CPU8636+22 complex
cells,328 positive NPU gates/four CSR rejections/four CANN stacks. Do not repeat
this unchanged-core qualification. No CUDA hardware performance claim.

Current implementation increment in tools/accelerator_scale and two Python wrappers:
--read-device cpu|model,--read-dtype float64|float32,--control-device cpu|model.
Default cpu/float64/cpu unchanged. CPU FP32 reference and NPU FP32 custom Read
profile scale-norm-fp32-v1; precision appears in graph identity and run manifests.
model means node shard for Read,fixed lowest-region-member shard for softmax.
Sorting,integer histories and event scheduling remain CPU. Gradients preserve
None-vs-zero. --reference-read-dtype float64 explicitly compares precision; exact
routes and all other observables stay checked. FP64 conversion is CPU-only.
Unsupported NPU FP64 or model scoring with host transport fail explicitly.

All acquired NPU contexts now initialize BEFORE lengthy CPU weight construction.
This makes reservations visible to process-based resource checks, but advisory
locks still cannot exclude non-cooperating jobs.

Development snapshots scoring-dev10/scoring-dev11 are frozen; never edit them.
build-{cpu,npu}-dev11 PASSED including analytic scoring/zero VJP and rejection
CTests. Dev10 tiny gates passed: check-cpu-default-dev10/check-cpu-fp32-dev10
8 cells each,check-npu2-default-dev10 8,check-npu2-fp32-dev10 4 (strict).
scoring-profile-dev10 PASSED: msprof observed41 FP32 LpNormV2/MIX_AIV tasks
and349 SoftmaxV2/AI_VECTOR_CORE tasks on both chips; not a performance result.
Dev09 missing-header compile failure retained. scoring-extra-dev10 exposed a
comparison-only NPU FP64 cast bug; fixed in dev11 by host copy before conversion.
No timed math changed. Preserve old failure.

scoring-extra-dev11 PASSED on physical9,11: both mixed read/control placements,
including explicit CPU FP64 comparison,and four465-node Add/Attention seed0/7
D8/B1 cells with basis-conditioned policy. Every cell has normal exit,complete
observables and isolated VJPs. Raw results: task-root/runs/scoring-extra-dev11.
Clean perf-a3 frozen at a12ee0c. Immutable qualification ALL PASSED:
build-cpu-a3/build-npu-a3,check-cpu-default-a3/check-cpu-fp32-a3 (8 cells each),
check-npu2-default-a3 (8),check-npu2-fp32-a3 (4),scoring-extra-a3 (6),
scoring-profile-a3. Total34 parity cells; both grad/no_grad,complete observables,
isolated VJPs. Trace confirms41 FP32 LpNormV2/MIX_AIV and349 SoftmaxV2 vector
kernels on both chips (softmax includes model operations). Reviewed evidence:
evidence/accelerator-scoring-20260928.{md,json},separate from implementation.
NPU8 new scoring mode,full backward/optimizer timing,and CUDA remain unverified.

## Full-size FP32 Read/control results (inspected 20:09 CST)

All four a3 pilots have ended. Source perf-a3/a12ee0c,client-npu-a3,
NPU model-device FP32 Read/controls,locality/resident,D2048/B512/V50304,
seed7,12 requested tokens,4 warmup,1800s native and512GiB RSS cap.
Exact commands/prerequisites: task-root/runs/scoring-pilots-a3.json.

- pilot-add-g0-n2-a3 PASSED,physical9,11,mean41.6797ms/sample-token,
  max per-device peak allocation19.874GiB;12 tokens/8 measured,exit0.
- pilot-attention-g0-n2-a3 PASSED,physical1,5,mean100.0539ms/sample-token,
  max per-device peak allocation42.653GiB;12 tokens/8 measured,exit0.
- pilot-add-g1-n2-a3 PASSED,physical2,8,mean64.5930ms/sample-token,
  max per-device peak allocation27.252GiB;12 tokens/8 measured,exit0.
- pilot-attention-g1-n2-a3 FAILED (OOM),physical9,11,completed7 tokens
  (indices0..6),failed during token7. Native log reports logical0 allocated
  58.66GiB,reserved61.05GiB,free1.71MiB;failed2MiB allocation. This is a
  capacity limit for the current placement/no-detach grad-forward window,
  not a correctness failure or a completed latency result.

All four records validated,tracking healthy,no remaining child processes.
Three successful tests are descriptive concurrent pilots,not controlled speedup
comparisons. Timing overlap exists beyond construction: Add no_grad/Attention
no_grad/Add grad-forward produced tokens during overlapping time spans. Token
end timestamps/raw durations are in each run/metrics.jsonl. Explicit vector
D2H counter is zero; CPU ranking/scalar extraction remains and is not in that
counter. Previous CPU FP64 results differ in devices and/or contention.

## Active capacity/control follow-ups (inspected 20:17 CST)

Both jobs use frozen perf-a3/a12ee0c and client-npu-a3. Exact submission
commands, budgets and launcher hash: task-root/runs/capacity-control-followup-a3.json.
Both full-size stages retain1800s native/512GiB RSS/12 tokens/4 warmup.
Unit: tide-npu-performance-NAME.service. These are new capacity/control cases,
not unchanged reruns of the failed two-device Attention placement.

- pilot-attention-g1-n4-a3: RUNNING, physical1,5,9,11, run run-f49bcacf.
  Stage states: tiny-gates=passed, wide-attention-gate=passed, full-size=running.
  Model construction176.396s; token indices0..2 completed at inspection.
  Four tiny resident cells and465-node D8/B1 Attention gate passed before
  full-size execution; complete observables/isolated VJPs, explicit basis-conditioned wide gate.
- pilot-add-g0-n2-cpuctrl-a3: PASSED, physical2,8, run run-823df71d.
  NPU FP32 Read with CPU FP32 controls; no-grad Add on two devices.
  Native exit0, 12 tokens/8 measured; tracking=healthy, record validated, no children.
  Mean27.9618ms/sample-token; exploratory, not a matched speedup comparison.

Inspect: python scripts/status.py; then read task-root/runs/NAME/status.json,
queue.json and run/{summary.json,stdout.log,metrics.jsonl}; for Attention also
stages.json and {tiny-gates,wide-attention-gate,full-size}.log. Validate a terminal
record with python /home/zlong/.agents/skills/run-ml-experiments/scripts/validate_run_record.py
task-root/runs/NAME/run. Never infer passage from unit inactivity.
Trackio: project tide-npu-performance, best-effort local, task-root/trackio,
storage auto; viewer interpreter /home/zlong/venvs/trackio/bin/python.
Next: inspect Attention terminal records; preserve failure; then matched2/4/8-card
placement/precision repetitions, complete backward/optimizer measurement, and
independent NPU ranking/scheduling. CUDA hardware validation stays pending.
Completed original a3 pilots: evidence/accelerator-scoring-pilots-20260928.{md,json}.

## Full-size results and concurrency correction

Task root /mi/data2T/zlong/tide-npu-performance. Each job has frozen source,unique
runs/NAME/{status.json,queue.json,task.log,run/*}; artifact links in repository.
launchers/freeze_run.py owns stable source and background.slice unit launch.
Unit pattern tide-npu-performance-NAME.service. Inspect run.json/summary.json,
metrics.jsonl/stdout.log/lifecycle.json; never infer a pass from a dead unit.

perf-a2 pilots,locality/resident,1800s native/512GiB RSS/7500s queue limit:
- pilot-add-g0-n4-a2b PASSED,physical1,5,9,11,mean16.2893ms/sample-token.
- pilot-attention-g0-n4-a2b PASSED,same4,mean56.0128ms/sample-token.
- pilot-add-g1-n2-a2c PASSED,physical9,11,mean48.0210ms/sample-token.
All12 tokens/8 measured,exit0,no children,records validated. Single pilots only;
Add2 grad-forward overlapped other construction. Do not compare it directly with
four-card no_grad numbers or claim matched/repeated speedups.
- pilot-add-g1-n4-a2c FAILED during placement,logical3 OOM,own allocation7.69GiB.
  Queue devices1,5,8,12; cannot interpret as a clean four-card capacity failure.
- pilot-attention-g1-n4-a2c CANCELLED during construction after other processes
  entered allocated physical2,8 (its pool1,2,5,8). Only our process was stopped.
Evidence: runs/parallel-pilots-a2c-{results,device-contention}.json.
Original fixed-four queued a2b grad pilots and earlier a2 pilots cancelled;
all records retained. Terminal a3 full-size results are recorded above.

The user correctly identified avoidable serialization. New pool is twelve
eligible devices1,2,5,6,7,8,9,11,12,13,14,15,not twelve allocated chips.
FIFO helper admits healthy,no-process,low-HBM devices using two snapshots and
locks; no priority changes/backfill/live queue edits. Actual Add2/Add4 lifetimes
overlapped,but Add4 never reached timing. Init-all-device fix addresses the window
where uninitialized allocated cards appeared free during CPU construction.
After qualification,submit independent cases separately and confirm actual overlap.
Compare2/4/8 cards,memory/locality,and three fresh-process repeats where feasible;
concurrent contention and isolated timing are separate. Preserve capacity failures.

## Environment and inspection

Use user-selected public /opt modules,not older personal guide defaults.
libtorch-npu/2.10.0-cann9.0.0,TorchNPU2.10/CANN9.0/driver25.3.rc1,A3
Ascend910_9392,16 chips×64GiB. Driver unchanged. Existing installed core builds:
/mi/data2T/zlong/tide-accelerator/builds/native-{cpu,npu-sdk}-dev03.
TASK_QUEUE_ENABLE=0 is explicit/recorded; async queue cross-card wait crashed.
Early SDK finalization after tensor/worker destruction fixes observed exit crash;
repeated-finalize warning is benign only with exit0. CSR pooling unrelated.

Trackio wrapper/viewer /home/zlong/venvs/trackio/bin/python,0.35.0,project
 tide-npu-performance,local task-root/trackio,storage auto; no dashboard exposed.
Raw records authoritative. Stop one task with systemctl --user stop
 tide-npu-performance-NAME.service. Implementation/STATUS/ROADMAP/accelerator-scale docs committed together;
evidence must be a separate commit after immutable qualification. No core/Python package edits. Use atomic handoff writes and git diff
before committing; active jobs read frozen snapshots only.
