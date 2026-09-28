# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch graph-execution-foundation.

## Authorized objective and boundaries

Finish locally feasible remaining work: independent NPU ranking/event candidates,
complete training performance, matched full-size comparisons, memory/locality and
2/4/8-device measurements, and local CANN qualification. External CUDA/x86 device
execution stays pending. No subagents/pushes. Reference repos are read-only.
Public single-device core/Python package unchanged; benchmark consumer is separate.
Use public /opt CANN/SDK modules as zlong; never alter the shared driver.

Historical full-size workload:465 nodes,2208 logical/4418 physical edges,
D2048/B512/V50304,Attention17,269,426,339 andAdd9,468,020,899 parameters (old
binary-unit8.8B label). FP32 payload,CPU seed7/original owner order,2 body ticks,
clear,all-softmax,4heads. Inference/grad-forward:12 tokens,4 warmup/8 measured.
No dimension/precision/window reduction to hide capacity failure.

## Completed implementation and immutable qualification

Implementation b4f26b3; frozen task-root/sources/perf-a4. Evidence238c2c3 records
200 configuration cells +4 analytic runs: CPU64,two-NPU64,four/eight-NPU36 each.
All PASSED, including real465-node D8/B1 Add/Attention forward/full training,
strict tiny isolated VJPs and explicit basis-conditioned wide VJPs. CPU/NPU
builds passed3 CTests/loader closure each. Do not rerun unchanged-core8636 CPU
regression or prior328 broad NPU gates. Historical development/failures retained.

New --ranking-device cpu|model and --event-device cpu|model are independent.
Exact stable int64 ranking/queue keys; histories and tensor handles still host
owned, host invokes C++ kernels. Not a completely device-resident control loop.
Model FP32 Read/controls are independently configurable; default CPU FP64 remains.
Explicit metadata transfer counters separate from payload counters.

--training-steps N,--training-warmup N,--optimizer adamw|sgd,--learning-rate X:
complete sequence window from empty graph state per update, full history within
window, mean token CE targets=(input_id+1)%vocab,HARD signaling,backward,optimizer.
Requires grad1/token warmup0. AdamW beta.9/.999,eps1e-5,decay.01;SGD momentum.9.
Not continuous-stream training or convergence evidence. See accelerator-scale.md.

## Additional standalone SDK2.9 locally CLOSED

Package /opt/software/libtorch-npu/2.9.0-cann8.5.0-aarch64-abi1, official source
12d689a08941d4a6e45eab16e3ad6fef96a9affd,Torch2.9.0+cpu,aarch64,ABI1.
Public modules libtorch-npu/2.9.0-cann8.5.0,8.5.1,8.5.2 (full prefix for each).
Same SDK/Torch build, three CANN/ATB runtime tests; no claim of3 separate builds.
All three smoke, qual-sdk29-c850-a4/c851-a4/c852-a4 and corresponding profile
jobs PASSED. Per runtime: core ring/diamond values/gradients/three AdamW updates,
checkpoint toNPU/CPU,installed CMake consumer,64 two-NPU consumer cells+analytic.
SDK/client loader closure and3 CTests passed. Public0644 modulefiles,SDK read/
traverse permissions and load/unload/conflicts checked. Existing2.10 untouched.

Retained failures: fetch-sdk29-direct-l1 (final compatibility fetch timeout,
one recorded proxy retry succeeded); build-sdk29-smoke-l1 (hashed wheel Fortran
link dependency; fixed matched torch.libs search path); build-tide-sdk29-a4
(official SDK export header gaps;46 exact-source ACL/HCCL/HCCLUtils headers added).
Successful follow-ups: build-sdk29-smoke-l2,smoke-sdk29-l2,build-tide-sdk29-a4b.
Package README,recipes,hashes,supplement manifest retain reproducibility.
Evidence b5166eb: docs/evidence/standalone-sdk29-20260928.{md,json}.

All four CANN stacks have actual operator traces: int64 Sort on card-local AiCPU,
FP32 norms/reductions on MIX_AIV,Softmax on AI_VECTOR_CORE.8.5 traces each6946
operators;9.0 two-device trace7012. Tiny placement profiles, not speed claims.

## Active performance work and exact continuation

Task root /mi/data2T/zlong/tide-npu-performance (all paths below relative there).
All current experiments use clean b4f26b3,builds/client-npu-a4,CANN9/Torch2.10,
TASK_QUEUE_ENABLE=0,A3 driver25.3.rc1. Native bounds and host RSS budgets explicit.

- screen-add-a4 PASSED seven two-card cases on5,6;12/8 observations each.
  CPU64=17.110,CPU32=18.419,mixed32=28.282,score32=39.415,rank32=35.383,
  queue32=40.650,all32=34.933 ms/sample-token. Exploratory concurrent observations.
- screen-attention-a4 PASSED on1,2,8,9: CPU64=56.671,CPU32=49.512,
  mixed32=63.671,score32=59.956,rank32=60.560,queue32=71.442,all32=67.420.
  All12/8. Both fastest FP32 screens are cpu32. plans/matched-inference-a4.json
  prepared (NOT submitted):12 runs,6-chip allocation,Add uses indices0,1;
  Attention uses2,3,4,5;3 repeats per CPU64/CPU32,interleavedAB/BA/AB.
- train-screen-add-a4 PASSED all7 on5,6,7,11.
  Seven same scoring/dispatch choices; full12-token forward/backward/AdamW,
  one cold complete update (training_steps1,warmup0). CPU64/cpu32/mixed32/score32/rank32 cold updates PASSED
  42.742/41.679/48.622/57.142/53.073 ms/sample-token;queue32/all32 PASSED56.014/57.273. CPU64 phase times202.553/57.514/2.540s,
  total262.607s,max peak38.177GiB. CPU32 total256.073s,peak38.177GiB. Per cell2400s/RSS256GiB. Stop plan on first
  failure, then inspect before another attempt. Cold screen is NOT warm throughput.
- placement-scale4-a4 PASSED on1,2,8,9: Add memory/locality4,
  Attention memory4/locality2. Fixed CPU FP32 scoring/CPU dispatch;12/4 tokens,
 1800s/RSS256GiB each. Add memory/locality4 PASSED14.442/16.609;
 Attention memory4 PASSED43.533;Attention locality2 PASSED73.028. Descriptive only.

- train-screen-attention-a4 CANCELLED QUEUED WAITER: external SGLang processes
  newly occupy1,8,9,12; only2,5,6,7,11 free. Preserve cancellation, no benchmark
  cell started. Do not touch other workloads or queue priorities. Resubmit a NEW
  job identity when8 devices available; original seven-cell plan unchanged.
- train-warm-add-a4 RUNNING on eligible5,6,7,11; prepared
  plans/train-warm-add-a4.json via run-training-screen-a4.py. Three fresh CPU32
  processes,2 complete12-token AdamW updates,first warmup. Stop on first failure.
  First process cpu32-r1 running; allocation physical5,6,7,11 to logical0..3.
  timeout3600/RSS256GiB per cell. This smaller ready job should not sit behind
  an impossible8-card queue head.

Full continuation scope is fixed in plans/followup-policy-a4.md:
1. Finish above jobs; inspect failures and validate new terminal records.
2. Resubmit Attention cold screen with a NEW job name after8 devices are free.
   Previous queue-only cancellation preserves records; never reuse its output. Seven cold
   complete training cells, stop on first failure. Preserve dimensions/window.
3. Isolated inference finalists: historical CPU64 vs fastest completed FP32
   screen config for each model;3 fresh-process repeats each,AB/BA/AB, fixed
   physical allocation. No other heavy own job during those timings. Use new
   run-matched-plan-a4.py if subsets of a held allocation are needed (device_indices).
   Then one controlled concurrent Add2+Attention4 pair on disjoint matching
   subsets; report observed finite-workflow throughput/overlap, no causal gain
   claim from one pair. External workloads remain recorded limitations.
4. Finish fixed-CPU-FP32 no-grad scaling at8 devices for each model (two cases).
   Prepared plans/scaling8-a4.json; NOT submitted;run-plan-a4.py.
   Existing/planned2/4 cases above complete finite scaling/locality comparison.
5. Training finalist:3 fresh processes/model,2 full updates,first warmup;
   capacity-qualified placement/count,full12-token windows,phase/total timing.
6. Validate all terminal records/cleanup,review small evidence,update ROADMAP/
   contract/STATUS,commit evidence separately. No required own live jobs at end.

Never mechanically repeat an unchanged capacity failure. Historical a3 Attention2
model-FP32 Read/control grad-forward OOM at token7:allocated58.66GiB,reserved61.05,
free1.71MiB,2MiB request. Attention4 follow-up pilot-attention-g1-n4-a3 PASSED12/8,
mean136.0246,max peak37.535GiB. Add2 grad-forward PASSED64.5930. These are prior
concurrent pilots, not matched scaling/training evidence. Include the4-card
follow-up in the final report; older scoring-pilots report points here for it.

## Operations and records

Read-only progress: python launchers/progress-a4.py [JOB...]. Default list covers
screens/SDK; explicitly add training/placement jobs. Queue state is separate from
outer job state. Units tide-npu-performance-NAME.service in background.slice.
Inspect runs/NAME/{status.json,queue.json,task.log,stages.json}; benchmarks have
NAME/CASE/{run.json,summary.json,metrics.jsonl,stdout.log,lifecycle.json}.
Stop only a named own unit with systemctl --user stop; never edit queue locks,
priorities or other workloads. Eligible pool1,2,5,6,7,8,9,11,12,13,14,15 is not
an allocation. Recent free pool was9 chips; other users occupy remaining chips.

New/changed record validation: python launchers/validate-new-records-a4.py JOB...
Hashes cached in runs/record-validations-a4.json;25 terminal identities validated
so far,healthy Trackio,no remaining descendants. Do not print full RSS arrays.
Trackio0.35 best-effort local project tide-npu-performance,task-root/trackio,
viewer /home/zlong/venvs/trackio/bin/python,storage auto;raw records authoritative.
freeze_run.py owns immutable snapshots/transient services; never edit an input
of a running OR queued job. Long waits belong inside the queued service.
Reentry main repo: git status --short --branch;python scripts/status.py.

Prepared launcher run-concurrent-pair-a4.py (not submitted) accepts2-cell plan
with disjoint device_indices covering the full allocation. It records total
finite-workflow wall time including construction/recording, plus individual
usual benchmark records; do not compare that combined metric directly with
warm per-token latency. New task launchers use only portable benchmark CLI.

Current reviewed evidence checkpoint: docs/evidence/accelerator-performance-20260928.{md,json}
contains25 completed full-size cells plus the historical Attention4 grad-forward
follow-up. ROADMAP,accelerator-scale and older scoring-pilots navigation updated.
Documentation-only checkpoint; active source remains b4f26b3.
Do not claim overall performance completion while required jobs remain.
