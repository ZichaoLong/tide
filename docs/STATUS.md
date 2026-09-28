# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch graph-execution-foundation.

## Objective and boundaries

Finish locally feasible remaining work: full-size inference/training selection,
matched repeats, controlled concurrency,2/4/8-device placement/scaling, and local
CANN qualification. CUDA/x86 device execution remains external. No subagents or
pushes; reference repos read-only. Public core/Python package unchanged during
this performance extension; tools/accelerator_scale is a standalone consumer.
Use public /opt CANN/SDK modules; never change the shared driver/other workloads.

## Completed immutable implementation and qualification

Implementation b4f26b3, task-root/sources/perf-a4. Evidence238c2c3 records200
configuration cells+4 analytic runs: CPU64,two-NPU64,four/eight-NPU36 each. All
PASSED, including real465-node D8/B1 Add/Attention forward/full training, tiny
strict isolated VJPs and explicitly basis-conditioned wide VJPs. CPU/NPU builds
passed3 CTests and loader closure each. Do not repeat unchanged-core8636 CPU
regressions or prior328 broad NPU gates.

Read CPU/model and FP64/FP32, controls CPU/model, ranking CPU/model and events
CPU/model are independently selectable. Stable exact int64 dispatch. Histories,
tensor handles and C++ call dispatch remain host-owned; not a fully device-resident
control loop. Operator traces on all four CANN stacks place int64 sorting on
card-local AiCPU, FP32 reductions on MIX_AIV and softmax on AI_VECTOR_CORE.

Complete training: --training-steps/--training-warmup, --optimizer adamw|sgd and
--learning-rate. Empty graph state per update, full autograd/KV history throughout
its12-token window, synthetic mean CE targets=(input_id+1)%vocab,HARD signaling.
AdamW beta.9/.999,eps1e-5,decay.01;SGD momentum.9. Not continuous-stream training
or convergence evidence. Defaults remain CPU FP64 Read and CPU control/dispatch.

SDK2.9/CANN8.5 work CLOSED in b5166eb. Official source
12d689a08941d4a6e45eab16e3ad6fef96a9affd, package
/opt/software/libtorch-npu/2.9.0-cann8.5.0-aarch64-abi1. Public modules
libtorch-npu/2.9.0-cann8.5.0,libtorch-npu/2.9.0-cann8.5.1 and
libtorch-npu/2.9.0-cann8.5.2 share one SDK/Torch2.9 build. All runtime gates,
smokes, traces, installed consumers,64 two-NPU cells+analytic per runtime passed.
Existing2.10 untouched. Exact46-header source supplement, failed direct fetch,
Fortran link failure and initial SDK include-closure failure are retained in
standalone-sdk29-20260928 evidence; do not relabel retained failures. Public
modulefiles/README0644,SDK traversable,load/unload/conflicts checked.

## Completed performance checkpoint

Task root /mi/data2T/zlong/tide-npu-performance. Paths below relative there.
All a4 experiments: clean b4f26b3,builds/client-npu-a4,public
libtorch-npu/2.10.0-cann9.0.0,TASK_QUEUE_ENABLE=0,A3,driver25.3.rc1.
Full historical465 nodes/2208 logical/4418 physical edges,D2048/B512/V50304,
Attention17,269,426,339 andAdd9,468,020,899 parameters (old binary8.8B label),
FP32 payload,CPU seed7/original initialization order,2 body ticks,clear,all-softmax,
4heads. No reduction of dimensions,precision or history to hide failures.
Inference12 tokens,4 warmup/8 measured; training12 tokens per full update.

Evidence f7e7c95: docs/evidence/accelerator-performance-20260928.{md,json}.
25 full-size terminal cells PASSED and validated: screen-add-a4 seven Add2,
screen-attention-a4 seven Attention4,train-screen-add-a4 seven cold Add4 complete
updates,placement-scale4-a4 Add memory/locality4 andAttention memory4/locality2.
Each screen's fastest FP32 candidate is cpu32. Cold Add cpu32=41.679ms/sample-token;
all seven candidates passed, but no warmed-throughput or causal speedup claim.
Locality Add4 reduces remote512.504 to169.346MiB/token yet observed latency
14.442(memory) vs16.609(locality); single concurrent descriptive observations.
All completed raw records have healthy Trackio and no remaining children.

Retain prior Attention2 model-FP32 Read/control grad-forward OOM on token7:
58.66GiB allocated,61.05GiB reserved,1.71MiB free,2MiB request. Four-card follow-up
pilot-attention-g1-n4-a3 PASSED12/8,136.025ms/sample-token,peak37.535GiB. Its source
is a12ee0c,grad-forward only; it is now in the performance evidence above, not an
unresolved link to STATUS. Never repeat unchanged OOMs mechanically.

## Live work and resource adaptation

- train-warm-add-a4 PASSED on physical5,6,7,11 =>logical0..3.
  Unit tide-npu-performance-train-warm-add-a4.service,background.slice.
  Plan plans/train-warm-add-a4.json via launchers/run-training-screen-a4.py.
  Three fresh cpu32 processes,2 full12-token AdamW updates,first warmup;
  timeout3600/RSS256GiB each,stop on first failure. cpu32-r1 PASSED: measured
  40.404ms/sample-token,248.242s,forward192.732/backward54.313/optimizer1.189s;
  peak41.058GiB after warmup vs38.177GiB first update.1356 grad/slot owners,
  second loss9.74259663. cpu32-r2 completed its measured update43.123ms/sample-token;
  cpu32-r2/r3 PASSED43.123/42.080ms/sample-token;all3 validated.
  comparison.json generated by compare-repeats-a4.py;no remaining children.
- train-screen-attention-a4 CANCELLED before any cell started,exit143. Other
  workload newly occupied1,8,9,12, leaving2,5,6,7,11 free. Eight-chip queue head
  could not fit; stopped only that own waiter to run ready Add4 repeats. Preserve
  status/logs/cancellation-note.json. Queue.json still reflects last queued
  observation; outer terminal status is authoritative. No priorities changed.
  Replacement train-screen-attention-a4b RUNNING on physical1,2,5,6,7,8,9,11
  =>logical0..7. CPU64 first cell running. Unit inbackground.slice verified.
  Same unchanged plan
  plans/train-screen-attention-a4.json. Seven full cold12-token AdamW candidates,
  2400s/RSS256GiB each,stop at first failure and inspect. Never reuse old output.

Async question asks whether the new four-chip SGLang workload has an expected
release time. No answer yet. Continue useful work without stopping other jobs.
The latest later snapshot showed1,8,9,12 released:9 eligible chips including
our4 allocated chips. Recheck before launch; low utilization alone is not freedom.

## Exact remaining work

Bounded policy: plans/followup-policy-a4.md (includes resource adaptation).
1. Monitor train-screen-attention-a4b andvalidate every new terminal cell.
   Add warmed3-run assessment is PASSED. Do not overlap matched inference with
   another heavy own job.
2. Run matched inference only when no other heavy own job is running. Original
   plans/matched-inference-a4.json holds6 chips: Add indices0,1;Attention2..5.
   If fewer than6 remain free, prepared plans/matched-inference4-a4.json uses
   one fixed4-chip pool: Add0,1;Attention0..3,strictly sequential. Same12 cells,
   CPU64 vs CPU32,3 fresh-process repeats each/model,AB/BA/AB; each model's
   physical subset stays fixed. Both plans NOT submitted. Launcher
   run-matched-plan-a4.py. Prefer candidate pool5,6,7,11 for four-chip variant.
3. Attention8 cold full-training screen as above. Do not block ready smaller
   work with an impossible8-device queue head. Candidate pool
   1,2,5,6,7,8,9,11,12,13,14,15 is not an allocation.
4. Eight-device no-grad scaling: prepared plans/scaling8-a4.json,NOT submitted,
   run-plan-a4.py. Fixed cpu32/locality,one Add andone Attention12/4 cell.
5. Attention training finalist:3 fresh processes,2 full updates each,first
   warmup; select from actual cold screen; capacity-qualified placement/count.
   Memory placement is a declared follow-up if locality fails. Do not shrink
   shape/window or blindly repeat OOMs.
6. One concurrent Add2+Attention4 pair with selected configs and disjoint chips.
   run-concurrent-pair-a4.py prepared,NOT submitted; plan not yet created.
   Cells need FULL config dictionaries (no defaults merge). device_indices cover
   allocation without overlap; optional physical_ids remaps exact desired groups.
   If pool differs from matched experiment, run one isolated selected-config
   reference/model on the same pair subsets before concurrent pair (bounded
   resource adaptation). Pair reports finite-workflow total including construction,
   all12 tokens,teardown and recording; not comparable directly with warm latency,
   and a single pair does not establish causal speedup.
7. Validate every terminal record/cleanup,update performance report,ROADMAP,
   contract/STATUS as appropriate,commit evidence separately. No required own live
   jobs at final. External CUDA/x86 verification remains explicitly pending.

## Operations

Launch from main repo using launchers/freeze_run.py --name NEW --snapshot perf-a4
--commit b4f26b3 --module libtorch-npu/2.10.0-cann9.0.0 --npu --npu-count N
--candidate-ids POOL --max-wait 7200 -- '{python}' '{base}/launchers/SCRIPT.py'
'{out}' '{base}/plans/PLAN.json'. Verify unit and actual queue/device state.
Never edit inputs of active OR queued jobs. Stop only exact own units.

Inspect: python TASK_ROOT/launchers/progress-a4.py JOB... (explicit names).
Validate new terminal records: python TASK_ROOT/launchers/validate-new-records-a4.py JOB...
Cached ledger runs/record-validations-a4.json currently28 identities. Extract
current a4 full-size results with launchers/summarize-results-a4.py JOB... --output FILE;
script asserts source/full size/complete counts/cleanup and preserves statistics.
Live native events are CASE/native/metrics.jsonl; finalized portable events CASE/metrics.jsonl.
Outer records runs/JOB/{status.json,queue.json,task.log,stages.json};cell records
CASE/{run.json,summary.json,metrics.jsonl,stdout.log,lifecycle.json,native/placement.json}.
Do not print full RSS arrays. Trackio0.35,best-effort local project tide-npu-performance,
root TASK_ROOT/trackio,storage auto;viewer /home/zlong/venvs/trackio/bin/python.
Raw records authoritative,no dashboard exposure needed. Disk recently88GiB free.

Current documentation checkpoint adds three completed warmed Add runs (28 total
cells) to f7e7c95 evidence. Active Attention work remains on immutable b4f26b3. No source change or new test required for
remaining experiment-only work. Reentry: git status --short --branch;python scripts/status.py.
