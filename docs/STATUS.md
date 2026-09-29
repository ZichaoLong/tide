# Current handoff

Updated: 2026-09-29. Branch graph-execution-foundation; no push authorized.
User authorized public Python/native API and standalone consumer FP16, full
correctness gates, profiling and bounded full-size inference/training pairs.
No sub-agents; reference repositories and ObsidianVault remain read-only.
The user explicitly chose to wait for seven/eight chips, not try six.

## Implemented and qualified

fc2a76f adds explicit FP16 payload and FP32 masters/slots/loss with static scale;
cb58110 fixes profiler stream ordering;595dccd strengthens matched-dtype consumer
VJPs and master trajectories. c8d2b61 rejects CPU Half CSR early;4a7dec7 records
that expected rejection in the named suite. Core C++ hashes are unchanged after
fc2a76f, and matching core builds are reused by the newer consumer.

All prescribed correctness gates passed:8645 CPU tests;39 Python and43 native
NPU FP16 cases plus explicit CSR rejection;five complete FP32 representatives
per implementation;108 consumer cells on CPU and2/4/7/8 NPUs, including three
optimizer updates;four consumer CTests. Public FP16 atol.001/rtol.02; consumer
explicit atol.002/rtol.02. Discrete routes and None/zero checks remain exact.
The CPU CSR follow-ups passed20 focused tests and one named CLI rejection.
CUDA-linked build and22 host checks passed; no NVIDIA hardware qualification.
Python/native FP16 tiny profiles passed with1644/1638 hardware kernels; host
scalar work remains. Existing FP32 four-CANN evidence is unchanged. New FP16
covers Torch/TorchNPU2.10/CANN9.0 only. Evidence commit6203063 is supplemented by
the7-chip qualification records in this documentation increment; final
performance documentation is still pending.

## Active final performance pair

TASK_ROOT=/mi/data2T/zlong/tide-npu-performance.
Unit tide-npu-performance-pair-train-attention7-fp16-a5d.service is active in
background.slice, Nice10. It started2026-09-29T01:44:19Z and is queued at position1
for seven chips. The site launcher excludes physical1 after two real collisions
on that chip; all other physical IDs remain candidates. Latest free candidates
are2,5,8,9,11,12: six chips, no cell started. User explicitly chose to wait.

Frozen source TASK_ROOT/sources/fp16-a5b, commit595dccd0b23bd809bed28874591ae4f5f268512f;
matching standalone build TASK_ROOT/builds/client-fp16-a5b. Loaded public module
libtorch-npu/2.10.0-cann9.0.0 with the shared /usr/local driver. The exact command,
source and cwd are in runs/pair-train-attention7-fp16-a5d/status.json. After the
cooperative queue wrapper, it runs:

    /opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python TASK_ROOT/launchers/run-dtype-pair-a5b.py TASK_ROOT/runs/pair-train-attention7-fp16-a5d TASK_ROOT/launchers/plan-train-attention7-fp16-a5.json

One allocation holds FP32 then FP16 fresh processes: D2048/B512/V50304,465 nodes,
Attention17,269,426,339 parameters,12-token complete windows, AdamW, two updates
(first warmup, second measured). Locality placement, resident payload, NPU FP32
Read, CPU controls/ranking/events,16 node workers/ATen1,TASK_QUEUE_ENABLE=0.
Queue wait bound7200s; each dtype runtime bound7200s, RSS256GiB. No dimension,
batch or history reduction. This7-chip pair is not directly comparable to the
old8-chip training timings. Seven-chip small-tensor gates passed before launch.

Inspect systemctl --user show UNIT -p ActiveState -p SubState -p ExecMainStatus
-p MainPID, and TASK_ROOT/runs/NAME/{status.json,queue.json,stages.json,task.log}.
Native progress is NAME/float32/stdout.log and NAME/float32/native/metrics.jsonl
(later float16). Stop only this identified service if needed using systemctl
--user stop UNIT. Do not stop other accelerator processes.

Three pairs/six cells completed on the same595dccd consumer:
- pair-infer-add-fp16-a5b:2 chips, CPU64 Read; FP32/FP16 mean17.412729/18.703945
  ms/sample-token; max per-chip allocator peak19.868474/9.954680 GiB.
- pair-infer-attention-fp16-a5b:4 chips, CPU64 Read;54.611865/45.855257;
  peak22.286432/11.192269 GiB.
- pair-train-add-fp16-a5b:4 chips, CPU32 Read;39.482509/42.135417;
  peak41.058367/47.849997 GiB. FP16 masters add35.27 GiB total.
All identity, placement and clean-exit pair checks passed. Six raw records and
six Trackio projections have been validated; one shared-host pair per mode is
exploratory, not causal speedup or stable variance evidence.

Trackio best-effort local project tide-npu-performance, TASK_ROOT/trackio,
storage auto resolving SQLite, version0.35.0; writer/viewer
/home/zlong/venvs/trackio/bin/python. Raw JSONL remains authoritative.

## Remaining actions

Monitor the final pair to terminal state, checking actual contention. Preserve
failures and diagnose before any changed retry. Validate the final terminal raw
records with launchers/validate-new-records-a5.py JOB and compare every Trackio
step/metric with raw JSONL; ledgers are runs/record-validations-a5.json and
runs/trackio-validations-a5.json. The collector accepts the final successful Attention job as its second argument
(default pair-train-attention7-fp16-a5d) and retains all three cancelled attempts. Then run
launchers/collect-fp16-evidence-a5.py qualification and performance, write the
performance report, update ROADMAP/support contract, validate schema and diffs,
and commit evidence separately. Do not mark H3 verified until all eight cells
pass. No production source is currently being edited.

## Retained limits and attempts

The original8-chip pair-train-attention-fp16-a5b was cancelled while queued with
no cell. The first7-chip pair-train-attention7-fp16-a5b and second a5c were cancelled after
unrelated processes entered physical1; both had zero updates/metrics, native
exit-15, wrapper143 and remaining_group_pids=[]. See runs/resource-contention-a5.json
and runs/pair-train-attention7-fp16-a5c/contention.{json,npu.txt}. Only the project
services were stopped. Both cancelled raw records passed lifecycle validation.
The current a5d excludes physical1; none of the three cancellations is a pass.
Other development failures are retained in the FP16 qualification report.

Payload/state/KV/messages can reside on NPUs and use device-to-device copies;
Read/controls/ranking/event work are configurable. Host histories, scalar/index
returns, tensor handles and C++ dispatch remain: no fully device-resident control
loop claim. Tiny older msprof trace has55 AiCPU int64 Sort tasks,14.62% of summed
operator task time, not end-to-end or full-size attribution. FP16 profiles do not
replace full-size profiling. Public standalone NamedOptimizer/owner checkpoints
remain FP32/FP64; the consumer owns its masters. CUDA and other architectures or
new software combinations still need target-machine acceptance.

Re-entry: git status --short --branch; python scripts/status.py.
