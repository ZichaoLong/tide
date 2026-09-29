# Current handoff

Updated: 2026-09-29. Branch graph-execution-foundation; no push authorized.
No sub-agents. Reference repositories and ObsidianVault remain read-only.
The user authorized completing public Python/native and consumer FP16,
correctness gates, profiling and full-size inference/training comparisons.
Latest resource authorization: more chips, nine chips, shared devices and
自主插空 to prioritize completion. No repeated sharing/card-count permission needed.

## Verified implementation

fc2a76f: explicit FP16 payload and FP32 masters/slots/loss with static scaling.
cb58110: profiler copy/stream ordering. 595dccd: matched-dtype consumer VJPs and
FP32 master trajectories. c8d2b61/4a7dec7: early CPU Half CSR rejection and its
named CLI case. Matching core hashes permit reuse of the fc2a76f core build.

Passed: 8645 CPU tests; 39 Python and 43 native NPU FP16 cases plus CSR rejection;
five full FP32 representative cases per implementation; 124 consumer cells on
CPU and 2/4/6/7/8/9 NPUs; four consumer CTests; 20 CSR boundary follow-up tests
and one CLI rejection. The nine-chip eight-cell gates completed at 02:35:08Z.
Public FP16 atol .001/rtol .02; consumer explicit atol .002/rtol .02. Discrete
routes and None/zero checks remain exact. CUDA-linked build and 22 host checks
passed, without NVIDIA hardware qualification. Python/native tiny FP16 traces
contain 1644/1638 hardware kernels; host scalar work remains. New FP16 evidence
covers Torch/TorchNPU 2.10/CANN 9.0 only; prior four-CANN FP32 claims are unchanged.

Qualification JSON, Markdown, ROADMAP and support contract include nine chips
(124 consumer cells) in this evidence-only increment. Production source is
unchanged; the final performance report and workload qualification remain pending.

## Active final performance pair

TASK_ROOT=/mi/data2T/zlong/tide-npu-performance.
Unit tide-npu-performance-pair-train-attention9-fp16-a5g.service is active in
background.slice. Started 02:33:02Z; physical 0,2,3,4,5,11,12,13,14 map to npu:0..8.
Nine-chip gates and FP32 full-size training passed; FP16 now runs in
a fresh process on the same allocation. At 03:03:51Z FP32 passed both updates and exited cleanly (warmup699.126114s;
measured717.549002s /116.788575ms per sample-token, finite loss10.2228537,
peak35.854233GiB/chip). Its raw record and Trackio projection both validated.
FP16 is running on the same allocation; no completed update yet.
Shared admission permits existing processes; health, used HBM <=24576MiB,
AiCore <=100%, two snapshots 3s apart and cooperative locks are checked.
Task-local run-on-shared-npu-a5f.py overrides only process-free eligibility;
installed helper and other workloads remain untouched. Foreign PIDs are expected;
record their presence rather than cancelling merely because they exist.

Frozen source TASK_ROOT/sources/fp16-a5b at
595dccd0b23bd809bed28874591ae4f5f268512f; build TASK_ROOT/builds/client-fp16-a5b.
Module libtorch-npu/2.10.0-cann9.0.0, public /opt stack/shared /usr/local driver.
Runtime Python /opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
The exact resolved command and cwd follow below; plan is
TASK_ROOT/launchers/plan-train-attention9-fp16-a5g.json.

Attention retains D2048/B512/V50304, 17,269,426,339 parameters, 465 nodes,
2208 logical/4418 physical edges, locality placement, resident payload, seed7,
workers16/ATen1, TASK_QUEUE_ENABLE=0. NPU FP32 Read, CPU controls/ranking/events.
AdamW, two complete 12-token updates per dtype: first warmup, second measured.
No dimensions, batch or window reduced. Each dtype timeout7200s, RSS256GiB.
This is shared-load exploratory performance, not exclusive or causal speedup.

Seven-chip fallback tide-npu-performance-pair-train-attention7-fp16-a5d.service
remains queued (priority0, started01:44:19Z, max wait7200s), no cell started.
The nine-chip launcher cancels that exact fallback only after both dtypes pass.
If it times out first, preserve its failed terminal state; adapt the collector
instead of relabelling it cancelled. Do not start duplicate runs.

## Remaining work

1. Monitor with TASK_ROOT/launchers/inspect-fp16-attempts-a5.py. It appends resource
   observations; stop appending to completed jobs before collecting their hashes.
   Inspect status.json, queue.json, pipeline-stages.json, stages.json, task.log,
   float32|float16/stdout.log and native/metrics.jsonl under the current run.
2. On terminal success, run TASK_ROOT/launchers/validate-new-records-a5.py JOB
   and validate-trackio-records-a5.py JOB. Ledger files under runs/ currently
   hold 11 raw cells (7 success +4 failed/cancelled) and 7 Trackio comparisons.
3. Run collect-fp16-evidence-a5.py performance pair-train-attention9-fp16-a5g,
   then write-fp16-report-a5.py. Both passed AST parsing. Review all generated
   numbers/wording and fallback status. Do not mark H3 verified until eight
   full-size cells pass. Update support workload evidence at exact 595dccd.
4. Replace this handoff with final state; schema-check the support contract,
   git diff --check, review evidence links/scopes, commit evidence only, no push.
5. If the pair fails, preserve terminal evidence, diagnose resource versus code
   failure and autonomously choose the next reasonable shared allocation.

Trackio: best-effort local project tide-npu-performance, TASK_ROOT/trackio,
SQLite tide-npu-performance.db, version0.35.0; writer/viewer
/home/zlong/venvs/trackio/bin/python. Raw JSONL is authoritative. No dashboard.

## Retained completed work and unsuccessful attempts

Three pairs/six cells passed on 595dccd; FP32/FP16 ms/sample-token and peak GiB/chip:
- infer-add, 2 chips/CPU64 Read: 17.412729/18.703945; peak19.868474/9.954680.
- infer-attention, 4 chips/CPU64 Read: 54.611865/45.855257; peak22.286432/11.192269.
- train-add, 4 chips/CPU32 Read: 39.482509/42.135417; peak41.058367/47.849997.
Add has 9,468,020,899 parameters; FP16 masters alone add35.27GiB total.
Each pair passed identity/placement/clean-exit checks and raw/Trackio validation.
One pair per mode cannot establish stable averages or causal dtype speedup.

Original 8-chip a5b: cancelled queued, no cell. Seven-chip a5b/a5c: cancelled
after external processes entered physical1, zero updates; native-15/wrapper143,
no descendants. See resource-contention-a5.json and a5c/contention.json.
Six-chip a5e: correctness gates passed; FP32 full-size OOM at02:17:47Z on
logical0/physical2 while an external process was present. Requested66MiB,
allocated36.88GiB, reserved38.04GiB, free34.63MiB of61.27GiB usable capacity.
This does not prove clean six-chip capacity insufficiency. FP32 failed exit1;
automatically started FP16 was cancelled under the earlier user condition.
Both zero updates, no descendants; failure-analysis/resource logs retained.
Outer status and terminal cell summaries override stale running stage snapshots.
Eleven-chip a5f: cancelled before any gates/cells as only nine chips had room;
cancellation-reason.json retained. No unrelated process was stopped.

## Interpretation boundaries

NPU payload/state/KV/messages and device-to-device copies are implemented;
Read/controls/ranking/events are independently configurable. Host histories,
scalar/index returns, tensor handles and dispatch remain; no fully NPU-resident
control loop. Single-process sharding, not DDP/HCCL. Transfer counters measure
requested bytes, not physical fabric. Older tiny msprof:55 AiCPU int64 Sort tasks,
14.62% summed task time, not end-to-end/full-size attribution. New FP16 traces
use a different path; absence of AiCPU there is not a dtype-causality result.
Public standalone NamedOptimizer/owner checkpoints remain FP32/FP64; consumer
owns FP32 masters. BF16/AMP, CUDA real-device and other architectures/stacks
need separate acceptance. Re-entry: git status --short --branch;
python scripts/status.py; read this file.

## Exact active command

Working directory: `/mi/data2T/zlong/tide-npu-performance/sources/fp16-a5b`.

```bash
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python /mi/data2T/zlong/tide-npu-performance/launchers/run-on-shared-npu-a5f.py --shared-use-authorized --job-id tide-pair-train-attention9-fp16-a5g --project tide-npu-performance --priority 20 --npu-count 9 --max-hbm-mib 24576 --max-aicore-percent 100 --max-wait-seconds 7200 --candidate-ids 0,2,3,4,5,8,9,11,12,13,14 --state-file /mi/data2T/zlong/tide-npu-performance/runs/pair-train-attention9-fp16-a5g/queue.json -- /opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python /mi/data2T/zlong/tide-npu-performance/launchers/run-attention9-gated-a5g.py /mi/data2T/zlong/tide-npu-performance/runs/pair-train-attention9-fp16-a5g
```

Inspect/stop only the identified project unit with systemctl --user show/stop;
never stop other accelerator workloads.
