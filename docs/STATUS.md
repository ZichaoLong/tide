# Current handoff

Updated 2026-10-09 (Asia/Shanghai). **Selection-focused closure in progress.**
The user authorized execution sufficient to answer engineering flow-selection
questions; all120 performance cells need not execute. Commit/push authorized.
No subagents. [execution-flows](execution-flows.md) outranks experiment-skill
overhead. Only minimal existing records; no new Trackio infrastructure.
Selector/new upstream semantics are deferred until the user supplies a new canon.
Reference repositories and ObsidianVault are read-only.

Repo `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. `TASK=/mi/data2T/zlong/tide-execution-flows`.
Re-entry:git status,python scripts/status.py,this file,[ROADMAP F1–F7](ROADMAP.md).
Use scripts.durable_records for atomic/fsynced handoff writes.

## Current scope and terminal queue

F1–F5 implementation/current local correctness gates are complete in their
qualified CPU/NPU profiles. F6/F7 require selection/report reconciliation and
terminal review of any targeted follow-ups, not all-matrix success.
Actual NVIDIA/x86_64 execution and other environment versions stay target-pending.

`unattended-matrix03` finished at2026-10-09T05:26:52.428668Z (13:26 Beijing),
receipt passed/exit0; unit inactive/MainPID0/empty cgroup. Queue state
`finished-awaiting-review` means the finite list is exhausted, not model success.
`TASK/runs/unattended-matrix03/{status.json,dispatch/result.json}`.
All32 recent children are terminal/empty;826 frozen inputs were verified unchanged.
No automatic repeats ran. Matrix01 disk-stop and matrix02 authorized cancellation
remain historical; no old manager should be resumed.

Bound FP32:61/120 accepted. The59 unaccepted cells comprise37 NPU admission
timeouts,10 CPU NUMA admission refusals,5 complete-over-time-bound observations,
4 execution timeouts,2 monitor process-exit races and1 NPU OOM. Eight accepted
historical unbound observations stay separate. The old audit-only summary omits
CPU pre-admission failures; its accepted metrics stay unchanged, while a new
reconciliation will merge terminal receipts as well as audits.
All4 FP16 companions have first-process evidence; automatic repetition is cancelled.

## Current follow-up plan

Implementation commit `d773fe294994711420480f6d679970e3feacf65f` fixes the monitor
ENOENT/ESRCH race and adds terminal-failure reconciliation. Clean immutable
`TASK/sources/control-selection-clean01` passed27 affected checks (20 resource/
process lifecycle+7 summary);no model code or broad Torch gate changed.
Checks: `TASK/plans/selection-monitor-check01/{result.json,tests.log}`.
Read-only reconciliation passed: `TASK/plans/selection-reconciliation01.json`;
all61 bound/8 unbound accepted metrics unchanged;all59 missing bound cases now
classified;five over-bound observations rechecked without changing old cases.

Only **two** of the authorized maximum4 follow-ups are selected, each once:
cell21 CPU PDG Attention prefill complete training, then cell56 LibTorch resident
Settle Attention prefill inference. Cells16/59 are deliberately not queued;their
remaining timing gaps do not block provisional Add/Attention guidance.
CPU21 uses NUMA0..7 memory with the original80-core CPU mask/BLAS16 and all memory
guards, in separate `numa-bound-solo-wide-memory-v1`. NPU56 retains11cards and
`numa-bound-solo-v1`;only the monitor fix differs. Workload/binaries stay e69b3bd.

Plan: `TASK/plans/selection-followups01.json`; preflight:
`TASK/plans/selection-preflight01/result.json` (two command/lane/admission
preparations,three invalid requests rejected;no device/model execution).
The task-local `*selection01.py` helpers and launchers are frozen by the manifest;
do not modify them while submitted. Old controllers/launchers/results unchanged.

Service submitted2026-10-09T09:48:58Z, **verified running**;CPU21 is running,
NPU56 is pending serial dependency. No follow-up result is claimed:
`tide-execution-flows-selection-followups01.service`;
cwd `TASK/sources/control-selection-clean01`;
entry `/bin/bash --noprofile --norc TASK/launchers/selection-followups01.sh`.
This runs `/usr/bin/python3 scripts/job.py --output-dir TASK/runs/selection-followups01`
with the frozen `run_selection01.py --plan TASK/plans/selection-followups01.json
--out TASK/runs/selection-followups01/dispatch` under timeout27920s.
Cell21 step12000s/child24400s/outer24760s;cell56 step900s/child2200s/outer2560s.
Whole protective bound27920s (~7h46m),not an ETA. CPU lane420.1097GiB plus136GiB
reserve;host half-memory budget checked again at dispatch. NPU admission120s.
Serial detached child services have PartOf propagation to this manager;no retries.
Receipts/logs: `TASK/runs/selection-followups01/{status.json,task.log,dispatch/result.json}`;
children: `TASK/runs/formal-bound-selection01-cell{021,056}`.

Inspect: `systemctl --user show tide-execution-flows-selection-followups01.service
-p ActiveState -p MainPID -p ControlGroup`;read its status and dispatch JSON.
Stop only this new queue if requested:
`systemctl --user stop tide-execution-flows-selection-followups01.service`.
On wake-up, verify terminal receipts,empty cgroups and strict domain audits;
merge accepted follow-up evidence by series,retain failed/admission outcomes,
then close F6/F7 local report reconciliation. No new job is automatically added.

Selection conclusions and complete reviewed scalar evidence:
[evidence/selection-review-20261009.md](evidence/selection-review-20261009.md).
Submission identity: `TASK/plans/selection-followups01-submission.json`.

## Contract and limitations

Independent CPU,mixed and resident execution; online greedy prefill and streaming
for all legal family inputs/topologies, PDG positive-delay feedback, packed work,
bounded capacity, continued windows and complete backward/optimizer/checkpoint.
No CPU-precomputed routes/events/gradients feed candidates. Preserve int64,
stable ties, physical duplicate-edge identity, missing/zero, None/zero and VJPs.
Five presets and fine controls remain configurable. PDG performance uses LibTorch;
TimedDAG/Settle use LibTorch and Python. Python-owned native resident is distinct
from pure PyTorch and standalone LibTorch. Training means the complete mechanism,
not downstream convergence/quality or training recipes.

Keep the strict original-scale near-tie failure accepted by the user as a
nonblocking limitation. CPU246 is ahead by one FP32 ULP while resident245/246 tie
and choose245; proposal error7.7039e-6,events2325/2327 in the retained witness.
Do not relax comparisons or call all wide trajectories strictly equivalent.
Actual event counts differ; timing ratios are observed input throughput comparisons,
not certified equal-event speedups. Shared-server contention is not excluded.
[Witness](evidence/original-add-route-witness-20261004.md).

Keep original3000s/1.15 refusals and all declared finite per-cell budgets.
No automatic timeout increases, retries or CPU worker sweep. One warmup and one
measured step,each2 continued windows; FP32 primary/FP16 separate. No statistical
stability or universal fastest-flow claim. Project heavy jobs remain serial.
NPU admission120s through the account helper;disk reserves data24GiB/root8GiB.
Long work uses background.slice/Nice10/KillMode=control-group, separate from focus.
The user may inspect on wake-up; no continuous polling/automatic callback required.

## Qualified evidence to reuse

- CPU integration78e9df6:9458 passed,654 scoped optional skips,12/12 CTests.
  [CPU](evidence/integrated-cpu-20261004.md).
- Exact discrete comparator e69b3bd:CPU121/NPU49,including int64>2**53.
  [Comparator](evidence/exact-discrete-comparison-20261004.md).
- NPU e69:49 eager+93 resident,no skips;families,schedules,FP32/FP16,SGD/AdamW,
  continuation and fresh-process2→3-owner restore.
  [NPU](evidence/integrated-npu-consumers-20261004.md).
- CPU RSS c6ef224:108 checks;CUDA-linked aarch64 host187+7 relocation,
  updated eager FP16/RSS108 CPU checks. Actual NVIDIA/x86 execution pending.
  [Target commands](eager-target-validation.md).
- All10 representative family/client/schedule submatrices completed.
  Workload `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`,
  `TASK/sources/compare-discrete-clean01`. Old controller103f5b6,
  `TASK/sources/control-isolation-clean02` stays immutable.
- Keep hashed builds under TASK/builds:integration-core-{standalone-clean02,python-clean03},
  integration-resident-{standalone-clean02,python-clean03},integration-online-clean01,
  eager-rss-training-cpu-clean01 and eager-half-cuda-clean01.
- FP16 Add/Attention inference194.098/222.541s;Add training2027.354s on8cards;
  Attention training6190.644s on11cards. Dtype/card/cache/placement differences
  prevent attributing timing differences solely to FP16.
  [FP16/profile evidence](evidence/formal-b512-fp16-training-profiles-20261004.md).
  Five-second sampled FP16 inference traces had no AiCPU tasks;unsampled work is
  not certified AiCPU-free. FP32 profiles also completed. Retain all raw traces.

## Environment and protected history

Public module `libtorch-npu/2.10.0-cann9.0.0`;Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
This authorization overrides stale private-guide paths. Driver/usr/local unchanged.
Module init/purge/use private directory/load public entry as existing launchers;
preserve module PYTHONPATH,prepend workload/python. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0,PYTHONDONTWRITEBYTECODE=1,ACL_OP_INIT_MODE=0.
CPU timings OMP/OpenBLAS16,MKL1,OMP_THREAD_LIMIT32;NPU startup BLAS1.
Put new snapshots and results on data filesystem. At this checkpoint free disk
is about8.9GiB root/129.4GiB data;recheck admission,do not investigate growth.

**Never resume,stop,signal or clean historical-cpu-attention-01(worker2686919).**
It is deliberately stopped and unrelated to this task. `scripts/status.py` exit1
is the known malformed historical build-reverse-gather-python-dev01 record;
do not change historical evidence. Prior cleanup records and hashed artifacts
remain;no new cleanup authorized/needed for this increment.
