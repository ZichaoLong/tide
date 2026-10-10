# Current handoff

Updated 2026-10-10 (Asia/Shanghai). **Local selection-focused delivery closed;no current task jobs.**
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
qualified CPU/NPU profiles. F6/F7 local selection/report/terminal review is now
closed under the decision-sufficient criterion,with explicit missing timings.
This is not all-matrix success or unconditional strict wide-trajectory equivalence.
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
historical unbound observations stay separate. The old audit-only summary omitted
CPU pre-admission failures; the separate read-only reconciliation merged terminal
receipts and audits without changing accepted metrics. These counts describe
the original matrix;the two later failed attempts are recorded separately below.
All4 FP16 companions have first-process evidence; automatic repetition is cancelled.

## Completed follow-up review

Implementation commit `d773fe294994711420480f6d679970e3feacf65f` fixes the monitor
ENOENT/ESRCH race and adds terminal-failure reconciliation. Clean immutable
`TASK/sources/control-selection-clean01` passed27 affected checks (20 resource/
process lifecycle+7 summary);no model code or broad Torch gate changed.
Checks: `TASK/plans/selection-monitor-check01/{result.json,tests.log}`.
Read-only reconciliation passed: `TASK/plans/selection-reconciliation01.json`;
all61 bound/8 unbound accepted metrics unchanged;all59 missing bound cases now
classified;five over-bound observations rechecked without changing old cases.

Exactly two one-attempt follow-ups were submitted. Both are terminal failed and
strictly audited;neither produced an accepted timing. The manager finished
2026-10-09T16:38:25.565851Z (**2026-10-10 00:38:25 Beijing**),exit0;this means
queue completion,not measurement success. Manager and both child cgroups are
empty/MainPID0. No follow-up remains running,queued or automatically scheduled.

- Cell21 CPU PDG/LibTorch Attention prefill complete training ran to its24400s
  execution bound (monitor24410.407s including cleanup). No complete consumer
  result. Peak group RSS393.451GiB was below420.110GiB lane allowance;failure was
  timeout,not an observed RSS breach. NUMA0..7 memory/original80-core CPU mask/
  BLAS16 is separate `numa-bound-solo-wide-memory-v1`. CPU full-size Attention
  training comparison remains unresolved;total elapsed time is not a step metric.
- Cell56 Settle/LibTorch resident Attention prefill inference failed11-card
  admission (configured120s,last poll136.013s). Model never started;there is no
  new NPU failure or full-size monitor-fix qualification. The27 regression checks
  remain valid. Old process-exit-race failure is not relabelled passed.
- Cells16/59 were deliberately omitted. No retry,budget increase or new matrix.

[Final disposition/support limits](evidence/selection-terminal-20261010.md);
[scenario selection and scalars](evidence/selection-review-20261009.md).
Add/LibTorch prefill favors CPU in measured cases;Attention prefill inference
favors resident;Attention training retains a useful mixed-A observation but lacks
a complete full-size CPU baseline. Python caller identity changes some choices.
Single shared-server observations do not establish a universal fastest flow.

Records remain under TASK:
`plans/selection-followups01.json`,`plans/selection-terminal-review01.json`,
`runs/selection-followups01/{status.json,task.log,dispatch/result.json,dispatch/audits/}`,
`runs/formal-bound-selection01-cell{021,056}`. Original manifests,helpers,
snapshots,binaries,audits and failed receipts remain unchanged;18 manifest inputs,
source identities and both audit hashes were rechecked. Terminal unit names:
`tide-execution-flows-selection-followups01.service` and
`tide-execution-flows-formal-bound-selection01-cell{021,056}.service`.

No pending local execution command. A later status request should read these
terminal records,not restart their services. Future GPU/x86/version work,extra
measurements or new selector/canon semantics require their own concrete scope.
The existing [target validation commands](eager-target-validation.md) remain ready.
Protected historical stopped CPU work remains unrelated and untouched.

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
