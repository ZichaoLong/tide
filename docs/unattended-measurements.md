# Completed finite selection follow-ups

On2026-10-09 the user authorized decision-sufficient evidence;all120 cells need
not execute. The previous matrix03 finished at13:26 Beijing with61/120 accepted
bound FP32 cells and59 explicitly classified outcomes. All4 FP16 first processes
exist. Repetitions remain cancelled. Old manifests,receipts,audits and failures
stay unchanged. Queue exit0 means the finite queue ended,not every model passed.
[Current handoff](STATUS.md) owns live state;[contract](execution-flows.md) owns scope.

`TASK=/mi/data2T/zlong/tide-execution-flows`.

## Terminal review,2026-10-10

The queue finished at2026-10-10 00:38:25 Beijing. Manager and both child cgroups
are empty. CPU21 reached24400s execution bound without a complete result;NPU56
failed11-card admission without running the model. Both are audited-failed;
no accepted timing was added and no retry is scheduled. Manager exit0 does not
convert those failures to success. [Reviewed terminal evidence](evidence/selection-terminal-20261010.md).
The scope/commands below identify the retained past submission,not an active queue.

## Retained finite scope

`TASK/plans/selection-followups01.json` freezes exactly2 serial jobs:

| Order/cell | Question | Preserved step/child bound |
| --- | --- | --- |
| 1/21 | PDG LibTorch CPU Attention prefill complete-training baseline | 12000s/24400s |
| 2/56 | Settle LibTorch resident Attention prefill inference after monitor fix | 900s/2200s |

Cells16/59 from the maximum4 candidate list are omitted for limited decision value.
No full-matrix restart,retry,statistical repetition,model resizing or budget growth.
External contention remains possible and does not trigger automatic reruns.
One warmup and one measured step,each2 connected windows,remain unchanged.

CPU21 retains80 allowed cores,OMP/OpenBLAS16,MKL1 and the original memory estimate.
Its memory mask becomes NUMA0..7;the old0..3 mask repeatedly refused the412GiB
estimated model peak. Lane420.1097GiB plus shared reserve136GiB must fit the
half-host budget and conservative selected-node availability estimate. This is
separate `numa-bound-solo-wide-memory-v1`;do not pool it with old NUMA timings.

NPU56 keeps the eleven-device qualified placement and120s admission through the
account helper. Insufficient cards ends that attempt without model execution.
NPU timings retain `numa-bound-solo-v1`;the controller patch handles process
exit during monitoring without changing workload code or unrelated error checks.

## Identity and launch

Workload `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8` and existing hashed binaries.
Controller `d773fe294994711420480f6d679970e3feacf65f`,clean worktree
`TASK/sources/control-selection-clean01`. Public module
`libtorch-npu/2.10.0-cann9.0.0`;Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
Task-local manager/dispatcher/case/auditor are versioned `*selection01.py`.
Manifest hashes helpers,launchers,inputs and the controller snapshot inventory.
Trackio is off under the user contract;existing durable project records are used.

Unit `tide-execution-flows-selection-followups01.service`,cwd the clean controller,
command `/bin/bash --noprofile --norc TASK/launchers/selection-followups01.sh`.
The wrapper records status via scripts/job.py and invokes run_selection01.py
with the frozen plan and output `TASK/runs/selection-followups01/dispatch`.
Each job runs in a distinct child service;PartOf propagates queue cancellation.
Services use background.slice,Nice10,KillMode=control-group,finite RuntimeMax,
and an environment independent of focus.service. They survive session/SSH
loss while the user manager lives,without automatic host-reboot recovery.

The submission serialized project heavy jobs using existing manager/timing locks. It required
24GiB data/8GiB root free space before each child. Whole queue protective bound
27920s (~7h46m) includes finite execution/lifecycle allowances;it is not an ETA.
No continuous agent polling or automatic callback is required.

## Inspect retained records

Read `TASK/runs/selection-followups01/{status.json,task.log,dispatch/result.json}`
and `dispatch/audits/`. Children are
`formal-bound-selection01-cell021` and `formal-bound-selection01-cell056`.
Read each child receipt/domain audit and verify no remaining cgroup. Domain
acceptance is required;successful submission or manager exit0 is insufficient.
Failures before an assessment exists are retained with their receipt/log and
classification. Keep the new CPU series separate from old timings.

```bash
systemctl --user show tide-execution-flows-selection-followups01.service -p ActiveState -p MainPID -p ControlGroup
```

Never resume/stop/signal/clean `historical-cpu-attention-01` (worker2686919).
It is unrelated protected history. Source/build/trace/failed-run artifacts remain.
