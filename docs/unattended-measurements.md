# Finite unattended measurements

The user requested background execution and a result review on the next wake-up.
Implementation/qualified CPU-NPU profiles remain those in [STATUS](STATUS.md);
this queue only fills performance evidence. It creates no recommendation or
production-model change. Actual NVIDIA/x86_64 execution remains target-pending.

`TASK=/mi/data2T/zlong/tide-execution-flows`.
Manifest:`TASK/plans/unattended-matrix01.json`,SHA256
`4635b10838ebbfda35a9484986003fbe4428d67d6b7c88eb450ec281e5a19242`.
Manager:`TASK/launchers/run_unattended_measurements.py`,with policy/runtime helpers
in the same directory. They and644input files are frozen by the manifest.
Workload:`e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`;
controller:`103f5b6c8e9b2e6e46a7f2733185433de2cb5f3f`.
All children reuse the existing source snapshots,qualified builds,dispatchers,
strict auditors,per-cell budgets and public module libtorch-npu/2.10.0-cann9.0.0.

## Order and scope

1. Wait without leases for the four already-submitted serial continuation jobs.
   Check terminal receipts,source,exit codes and empty cgroups;strictly audit
   available results. Preserve failed or unstarted predecessor cells separately.
2. Run the prepared default-cache FP32 Attention inference trace once. The
   completed FP32 Add and both FP16 inference traces are not repeated.
3. Fill the120-cell FP32 matrix in the NUMA-bound solo series. There are113
   potential first-process slots:the six accepted bound cells and attempted
   over-bound cell36 are excluded. Cells60/84/108 skip if the existing chain
   already attempted them. Eight old unbound cells get new bound measurements
   for comparable repetitions;their old results are not relabelled or pooled.
   Shorter declared child allowances run first to obtain breadth earlier.
4. For each of40family/client/schedule/model/mode scenarios,select its accepted
   CPU baseline and fastest accepted first-process NPU candidate for repeats2/3.
   At most160FP32 repetitions. This is screening,not a claim that the winner is
   globally fastest. Failed or missing competitors remain explicit;the slower
   unselected NPU flow retains its first process. A failed repeat2 suppresses3.
5. Independently repeat each accepted FP16 companion to three processes,
   at most8new jobs. Attention training requires its pending first process to
   pass. New v4execution/v5audit wrappers change repetition metadata only;the
   old helpers and raw records stay immutable. FP16 series never enter FP32
   aggregates. Final recommendations and support/evidence review await the user.

There are362fixed command slots with conditional eligibility,at most282new
heavy executions;not362guaranteed runs. One attempt per eligible slot,no automatic
retries,budget increases,model resizing or switch-combination expansion.

## Resource and failure limits

Only one heavy workload runs at a time. The earlier CPU/NPU overlap screen was
rejected,so this queue preserves solo timing. Primary NPU cells lease11devices,
FP16 uses8or11. Original60GiB/card NPU/512GiB CPU static budgets,host/NUMA admission,
thread limits and runtime monitors remain in force. CPU BLAS/OpenMP16,MKL1;
resident ATen8 with startup BLAS1. No device lease while waiting on predecessors.
Each NPU allocation waits at most120s under the account cooperative queue.

The total conservative finite allowance is3704364s(**1028.99hours**,including
barrier/cleanup/audit allowances). This is a sum of upper bounds,**not an ETA**.
Existing transferred forecasts imply hundreds of hours;repetitions can take
weeks. At least24GiB free on the data filesystem and8GiB on root are required
before every new child. Reaching a reserve stops submission without deleting
logs,checkpoints or other users' data.

A workload or admission failure stays failed;after verified cleanup the manager
can continue independent jobs. A failed partial record rejected by its domain
auditor is labelled failed-unaccepted,with the audit log retained. A successful
receipt rejected by strict auditing,changed immutable input,cancellation,missing
receipt,cleanup failure or disk exhaustion stops the manager for review.

Services run in background.slice,Nice10,KillMode=control-group. New child services
have PartOf the manager,so stopping it also stops its current new child. It never
stops the four pre-existing dependencies or protected historical CPU worker.
There is no automatic restart or reboot recovery. The services survive loss of
the current agent/SSH session while the user manager remains alive.

## Inspection and stop commands

Unit:`tide-execution-flows-unattended-matrix01.service`. Verified active/transient,
background.slice/Nice10,MainPID958799 at2026-10-04T07:18:12.346503Z. Initial state
waits for the existing CPU streaming chain;the two FP16 profile dependency audits
already passed. Submission receipt:`TASK/plans/unattended-matrix01-submission.json`.
Working directory:`TASK/sources/control-isolation-clean02`.
Resolved manager command (inside the persistent scripts/job.py receipt):

```bash
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python \
  /mi/data2T/zlong/tide-execution-flows/launchers/run_unattended_measurements.py \
  --plan /mi/data2T/zlong/tide-execution-flows/plans/unattended-matrix01.json \
  --out /mi/data2T/zlong/tide-execution-flows/runs/unattended-matrix01/dispatch
```

Inspect without importing Torch or reserving hardware:

```bash
systemctl --user show tide-execution-flows-unattended-matrix01 \
  -p ActiveState -p SubState -p MainPID -p Result -p ExecMainStatus
cat /mi/data2T/zlong/tide-execution-flows/runs/unattended-matrix01/dispatch/result.json
systemctl --user stop tide-execution-flows-unattended-matrix01
```

`TASK/runs/unattended-matrix01/status.json` and `task.log` retain manager lifecycle
and exit status. `dispatch/result.json` distinguishes waiting,running,failed,
skipped and finished-awaiting-review;it lists every submitted child,audit and
current item. Each child has its own `TASK/runs/NAME/status.json`,`task.log`,queue,
assessment and consumer artifacts. `dispatch/audits/` contains strict audits or
retained refusal logs. Each completed phase writes `summary-PHASE/summary.json`
and a separate `fp16-summary.json`;`repeat-selection.json` fixes the candidates
before repetitions. A passed manager receipt means the finite plan ended,not
that all experiments passed or that the overall project is complete.

## Validation of the queue itself

13policy/lifecycle guards passed,including changed input,cancellation,stale
receipts,live cgroups,unstarted versus attempted cells,failed-repeat suppression,
low disk and continuation after a failing real short child. Three additional
real transient-service guards passed natural success/failure and PartOf stop
propagation with empty cgroups. The first ad hoc service guard checked the child
too soon after stopping its parent;that failure and original receipts remain.
The separate second guard waits at most10s for propagation. No model or device
was used by these guards. All362launchers passed shell syntax checks.

Evidence:`TASK/plans/unattended-queue-guards01/result.json`,
`TASK/plans/unattended-service-guards02/result.json`,and
`TASK/plans/unattended-matrix01-validation.json`. The queue is not evidence that
any currently unmeasured model cell has passed.
