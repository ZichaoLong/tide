# Finite unattended measurements

The user authorized single-process coverage on 2026-10-08: retain first attempts,
cancel automatic FP32/FP16 repetitions, and accept shared-server observations.
Implementation and correctness gates remain those in [STATUS](STATUS.md).
This queue fills performance evidence; final F6/F7 review remains separate.
Selector/new semantics are deferred. Actual NVIDIA/x86_64 execution is target-pending.

`TASK=/mi/data2T/zlong/tide-execution-flows`.

## Current scope and policy

The frozen `TASK/plans/unattended-matrix03.json` contains31 new FP32 first-process
jobs and one barrier for the already-running `formal-bound-auto01-cell113-r1`.
At handoff:58/120 bound FP32 cells accepted,30failed,one running and31unstarted.
Four separate FP16 companions and the required inference profiles already have
first-process evidence. All248 conditional repeat slots are removed (at most160
FP32 and8 FP16 executions). No completed or failed first attempt is repeated.
One warmup and one measured step, each with two connected windows, stay unchanged.

Single-process results are engineering observations. Preserve actual work counts,
measurement series, dtype/card/cache differences and the strict near-tie limitation.
Do not claim statistical stability or a universal fastest flow. Small differences
may remain inconclusive. Existing multiprocess evidence remains valid in its scope.
The summary retains the factual three-process count but adds first-process coverage;
three processes are no longer a mandatory recommendation gate.

External contention is not excluded or systematically monitored. Reuse existing
configuration, device mapping, timestamps and raw records; an observed contention
may receive a short note. Contention does not trigger retries. During final review,
only a concrete anomaly or unresolved flow choice may justify a targeted follow-up,
initially capped at four configurations; none is automatically scheduled.

## Drain and switch

The original matrix02 manifest and its running source remain unchanged. New service
`tide-execution-flows-unattended-matrix03.service` has verified the old coordinator's
PID/start identity, command, unit/cgroup, manifest and current child. It suspends
only that Python coordinator, preventing new submissions. Cell113's separate
service and11-card workload continue normally.

The handoff waits for the current child to finish naturally with a consistent
terminal receipt and empty cgroup. It then retires matrix02 and starts the new
single-process manager. Matrix02's resulting cancellation means authorized queue
replacement; the child is independently checked by the unchanged strict auditor.
The new manager acquires the existing serialization lock and runs remaining jobs.
No active workload source, child launcher or old manifest is rewritten.

`TASK/runs/unattended-matrix03/handoff.json` owns transition state. During
`draining-current-child`, do not resume the stopped matrix02 coordinator or start
another queue. `dispatch/result.json` is created after drain/retirement.
An error or cancellation stops the owned predecessor; ExecStopPost also handles
interruption during drain, so the coordinator is not intentionally left paused.
Stopping matrix03 during drain cancels the old manager/current child; after the
switch, PartOf propagation cancels the new current child. Protected historical
CPU work and unrelated services are never signalled.

## Identity, resources and limits

Workload: `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`.
Controller: `103f5b6c8e9b2e6e46a7f2733185433de2cb5f3f`.
Working directory: `TASK/sources/control-isolation-clean02`.
Public module: `libtorch-npu/2.10.0-cann9.0.0`.
Manifest SHA256: `02b49ea6ef694077ca0c576b248c52ab24221013938e8a68bdf53958204dd669`.
Task-local manager, handoff and summarizer are versioned `*once03.py`; their hashes,
unchanged prior inputs and launchers are frozen in the new manifest.

Only one project heavy workload runs at a time. NPU jobs still lease11 devices
through the cooperative helper, with120s maximum admission wait. Memory/NUMA,
CPU thread limits, per-child timeouts, runtime checks and strict auditors are
unchanged. Device availability checks remain required; no forced allocation onto
busy cards. Before each new child, require24GiB data and8GiB root free space.
No automatic retry, budget increase, model resizing, cleanup or switch expansion.

The new aggregate protective ceiling is807265s (about224.24h), including drain,
child bounds and lifecycle allowance. It is **not an ETA**. The first-process work
was estimated at about160h before handoff (roughly6–7days), with interruptions and
failed-case review additional. Queue completion does not imply all120 cells pass.

Services use background.slice, Nice10 and KillMode=control-group, outside
focus.service. They survive agent/SSH disconnection while the user manager lives;
there is no automatic reboot recovery. No continuous agent monitoring is required.
Tracking uses project-owned records; Trackio is off under the execution contract.

## Commands and records

Submission: `TASK/plans/unattended-matrix03-submission.json`.
Preflight: `TASK/plans/unattended-matrix03-validation.json`.
Lifecycle/log: `TASK/runs/unattended-matrix03/{status.json,task.log}`.
Transition: `TASK/runs/unattended-matrix03/handoff.json`.
Results after drain: `TASK/runs/unattended-matrix03/dispatch/result.json`.
Phase summary: `dispatch/summary-breadth/{summary.json,fp16-summary.json}`.
Each existing child name retains its own `TASK/runs/NAME/` receipt, log and artifacts.

Resolved handoff command, inside the durable job wrapper and bounded timeout:

```bash
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python \
  /mi/data2T/zlong/tide-execution-flows/launchers/handoff_unattended_once03.py \
  --plan /mi/data2T/zlong/tide-execution-flows/plans/unattended-matrix03.json \
  --state /mi/data2T/zlong/tide-execution-flows/runs/unattended-matrix03/handoff.json \
  --out /mi/data2T/zlong/tide-execution-flows/runs/unattended-matrix03/dispatch
```

Inspect or stop the exact owned service:

```bash
systemctl --user show tide-execution-flows-unattended-matrix03 \
  -p ActiveState -p SubState -p MainPID -p Result -p ExecMainStatus
cat /mi/data2T/zlong/tide-execution-flows/runs/unattended-matrix03/handoff.json
systemctl --user stop tide-execution-flows-unattended-matrix03
```

After drain, the wrapper invokes the same interpreter with
`TASK/launchers/run_unattended_measurements_once03.py --plan TASK/plans/unattended-matrix03.json --out TASK/runs/unattended-matrix03/dispatch`.
The submission receipt retains the complete argv, including timeouts and service
properties. See [policy-change evidence](evidence/unattended-single-process-20261008.md).

## Validation and retained history

Two real transient-service guards passed natural child drain and cancellation
cleanup; the child advanced while its coordinator was stopped, no next slot was
submitted, and all guard cgroups ended empty. These used no model or NPU.
Summary validation preserved all old metrics and failures; repeat jobs and nonzero
repeat limits were rejected before execution. Child launchers passed shell syntax.

Records: `TASK/plans/unattended-once03-guards/result.json` and
`TASK/plans/unattended-once03-summary-check/result.json`.
Original queue guards, failed attempts, matrix01/matrix02 manifests and terminal
records remain unchanged in their original paths. The prior
[continuation evidence](evidence/unattended-resume-20261008.md) records matrix02's
original scope; it is historical and does not restore its cancelled repeat policy.
