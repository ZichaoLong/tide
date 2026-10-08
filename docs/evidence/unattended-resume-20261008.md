# Finite unattended continuation after disk reserve stop

The user authorized direct continuation on2026-10-08,without investigating storage
growth. This changes no graph/model semantics or existing experiment policy.
[Machine-readable observation](unattended-resume-20261008.json).

The first manager stopped with exit1 at2026-10-07T08:47:16.271693Z before launching
its next child. Root free space was8353996800bytes(7.78GiB),below the8GiB reserve.
Its cgroup is empty;its manifest,receipts,results and audit failures remain intact.
The stop was not caused by the agent session disconnecting.

At that stop,the same-series FP32 matrix had58/120accepted first processes,
30failed and32unstarted. The stopped queue contributed52passes/28failures;
prior jobs contributed6passes/2failures. One policy skip reused an already failed
cell rather than rerunning it. Nineteen failures were device admission without
model execution. Four FP16 companions each have one accepted process;no second
or third processes had begun. Eight older unbound FP32 observations stay separate.

The new manifest`TASK/plans/unattended-matrix02.json` uses the unchanged manager,
controller103f5b6 and workloade69b3bd. Preflight verifies819immutable inputs,
85terminal audit/receipt pairs and empty child cgroups. It partitions the120cells,
imports accepted evidence,retains failures and rejects existing future run paths.
All280unchanged child launchers and the new manager launcher pass`bash -n`.
The manager's actual startup summary reconstructs58bound plus8separate unbound
cells,and one process for each of the four FP16 companions.

Remaining slots:32first-process FP32 jobs,240conditional FP32 repeat slots
(at most160executions),and8conditional FP16 repeats. No old profile is repeated.
There are at most200new heavy executions. Prior active time is charged against
the original finite allowance;the smaller remaining bound is3077240s,not an ETA.
Per-child allowances,120s device waits,solo timing and24/8GiB disk reserves stay.
No failed-slot retry,automatic budget increase,new tracking layer or cleanup.

`tide-execution-flows-unattended-matrix02.service` and its first child are verified
active/transient in`background.slice`,outside`focus.service`. The child is
`formal-bound-auto01-cell113-r1`:Settle/Python/streaming/Add complete training,
FP32 resident,11physical NPUs mapped to logical0..10. It is running,not passed.
The queue keeps the original conditional repeat selection and failure-stop rules.

`TASK=/mi/data2T/zlong/tide-execution-flows`. The preparation,validation and
submission records are`launchers/prepare_unattended_resume02.py`,
`plans/unattended-matrix02-validation.json` and`plans/unattended-matrix02-submission.json`.
Manager outputs:`runs/unattended-matrix02/{status.json,task.log,dispatch/result.json}`.
Resolved command,cwd,source and resources are in the submission record.
Tracking remains project-owned records;no Trackio service is added under the
user-aligned minimal recording contract. Inspection/stop commands are in
[the queue contract](../unattended-measurements.md).

This verifies continuation/submission,not completion,throughput,training quality
or new device support. F6/F7 stay open. Selector extensions wait for updated
upstream documents;the protected historical CPU worker remains untouched.
