# Single-process queue amendment — 2026-10-08

The user approved retaining first-process coverage and cancelling automatic
FP32/FP16 repetitions. Shared-server observations are accepted without
contention-triggered retries. Correctness, complete execution, resource admission,
warmup, per-child timeouts and disk reserves remain unchanged. This updates the
current [execution contract](../execution-flows.md); historical measurements and
plans retain their original scope.

The new `unattended-matrix03` service was verified active in `background.slice`,
independent of `focus.service`, at the timestamp in the [machine record](unattended-single-process-20261008.json).
It is draining current `formal-bound-auto01-cell113-r1` (Settle/Python/streaming/Add
complete training, resident FP32). The old coordinator alone is stopped; every
observed process in the child's cgroup remained unstopped. The11-card lease and
physical-to-logical mapping are retained. This is running work, not a passing result.

After the child ends, the handoff verifies its terminal receipt/empty cgroup,
retires matrix02, and runs31 remaining first-process jobs using the same clean
workload e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8 and controller
103f5b6c8e9b2e6e46a7f2733185433de2cb5f3f. The unchanged strict auditor decides child
acceptance. All248 conditional repeat slots are removed (at most168 executions).
No completed or failed first attempt is repeated. At handoff:58accepted bound
FP32 cells,30failed,one running,31unstarted; four FP16 first processes already exist.

The old manifest, source and819 frozen inputs still match. New task-local scripts,
plan, launcher and provenance are separately hashed; no live input was edited.
Two real, resource-free transient-service guards passed: natural drain without
submitting the next slot, and cancellation that leaves all owned cgroups empty.
The child heartbeat continued while its coordinator was stopped. No model or NPU
was used. Summary checks preserved all66 prior accepted process metrics (58bound
and8historical unbound, kept separate) and all36 partial-group records. Invalid
repeat jobs and nonzero repeat limits were rejected before creating an output.
The31 remaining child launchers and new shell entry passed syntax checks.

The single-process summary labels engineering observations and shared-server
uncertainty; it does not claim stable statistical rankings. Existing three-process
counts remain factual fields, with no requirement to fill them. At most four
concrete decision-driven follow-ups may be considered at review; none is queued.
Protective aggregate ceiling807265s is not an ETA. Before this switch, remaining
first-process work was estimated at about160h versus about495h with routine repeats.

Lifecycle and failure cleanup are handled by the independent service, including
ExecStopPost during drain. The existing child keeps its own time limit. Stopping
matrix03 intentionally stops the owned current work, including the predecessor
during drain. No protected historical CPU worker or unrelated service was signalled.
No automatic callback, retry or reboot recovery is configured.

Exact commands, record paths, inspection and stop instructions are in
[unattended measurements](../unattended-measurements.md). The machine record lists
artifact hashes under `TASK=/mi/data2T/zlong/tide-execution-flows`.
F6/F7 final review remains open; queue submission is not overall completion.
