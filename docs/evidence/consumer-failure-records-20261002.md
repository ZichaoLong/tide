# Keep measurements when allocator calibration rejects a run

Implementation `3462dae` is qualified by the [audited fixed-source record](consumer-failure-records-20261002.json):
19 CPU checks, 9 NPU checks and one independently installed/linked C++ consumer
build, all passed without skips. The resident backend remains the qualified
`38858d0` binary. Audit: `TASK/launchers/failure_records_evidence.py`.

Previously, a post-run allocator peak above the estimate produced a minimal error
record and discarded collected measurements. Both consumers now retain every
device's estimate and observed peak, phase samples, completed-step timings,
losses, work counters and final cut. They still exit unsuccessfully and record
`state=failed`, `failure_phase=post_run_memory_calibration` and
`allocator_within_estimate=false`. This is a post-execution check; it does not
roll back completed updates. No planner estimate, physical cap, safety margin,
kernel, scheduling or numerical behavior changed.

CPU tests cover a miss on the first or later card, previously freed peak storage,
cross-language observations, the Python durable writer and the native process
boundary. The latter uses a deliberately failing executable: it tests record
propagation, not NPU computation. Incorrect workload identities, a success record
accompanying a failed process, malformed JSON and non-object JSON are refused.

The NPU failure test first runs safely admitted real two-card Attention training
for two complete AdamW updates, with an independent CPU loss/output/cut reference.
Only at the post-run boundary does it inject an underestimated planned peak.
The resulting failed record retains all observations; a fresh owner then completes
the same work with the unchanged estimator. Other device cases cover actual
LibTorch/Python-native phase recording, mixed execution, successful CLI results
and ordinary capacity refusals. They do not certify the original-size estimator.

The earlier [wide Add refusal](original-width-add-b2-refusal-20261002.json) remains
failed, with its missing measurements explicitly documented. Natural large-scale
underestimation and any subsequent estimator correction require separate evidence.
