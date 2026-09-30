# Content-driven device inference loop — 2026-09-30

Clean source `4d2f09ed55b2714cd66fba7f037b650e3dad1f49` passed its standalone component build,
four CPU CTests/loader, all13 NPU component cells and a separate msprof placement
run. The [manifest](content-loop-20260930.json) records source/core/binary and
terminal-job hashes. The core from clean eff5945 was reused only after source and
binary fingerprint checks. Original failed development build remains retained.

This is the first tested numerical loop that generates **actual feedback messages**
and discovers subsequent work on device. Its finite profile is sum Aggregate,
content-mode linear Read, identity/EMA state, count-v1/positive-v1 selection,
observe-all/active-only adoption with optional clear, and identity broadcast Full.
FP32 inference only, no autograd. See [the profile contract](../content-flow.md).

The80 full-observable comparisons against independent CPU Streaming and Greedy
cover positive-delay feedback, parallel edges, unequal delays, DAG/disconnected
structures, two ragged samples, present-zero messages, count histories above2^55,
large int64 clocks, continuation, empty windows and restoring the candidate's own
cut with the other scheduling policy. There were394 events and244 actual emitted
edge messages;10 windows formed batches with multiple times for the same node.
Outputs, messages, pending, states, clocks/counts, history, routes, source
contributions and numeric intermediate event snapshots agreed at FP32 tolerances
(rtol1e-5/atol1e-6); discrete data were exact.

Invalid input is retryable before submission. Capacity or stage exhaustion
poisons the execution object; the caller must restore an earlier complete cut.
The gates also reject unsupported modules and an insufficient declared buffer
budget. All per-stage queue/history/state/log preflights precede commits.
Per-event diagnostic history maps are reconstructed at the boundary from device
activity records; final history is independently read from device storage.

The separate trace contains120 content/state/selection/broadcast/output stages,
204 closure/ready stages,444 queue proposals and720 journal tasks. All11606
recorded device tasks were AIV, without a host fallback warning. Profiling includes
construction, uploads, CPU references/assertions and diagnostic export. It does
not establish latency, throughput or a CPU/NPU speed ratio. Numerical loops still
use scalar AIV code; placement alone is not efficient accelerator computation.

This evidence does not cover tanh or other Full modules, attention, proposal/old
Read, input-origin projections, FP16, public Python integration, peer progression,
training/VJPs/optimizer, full-scale memory chunking, or abnormal runtime teardown.
New selected matrix-Full development is outside this immutable source's scope.
