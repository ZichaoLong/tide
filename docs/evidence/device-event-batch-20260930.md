# Device event-attention node-time batches — 2026-09-30

Clean `26aa09f` passes the full standalone build, four CPU CTests, all 30 device
cells and independent CANN profiling. The [manifest](device-event-batch-20260930.json)
records source, component/core identity, raw result hashes and placement counts.
This qualifies single-device FP32 HARD inference within the content-flow profile.

The device forms legal node-time prefixes from actual inputs, queued messages and
closure. Observe-all regions without selected clear prepare multiple times together.
One device loop packs all actual QKV rows; another packs queries and projections.
Immutable old KV plus one compact new row per actual event preserves each query's
causal prefix and sliding window. Only the final owner tail commits; intermediate
old/proposed states remain observable. Selected-only adoption and selected clear
retain complete single-frame fallback. No CPU numerical route prepass is consumed.

The new check passes 72 analytic/restore cases and 216 general windows/restores.
The analytic cases require four real node times in one device stage, with widths
4/33/257, GQA, windows 0/1/3, query chunks 1/4 and full/tiled keys. Independent
uniform-attention formulas check each query's denominator, including a stage longer
than its cache window. Restore changes physical chunking and scheduling policy.
Feedback, DAG and isolated graphs exercise all three Read modes and adoption/clear
policies, exact large int64 clocks/counts, pending state and full trace comparisons.
Eighteen general windows actually form multi-time batches. All comparisons retain
FP32 rtol 1e-5/atol 1e-6 and exact discrete membership/order/history checks.

The separate trace records 97,088 AIV and 3,075 AI Core tasks. No AiCPU task or CPU
fallback diagnostic was observed. Setup, assertions and result exports are included;
these counts establish execution placement, not throughput or a CPU speed ratio.
The 30-cell regression scope contains earlier separately qualified FP16 components;
the complete content flow remains FP32 only.

Fiber node-time batches, other module adapters, the public five-preset matrix,
peer progression and resident backward/VJP/optimizer remain separate work. F1–F7
are not complete. Earlier component qualifications and failures retain their scope.
