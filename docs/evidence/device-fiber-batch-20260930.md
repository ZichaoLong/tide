# Device fiber-attention node-time batches — 2026-09-30

Clean `c83aec3` passes the full standalone build, four CPU CTests, all 31 device
cells and independent CANN profiling. The [manifest](device-fiber-batch-20260930.json)
records exact source, component/core identity, raw hashes and placement counts.
This qualifies single-device FP32 HARD inference within the content-flow profile.

Observe-all regions without selected clear admit multiple legal node times from
actual inputs, queued messages and closure. Device planning appends each real KV
message once, preserves per-event prefix lengths and repeated per-tick FP32 bias
decay, and exposes every key in the current fiber to every current query. It does
not impose a triangular mask within that fiber. Only the final owner KV/bias/length
commits; intermediate old/proposed values remain observable. Selected-only and
selected-clear regions retain the complete single-frame fallback. No CPU numeric
route prepass is consumed. Bounded per-event bias scratch enters the memory budget.

The new check passes 96 analytic/restore cases, 480 general windows/restores and
three repeated-decay saturation cases. Thirty-one general windows actually form
multi-time batches. Widths 1/33/257, initial cache lengths 0/257, query chunks 1/4,
full/tiled keys, periodic clocks, all five pooling formulas, three Read modes,
feedback/DAG/isolated/self-loop graphs, large int64 coordinates/counts, restore and
lean boundaries are covered. FP32 rtol 1e-5/atol 1e-6 and exact discrete checks are
unchanged. Intentional saturated negative-infinity bias is compared exactly.

The separate clean trace records 221,788 AIV and 8,181 AI Core tasks, with no AiCPU
task or CPU fallback diagnostic. Counts include construction and assertions and
are observations of this trace, not a throughput or CPU speed comparison. Earlier
FP16 component checks remain separately scoped; the complete flow is FP32 only.

Normalized Aggregate, remaining module adapters, public presets and matrix, peer
progression and resident backward/VJP/optimizer remain separate work. F1–F7 are
not complete. Development and historical failures retain their original records.
