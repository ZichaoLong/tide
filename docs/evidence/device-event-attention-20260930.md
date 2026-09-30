# Device event attention, GQA and windows — 2026-09-30

Clean `b668f2f` passes a full standalone build, four CPU CTests, all27 device
component cells and an independent CANN placement profile. The
[manifest](device-event-attention-20260930.json) records source/core/component
identities, raw artifact hashes and retained failures. This qualifies the existing
`memory="attention"` event-GQA semantics for single-device FP32 HARD inference.

Every actual event appends one compact KV row after Aggregate. Static head-geometry
groups support MHA/GQA/MQA; device-generated indices map query heads to compact KV
heads. Positive windows retain the declared tail including the current event;
window0 never implicitly discards keys. Proposed caches are adopted only under
observe-all or actual selection. Clear preserves comparison before emptying live
KV. Mixed fiber/event owners overwrite their own proposal rows without adding
stale scratch from a prior ready layout. Device caches survive lean window boundaries.

The new gate passes98 shape/analytic/cache anchors,192 complete windows,40
lifecycle windows and5 refusals. Coverage includes widths1/4/7/33/257, query/KV
head configurations, windows0/1/3, query chunks1/4, message-count versus event-count,
three Read modes, feedback/DAG, mixed fiber/event profiles, InputOrigin, large int64
counters, legal periodic clocks, selected-only adoption, clear, restore, window
recycling and lean continuation. Complete tensors retain rtol1e-5/atol1e-6 and
discrete observables remain exact.

Profiling records92,569 AIV and2,530 AI Core tasks, including device event planning,
QKV placement, cache retention/commit, batched projections, gather, transpose and
softmax. No AiCPU task or CPU fallback diagnostic was observed. The trace includes
construction, exports and CPU assertions; this establishes placement, not throughput.

The initial development fixture used a semantic documentation name as a configuration
value and was corrected. Later diagnostics isolated unstable output to32 AIV blocks
writing adjacent scalar metadata in shared cache lines. The final indices launch
uses one metadata writer, matching the existing fiber path. Matrix/vector payload
work remains packed. Original failed snapshots/logs are retained; neither formulas
nor numerical tolerances changed.

At this revision attention exposes one complete region-time frame per stage.
Full node-time attention batching, key-axis tiling, complete memory planning,
FP16 flow, public presets/matrix, peer progression and resident backward/optimizer
remain separate obligations. This evidence does not certify training or speedup.
