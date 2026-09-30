# Device attention key tiling — 2026-09-30

Clean `7b03614` passes a full standalone build, four CPU CTests, all28 device
component cells and an independent CANN placement profile. The
[manifest](device-key-tiling-20260930.json) records exact identities, raw artifact
hashes and retained failures. This qualifies physical key splitting for the
existing fiber and event attention adapters in single-device FP32 HARD inference.

`attention_key_rows` bounds physical gathered keys independently of logical cache
capacity and window semantics. Device metadata chooses each actual key interval
and GQA head mapping. Batched matrix multiplication computes scores and weighted
values; AIV vector operations carry the global maximum, denominator and weighted
sum across tiles. The final division uses the complete denominator. Every query
still sees all keys of its current fiber. Real KV/cache contents are never dropped.
A bound covering capacity retains the complete-key path; vector key tiles are at
most256 rows, and remaining scratch budget can further reduce key/query rows.

The new gate passes96 complete windows/restores and12 extreme/saturation/budget
cases. It covers widths1/4/33/257, MHA/GQA/MQA, ragged old KV lengths0/5/257,
key limits1/7/128/300, complete-key comparison, exact uniform-denominator anchors,
extreme logits and changing physical tile size on restore. Existing event,
fiber and five-pooling gates now exercise key sizes1/7, retaining their feedback,
Read, clear, selection, InputOrigin, large int64 and lifecycle checks. Complete
finite comparisons keep rtol1e-5/atol1e-6 and exact discrete observables.

Finite repeated decay may saturate old log bias to negative infinity. A tile
containing only such old logits contributes no mass; later finite current keys
still normalize correctly. Empty padding is distinct from an actual completely
zero-mass query. The dedicated saturation test checks exceptional bias slots
exactly and all remaining observables through the unchanged finite comparator;
it leaves raw results intact. Earlier fixture/count and comparator failures are
retained. No numerical tolerance was relaxed.

The placement trace records96,157 AIV and4,780 AI Core tasks, including the tile
planner, online softmax, weighted-sum merge, gather and batched matmul. No AiCPU or
CPU fallback diagnostic was observed. The trace includes setup, CPU assertions and
exports. It certifies placement, not throughput or peak HBM. Actual tile counts
and real/padded query-head-key work are reported separately from QKV/output chunks.

The bounded adapter still admits one complete attention region-time frame per
stage. Complete calibrated memory planning, node-time attention batching, FP16
flow, public presets/matrix, peer progression and resident backward/optimizer
remain separate obligations. No overall F1–F7 completion or speedup is claimed.
