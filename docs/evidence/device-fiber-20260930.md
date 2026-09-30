# Bounded same-fiber attention — 2026-09-30

Clean `081f567` passes a standalone build, four CPU CTests, all 25 device
component cells and a separate CANN placement profile. The
[manifest](device-fiber-20260930.json) identifies source/core/component hashes,
raw records and retained development failures. This qualifies single-device
FP32 HARD inference for `lh-fiber-attention-sum-repeat-v1` only.

The device plans actual per-source weighted rows in local slot order, packs
QKV/query/output chunks and runs matrix multiplication, gather, head permutation
and softmax. Each query sees all old keys and all current fiber keys. Persistent
key/value/log-bias arenas and int64 lengths remain on-device. Clear, selected-only
adoption and pre-clear comparison preserve their declared behavior. Optional
journals expose all four state-slot views for independent comparison; lean windows
can continue without downloading live KV. No numerical CPU routing prepass is used.

The new gate passes 16 analytic/shape anchors, 192 complete windows, 6 refusals
and 40 lifecycle windows. It covers widths 1/7/33/257, heads 1/3, chunk sizes 1/4,
slot permutation, InputOrigin, feedback/DAG, all three Read modes, vector/scalar
Read/state, selected-only/empty selection/clear, large int64 clocks/counters,
periodic phases, restore/lean export and repeated cache recycling. Tensor
comparisons retain rtol1e-5/atol1e-6; discrete observables remain exact.

The trace records 64,617 AIV and 1,984 AI Core tasks, without observed AiCPU
or host fallback diagnostics. It includes setup, export and CPU assertions;
this is placement evidence, not a throughput comparison.

Attention currently allows one complete time frame per region per device stage;
independent owners and message rows are packed. This explicit adapter fallback
does not deliver full attention node-time batching. Query chunks retain the
complete denominator; key-axis tiling remains pending. Cache, trace and exact
repeat-work capacities fail explicitly. Other pooling profiles, event-GQA/window,
FP16 flow, resident backward/optimizer, multi-card progression and the public
matrix remain separate obligations.
