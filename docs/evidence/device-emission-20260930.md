# Packed device slot emission — 2026-09-30

Clean `6445121` passes the independent standalone build/four CPU CTests, all
23 device component cells and a separate CANN placement profile. The
[manifest](device-emission-20260930.json) ties source/core/component identities to
the existing raw results. This qualifies single-device FP32 HARD inference only.

The device decides actual slot presence from selected actions and exact logical
time, then packs present slot-affine projections in bounded batches. Physical
edge/output scales apply after projection. Missing slots produce no records;
present zero values still produce messages. Full auxiliary values and unscaled
slot values are recorded separately, including zero-scale physical deliveries.
Internal edges and outputs both use batched gather/multiply. No numerical CPU
routing prepass or host per-event projection is used.

The new gate passes 16 component cases, 66 complete windows and 6 refusals:
widths 1/7/33/257, chunk limits 1/4, large int64 clocks, permuted local bindings,
parallel edges, zero scales, absent/inactive poisoned data, changed action times,
feedback/DAG, two inputs, both schedules, continuation and lean result export.
All-absent emissions retain state/history and skip affine work. Capacity and real
arrival overflow fail explicitly; absent edges do not cause spurious overflow.
Public model validation still rejects nonfinite parameters; poison is injected
below that boundary only in the isolated component test. Tensor comparisons keep
rtol1e-5/atol1e-6 and discrete/source/route observables remain exact.

The trace records 14,375 AIV and 176 AI Core tasks, including the slot planners,
gathers and BatchMatMulV2. No AiCPU task or host fallback diagnostic was observed.
Setup, output materialization and CPU assertions are included, so this is placement
evidence, not throughput. Development failures were a test helper name collision
and unsupported ATen bool initializer-list construction, plus dependent jobs;
original records remain failed. Corrected tests changed no formula or tolerance.
FP16, attention/KV, complete memory planning, public presets/matrix, peer progression
and resident backward/optimizer remain separate obligations.
