# Per-device saved continuation pool

Qualified source `48e44b03ec592523d4f5e78980c5a70b86156215`.
[Audited receipts](resident-context-pool-20261002.json),
[consumer contract](../online-consumers.md), [API contract](../resident-training.md).

`--resident-context-bytes` enables compact snapshots and limits **all simultaneously
saved sample continuations per NPU**. The consumer subtracts the other live handles
before each save; the library checks total and per-device budgets before payload
copies. A too-small budget fails explicitly. The active owner, metadata packing
workspace and vendor allowance are separately charged. Zero preserves dense saving.
The offline planner accepts the same sample-size and pool controls.

The fixed-source qualification passed CPU17, Python/native consumer64, and four
standalone dense/compact × FP32/FP16 cells:32 trajectories,768 windows,96 updates.
There were no skipped tests. Checks include exact per-device limits and one-byte
refusal, independent CPU continuations/gradients/updates, all three graph families,
both schedules, ragged sample tails, connected windows, None/zero and continuation
across shared parameter updates. The unchanged FP16 policy separates independent
CPU FP32 rounding from whole-vs-sliced FP16; no tolerance was relaxed for this change.

Both runtime owners were freshly linked from source/options-checked development
objects and byte-verified unchanged core/CANN archives. The installed standalone
client was separately generated, installed, relinked and loader-checked. Qualification
ran the clean immutable source; the development snapshot is not qualification.

A separate two-device actual Attention consumer trace (three sample ranges, two
complete AdamW updates, FP32) recorded46,432 Vector,1,797 AI_CORE and607 MIX_AIV
operator rows; no AiCPU rows or detected CPU fallback. This is path evidence,
not a throughput comparison. Dynamic nonzero output extents synchronize at the
explicit detached snapshot boundary; online event decisions remain device-owned.

One fresh process per storage policy used the same D128/B8/T4/V257 Attention,
four B2 ranges, one FP32 AdamW update and two connected windows:

| Observation | Dense | Compact pool |
| --- | ---: | ---: |
| Declared saved-state budget per card |144,522,432 bytes|8,388,608 bytes|
| Peak live saved tensor/index bytes |144,506,048|2,755,300|
| Total allocator peak bytes |1,975,361,024|1,868,793,344|

The allocator peak decreased5.39% (1.840→1.740GiB); saved tensor/index storage
alone decreased98.09%. These are different quantities. Both runs produced
loss5.612767696380615,3,145 events,64 outputs and the same final cut; allocator
peaks stayed within their complete-consumer estimates. The observation calibrates
this shape and stack, not universal memory bounds or throughput. Construction,
packing, saving and restore costs remain in the corresponding consumer timers.

Live KV and retained reverse tapes remain dense. Automatic sample-size admission,
original full-size execution/comparisons, CUDA hardware and other CANN tuples are
still pending. A smaller saved-state pool does not by itself close full-size training.
