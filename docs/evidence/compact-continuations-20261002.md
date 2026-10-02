# Compact saved device continuations

Qualified source `f8cc0526466bcd3818eb63eb2ff769732e157bde`.
[Audited receipts](compact-continuations-20261002.json),
[public contract](../resident-training.md).

Optional `snapshot_device(max_bytes=..., compact=True)` preserves complete
same-owner detached continuation using batched NPU selection/gather of valid
pending rows and cache prefixes. Shared int64 indices preserve physical positions;
unique-index scatter restores them and resets unused padding/sentinels. Zero-valued
valid rows remain present, and dense groups are retained when indices would cost
more. Parameters, optimizer and accumulated gradients remain shared by the owner.
Default dense saving and existing one-argument C++ entry points remain available.

Dynamic nonzero output extents synchronize at the explicit snapshot boundary.
Row indices, length vectors and numerical payloads stay on device; online event
scheduling does not gain a host branch. Saved-tensor budgets include indices;
metadata packing workspace and all other live handles must be budgeted separately.
This implementation compacts opaque snapshots, not live KV or retained tapes.

Two clean runtime builds, two affected gates, an independent profile and a memory
observation passed. Exact source/options-checked development objects were reused,
affected archive members replaced and public libraries freshly linked. Core and
CANN kernels are unchanged; standalone/Python loader closures remain separate.

- C++: dense and compact FP32/FP16,32trajectories/768windows/96updates. Independent
  CPU FP32/FP64 schedules check complete forward/VJP/optimizer state, event/fiber
  caches, different input streams, one shared update, explicit owner maps,
  HARD/HST/SOFTP, both schedules, SGD/AdamW and None/connected-zero gradients.
- Python:18tests, no skips, covering both storage policies across all families,
  one/two devices, ordinary/identity/Add/event/fiber memory, independent
  continuations, shared training updates, ownership/capacity refusals and snapshot
  restore across parameter updates. Failed capacity admission preserves state.
- The separate two-device FP32 training profile observed26,322Vector Core,
  350AI Core and387MIX_AIV tasks, zero observed AiCPU tasks and no fallback warning.
  It includes construction and assertions and is not timed-throughput evidence.

One fresh Python-owned process per storage policy ran the same D128/B8/T4/V257
Attention packet, four independent two-sample continuations and two inference
windows each. Both produced64outputs and identical event totals/cuts. Saved tensor
storage fell from144,506,048bytes(137.81MiB) to2,755,300bytes(2.63MiB),98.09% less.
Observed allocator growth above the active owner fell from144,533,504 to3,945,472
bytes(97.27% less). This measures additional saved-state storage, not the total
process/owner peak or complete-training memory, and is not a throughput claim.

Two initial observation-script failures remain recorded: backend registration
order, then an unsupported `ResidentLimits` argument. Both failed before model
execution; the corrected script passed without changing production source.

Consumers still need compact-pool admission and simultaneous-live accounting.
Dense live KV/tapes, original wide execution, full-size comparisons, CUDA and
other CANN versions remain outside this qualification.
