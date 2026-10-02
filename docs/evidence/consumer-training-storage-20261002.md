# Consumer training storage lifetimes

Implementation **4dd8368159ff74e8b45c052cfe85ecbca1e294f2** passed
[four clean qualification jobs](consumer-training-storage-20261002.json).
Audit: `TASK/launchers/training_storage_evidence.py <full implementation SHA>`.
All services terminated successfully with empty control groups; leases released.

The consumer charges shared immutable Attention parameters once across retained
windows. Dynamic KV/log-bias, lengths, journals and state remain per-window.
For sample accumulation it charges the one extra live private FP32 numeric bank.
The separate two-input `accumulate(max_bytes=...)` limit is unchanged, as are device
headroom, safety factors, logical capacities and update boundaries.
`retained_attention_parameters` is part of `retained`, not an additional allocation.
`gradient_accumulation_live` distinguishes process storage from the unchanged
`gradient_accumulation` API limit.

CPU **17** and NPU **25** tests passed without skips. CPU cases include materialized
Add/Attention inventories in FP32/FP16, one/two/three retained windows, and exact
Python/C++ plan agreement. NPU coverage comprises 24 executed candidates with
independent CPU comparisons and one preallocation refusal: automatic operator
splitting, automatic sample admission and sliced continued updates in native
Python/standalone LibTorch, FP32/FP16, SGD/AdamW and one/two devices. Conservative
and aggressive policies are covered. Actual retained Attention bytes and allocator
peaks stay inside corrected estimates.

A two-card D512/B8/T4/V257, 128-body-node Attention calibration used physical B2 ×4,
two connected windows and one complete FP32 AdamW update. It reused the immutable
input packet from the qualified private-accumulation experiment; prior observations
were compared only after execution. Loss **7.532631874084473**, outputs, statistics,
continuation, physical groups and effective operator chunks are exactly unchanged.

| Per logical device | Previous estimate (bytes) | Corrected estimate (bytes) | Observed allocator growth, before and after (bytes) |
| --- | ---: | ---: | ---: |
| 0 | 25,965,880,344 | 25,138,936,228 | 8,368,268,800 |
| 1 | 18,093,475,736 | 17,266,531,620 | 7,520,954,880 |

This change corrects static accounting; it does **not** reduce actual allocations.
Different leases preclude a throughput comparison. Resident/core/CANN bytes match
the previously qualified backend. The installed consumer was freshly linked using
source/header/options-verified objects. No new profile or unchanged representative
matrix was needed for this estimator-only change. Original-width complete training
and full-size comparisons remain open.
