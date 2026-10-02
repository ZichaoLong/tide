# Original-width Add after parameter-gradient reuse

Clean implementation **789e1a56a8e9f72e20814dd863f614a7cf366df2** completed one
nine-card LibTorch resident TimedDAG/prefill FP32 SGD pilot. The [audited record](original-width-add-gradient-lifetime-20261003.json)
pins source, binaries, packets, static plan and terminal results. Audit:
`TASK/launchers/wide_add_gradient_lifetime_evidence.py <full implementation SHA>`.
The [consumer qualification](consumer-gradient-lifetime-20261003.md) supplies
independent small-model correctness; this pilot adds original-width evidence.

The unchanged original workload has 480 body nodes, 2,208 body edges,
D2048/B512/T12/V50304 and **9,468,053,696 parameters**. The pilot changes only B512
to B4 and uses two physical B2 groups, two connected windows and one complete
update, with eight ATen CPU threads and no warmup or profiler.

The generic planner uses original B512 geometry without executing values or
events. Previously exercised physical-B2 capacities are queue/arrivals896,
outputs64, trace3072, KV256 and KV-trace8192. Effective rows are Full16,
emission8, aggregate8, attention8, keys128, reverse2 and head64. The per-card cap
is still 60 GiB with **53.875 GiB usable**; head and context limits are 4 GiB.

| Measurement | Observed |
| --- | ---: |
| Construction, separately timed | 84.305 s |
| Complete B4 update | 21.179 s |
| Maximum incremental card allocator | 47,394,238,464 bytes |
| Loss | 31.58603858947754 |
| Outputs / logical events / final cut | 96 / 9,265 / 408 |
| Pending peak / maximum window events | 768 / 2,335 |

All nine cards passed allocator calibration; all context pools remained within
budget. Parameter count, outputs, logical events and cut match the previous
nine-card B4/physicalB1 pilot exactly. Loss differs by 0.0000019073486328125,
inside the existing FP32 atol=1e-6, rtol=1e-5. Physical grouping legitimately
changes packing, retention and reuse counters. This run has no full-size CPU
numerical oracle; no CPU-fallback warning was observed, which does not replace
a device profile or establish AiCPU attribution.

The unchanged cost gate projects **21.179427721 × 128 × 1.15 = 3,117.612 s**,
above **3,000 s**. **B512 was not launched.** This is a conservative estimate,
not measured B512 latency. The previous pilot was 27.300 s, but different
physical grouping, chunks and leases preclude a causal speed/memory claim or
formal recommendation. The job terminated successfully and released all leases.
Further work must reduce costs or storage before a new bounded scale assessment;
repeating this run or relaxing its gate is not completion.
