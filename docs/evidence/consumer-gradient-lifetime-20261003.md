# Consumer physical-gradient storage lifetimes

Implementation **789e1a56a8e9f72e20814dd863f614a7cf366df2** passed
[four clean qualification jobs](consumer-gradient-lifetime-20261003.json).
Audit: `TASK/launchers/gradient_lifetime_evidence.py <full implementation SHA>`.
All services terminated with exit 0 and empty control groups; leases released.

Aggressive multi-device training now charges physical projection and Attention
parameter gradients once per backward group, matching the qualified reuse in
233bf01 and 6b9224c. Conservative and legacy single-device training retain their
per-window charges. State/cache/message bridges, other adjoints, workspace,
canonical outputs, optimizer/accumulation budgets and safety margins are unchanged.
`projection_parameter_gradients` is an included component, not an additional
allocation. C++ and Python use the same lifetime rule.

CPU **23** and NPU **25** tests passed without skips. The CPU materialized inventory
covers both policies and one/three devices; paired C++/Python plans agree. NPU
coverage includes 24 executed candidates with independent CPU comparisons and one
pre-construction refusal: Add/Attention, FP32/FP16, SGD/AdamW, single/multiple
devices, automatic operator/sample splitting and continued updates. No candidate
consumes reference events, routes or gradients.

The two-card D512/B8/T4/V257 Attention calibration used physical B2 ×4, two
connected windows and one complete FP32 AdamW update. Loss **7.532631874084473**,
outputs, statistics, continuation and effective chunks match the previous result.

| Logical device | Previous estimate (bytes) | Corrected estimate (bytes) | Observed allocator growth, before and after (bytes) |
| --- | ---: | ---: | ---: |
| 0 | 25,138,936,228 | 24,575,275,428 | 7,804,784,128 |
| 1 | 17,266,531,620 | 16,702,870,820 | 6,957,622,784 |

This is an accounting correction; it produces no additional actual memory
reduction. The installed consumer was freshly linked using source/header/options
verified objects. Resident/core/CANN execution bytes match the qualified backend,
so no new profile or unchanged representative matrix was required. Different
leases preclude a throughput comparison. Original-width complete training and
full-size comparisons remain open.
