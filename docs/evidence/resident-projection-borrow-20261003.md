# Private frozen projection banks during sharded training

Implementation **44674936efcc4177d4683a7e8542816aec0928eb** passed
[eight clean qualification jobs](resident-projection-borrow-20261003.json).
Audit: `TASK/launchers/projection_borrow_evidence.py <full implementation SHA>`.
All jobs terminated with exit 0 and empty control groups; leases released.

Aggressive sharded training now borrows its private frozen emission projection
banks during a backward group. The session exposes no such banks, refuses
publication with outstanding windows, and closes all reverse programs/releases
tapes before an update. Ownership-mode changes, replaced/mutated sources and
alias structure are checked. The default retention overloads still create
independent copies; conservative and legacy single-device training retain their
previous behavior. Explicitly placed one-device sessions use the sharded path.
Attention/Full snapshots, state/KV/message records and all VJP bridges are unchanged.

Retained API budgets, existing byte counters and consumer admission estimates
remain unchanged and conservative. `borrowed_projection_bytes` reports an included
footprint, not an additional allocation or an allocator measurement. Public ABI,
class layouts, core and CANN kernels are unchanged. Affected retention archive
members and owner/backward units were rebuilt; clean linking reused development
objects only after source/header/options identity checks. This is not a new full
core or from-scratch CANN build claim.

Native **160 trajectories / 2,560 windows / 640 updates**, Python **16** and actual
consumer **32** tests passed without skips. Native FP32/FP16 trajectories compare
against independent CPU FP32/FP64, including projection emission, connected/None/
zero gradients, multiple updates and continuation. Python covers retained-capacity
boundaries, update/restore lifetimes and compact journals. Actual consumer cases
cover Add/Attention, FP32/FP16, sample slicing and complete updates through both
standalone LibTorch and the Python-owned adapter. Default snapshot isolation and
private-bank alias/mutation guards have separate directed checks.

One process per old/new binary ran on the same two-card lease: D512/B8/T4/V257,
128-body-node Attention, physical B2 ×4, two connected windows, one FP32 AdamW
update. Loss **7.532631874084473**, outputs, all prior statistics, continuation,
operator chunks and estimates are unchanged.

| Logical device | Previous allocator peak (bytes) | New allocator peak (bytes) | Reduction (bytes) |
| --- | ---: | ---: | ---: |
| 0 | 7,804,784,128 | 7,508,509,696 | 296,274,432 |
| 1 | 6,957,622,784 | 6,661,348,352 | 296,274,432 |

The measured reduction is about **282.5 MiB/card**. The included borrowed footprint
is 590,450,688 bytes across both cards. Separate FP16 profiling recorded **53,176**
operators: 50,628 AI_VECTOR_CORE, 723 MIX_AIV and 1,825 AI_CORE, with **zero observed
AiCPU operators**. No throughput recommendation or original-size training claim
follows; the next scale assessment retains its existing memory and cost gates.
