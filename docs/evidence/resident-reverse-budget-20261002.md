# Resident reverse budget qualification

Implementation `106cbeb293847e054525fce21f68a8d1c8a13b13` passed eight immutable-source
jobs. The [machine-readable audit](resident-reverse-budget-20261002.json) records
source identity, recursive build dependencies, installed clients, raw results
and the independent profile. No skips or numerical tolerance changes occurred.

| Gate | Result |
| --- | --- |
| Standalone, Python-owned backend and installed client | Three successful builds/links; unchanged objects and kernels reused only after recursive source/header and byte checks |
| CPU reservation boundaries | 1 passed, including D2048 geometry without large allocations |
| Public event/fiber/FP16 training | 83 passed |
| Retained model matrix | 16 trajectories per dtype, 512 windows and 128 updates in total |
| Actual model consumers | 21 passed across both clients, including two D32 automatic-splitting cases |
| Separate two-card D32 Attention training profile | Four continued windows, two complete AdamW updates |

The planner retains the existing disjoint tensor budgets. It chooses physical
owner/query/key capacities before allocation; it does not truncate logical
fibers, KV visibility, loss reduction, retained windows or update boundaries.
A smallest complete owner that cannot fit still fails explicitly.

For the actual D32 Attention workload, requested maximum16 automatically selects
owner12/query12/key64. Both clients match explicit one-row execution and the
independent CPU full-state, gradient and update reference. Per-step estimates
across four retained window/owner groups total20,267,360 bytes within assigned
budgets44,739,240 bytes. These statistics describe constructed buffer capacities,
not actual active rows or allocator peak memory. Public gradient ABI changed;
clients and bindings must be rebuilt.

The separate trace observes14,782 AI_VECTOR_CORE,607 AI_CORE and218 MIX_AIV tasks,
with **no observed AiCPU**. It contains32 model executions,570 notification
pairs,2,318 device switches and4,552 asynchronous copies. Construction is included;
this is execution-placement evidence, not formal throughput or a complete
bottleneck-time attribution. The earlier D32 reverse16-row refusal remains
preserved as a failure in the audit.

This qualifies local safe reverse splitting and its FP32/FP16 library adjoints
plus actual FP32 consumers. Compact projection ownership, total-memory admission,
consumer FP16, public multi-device inference and full-size performance remain
outside this result.

Reproduction uses the retained task-local `launchers/reverse_budget_evidence.py`
with the full implementation hash. Runs are `build-reverse-budget-clean01`,
`build-reverse-budget-python-clean01`, `build-reverse-budget-consumer-clean01`,
and `reverse-budget-{cpu,cache,matrix,consumer,profile}-clean01` under the project
artifact root. The audit checks terminal status, source identity, build reuse,
test totals, split statistics, traces and unchanged tolerances.
