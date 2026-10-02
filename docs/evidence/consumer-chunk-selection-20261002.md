# Aggressive operator chunk selection

Implementation **e82f97199c63dc8505137595adaae9a9c2d12221** passed all
[four clean qualification jobs](consumer-chunk-selection-20261002.json).
Audit: `TASK/launchers/chunk_planner_evidence.py <full implementation SHA>`.

When a requested configuration exceeds a card's usable memory, aggressive mode
now halves the single operator maximum that most reduces the summed positive
per-card peak excess. Ties use the same fixed field order in Python and C++.
Joint halving handles a plateau; conservative mode retains its old joint-halving
rule. Accepted plans use the **unchanged** memory envelope and safety margins.
No topology/input specialization, numerical prepass, scheduler change, logical
capacity reduction, dtype switch or changed loss/update boundary is introduced.
`row_selection` records `greedy_peak_excess` or `joint_halving`.

CPU13 tests include24 varied shape comparisons,32 constrained comparisons across
both policies, materialized parameter inventory, overflow/minimum-capacity refusal,
and automatic sample admission. They confirm exact Python/C++ plan equality,
unchanged placement/canonical owners and accepted peaks inside the original
per-card envelopes. A directed case preserves nonlimiting Full/aggregate batches
while shrinking the dominant reverse workspace.

NPU17 tests passed without skips:16 actual Add/Attention comparisons with an
independent CPU execution, plus one refusal before model allocation. Coverage
includes Python-native/standalone LibTorch, FP32/FP16, SGD/AdamW, complete continued
updates, automatic operator splitting and logicalB17 split into physical groups.
Every executed candidate passed allocator calibration and the existing state,
gradient, loss, parameter/update and continuation checks.

A separate two-card FP16 actual-consumer profile forced operator splitting under
an explicit1,839,217,549-byte incremental cap. Attention rows8→4 and reverse4→2;
other requested maxima stayed unchanged. Three physical sample groups completed
two AdamW updates. **65,727 operators, zero observed AiCPU**; the run is instrumented,
not throughput evidence. No original-width performance claim follows yet.

Only consumer planning/recording changed. The qualifiedb3a6a24 resident libraries,
core and CANN kernels are byte-identical. The installed consumer was freshly
linked from source/header/options-verified objects; no backend requalification
or unchanged representative matrix was repeated. Raw snapshots, manifests,
JUnit, allocation observations and profile CSVs remain in task artifacts.
