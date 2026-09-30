# Resident normalized Aggregate training qualification

Exact implementation **0ba9fc62fb583036d6b906a2e182bf5982b0d1a7**, aarch64
Ascend910_9392, Torch/TorchNPU2.10 and CANN9.0.0. Clean standalone and
Python-owned builds passed. [The audit](resident-aggregate-training-20261001.json)
checks frozen source, matching core, binaries, loader closure, results, logs
and profiler CSV hashes. The Python plugin does not link the standalone SDK.

The standalone gate passed all 46 component cells and five CTests. Normalized
Aggregate adjoints passed 39 configurations/117 long, short and empty replays,
each compared with independent CPU FP32/FP64 autograd: 234 comparisons. They cover
mean, positive weighted mean, active softmax and all-source softmax; widths1/7/257,
logical domains up to257, physical/logical permutations, poisoned unused owners,
missing versus present-zero sources, zero physical scales with nonzero raw
messages, None versus connected zero, budget refusal and duplicate-source refusal.

Public C++ training passed 57 trajectories, 912 retained windows and 228 updates.
The cases include both schedules and optimizers, feedback/self-loops, mixed
Aggregate profiles, exclusive physical source aliases, shared coefficient/scale
owners, width257 and complete-cut checkpoint continuation. Every candidate runs
its own forward, backward and update. No CPU route, event trace or numerical
result enters candidate execution.

The Python client passed 106 device tests with no skips, including all three graph
families, actual NPU loss cotangents, disk and fresh-process continuation. Its CPU
interface gate passed 76 tests with 103 optional-NPU skips. These are C++/CANN client
results; they do not qualify an independent pure-PyTorch resident scheduler.

Aggregate trajectories use explicit AdamW epsilon 1e-5. The public default 1e-8
and ordinary FP32 tensor tolerances are unchanged; control probabilities use the
strict comparison and routes remain exact. The existing LH/SwiGLU regression
cell separately uses its explicit `conditioned` control policy, with the retained
numerical limits in [that qualification](resident-extra-full-20261001.md).
This does not convert prior strict failures into passes.

The separate complete Aggregate correctness profile observed 72,933
AI_VECTOR_CORE, 578 AI_CORE and 1,213 MIX_AIV records, including 1,153 Aggregate-VJP,
1,704 optimizer, 2,867 graph-reverse and 410 window-bridge records. No AiCPU record
or host-fallback diagnostic was observed. The trace includes construction and
CPU assertions; record counts and summed task time are not throughput or wall time.

This qualification extends single-NPU FP32 HARD training with
[normalized Aggregate adjoints](../resident-aggregate-vjp.md). Attention adjoints,
HST/SOFTP,FP16,peer progression and the required performance matrix remain
separate work. The audit retains the original development compiler failures and
the cancelled recursive multi-target build, including their nonzero terminal
codes. Development archive reuse was not used in place of these clean builds.
