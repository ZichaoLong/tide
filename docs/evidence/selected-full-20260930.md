# Selected matrix Full on device — 2026-09-30

Clean source `5bf61e3bafaa4beda5f5ab8780e34f854582f62f` passed its standalone
build, four CPU CTests/loader checks, all 14 NPU component cells and a separate
Full placement trace. The [manifest](selected-full-20260930.json) records exact
source, core, binary, job and log identities. The existing standalone core was
reused only after matching its C++ source and binary fingerprints.

`PackedFull` gathers actual selected tanh actions into bounded physical chunks.
The NPU controls the cursor and subsequent chunks, executes FP32 KEEP_DTYPE batch
matmul, bias, tanh and content addition, and bulk index-copies results back.
Identity Full stays content. Comparison snapshots precede selected clear.
Padding uses zero sentinel inputs/parameters and distinct scratch destinations;
inactive owners do not enter Full arithmetic. No numerical CPU route prepass or
per-chunk host decision is used. See [the finite profile](../content-flow.md).

The 24 Full cases cover widths 1/7/33, chunk limits 1/4, empty and identity-only
selection, selected tanh, partial tails and inactive NaN parameters/values.
160 complete-window/continuation comparisons against independent CPU Streaming
and Greedy cover identity/tanh Full, feedback, unequal delays, parallel edges,
ragged/zero inputs, large exact int64 clocks/counts, active-only adoption, selected
clear and restoring the candidate's own cut with another schedule. They contain
788 events and 488 actual emitted messages; 20 windows form multi-time node batches.
Full observables agree at FP32 rtol 1e-5 / atol 1e-6; discrete data are exact.
Existing invalid-input retry and execution-failure poisoning checks also pass.

The Full component trace records 426 tasks: 410 AIV and 16 AI Core, including
48 planners, 16 BatchMatMulV2, 24 Tanh and 24 ScatterUpdate tasks. The width-one
matmul lowers to vector Mul. No AiCPU task or host-fallback diagnostic was found.
This trace includes setup, input updates, synchronization and CPU assertions;
it establishes placement, not full-flow latency, throughput or a CPU/NPU ratio.

Full chunk rows are configurable and subject to a local parameter/scratch estimate.
This does not qualify complete model/KV/training memory planning or aggressive-safe
chunking. The integrated profile remains FP32 inference with sum Aggregate,
content-mode linear Read, identity/EMA state, count/positive selection, adopt/clear
Next and broadcast delivery. Other modules/Read modes, FP16, public packaging,
peer progression, VJPs/optimizer and abnormal runtime teardown remain outside
this immutable source's qualification.
