# Retained device backward qualification

Exact implementation **c2423f042c242a5a9ab893c04cf7962a794e97a8**, clean standalone
C++/LibTorch2.10/CANN9.0.0 on aarch64 Ascend910_9392. The accompanying
[JSON audit](device-retained-20261001.json) records source, binary, loader, logs
and raw profiling CSV hashes. All three durable jobs terminated with exit0:
`build-retained-clean01`, `retained-clean01`, `retained-profile-clean01`.

Four build CTests and all41 device component cells passed. The retained checker
compares26 trajectories/104 windows against independent CPU FP32/FP64 Streaming
autograd: both schedules, feedback/self/parallel edges, widths3/257, large int64
coordinates, an empty window, isolated/combined/None/zero roots, parameter aliases,
initial-state and external-input gradients. Saved tapes survive closing and
poisoning the forward owner. Replay, budget and missing-boundary refusals pass.

Profiling observed49,324 AI_VECTOR_CORE tasks,838 AI_CORE tasks and832 MIX_AIV
tasks, including314 window-bridge records. There were no AiCPU tasks or host
fallback diagnostics. This scope includes checker construction, forward and CPU
assertions; summed task time is not wall time and this is not a throughput result.

The [retained contract](../resident-retained.md) remains restricted to the HARD
sum/broadcast, identity/EMA/Add-state, identity/tanh-Full VJP. It does not certify
public training ownership, other adjoints, FP16, peer progression or full-size
performance. Each retained tape currently copies its parameter banks.
