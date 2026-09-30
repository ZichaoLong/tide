# Device control and readiness — 2026-09-30

Clean source `5c5b58275e78b3fc8643ccd488bf7d6eb29985e7`. The
[manifest](device-control-20260930.json) binds terminal records, binary/source
hashes and profiler summaries. All three jobs passed on the standalone
aarch64 LibTorch-NPU2.10/CANN9.0.0 stack, Ascend910_9392.

The clean component build passed four CPU CTests and standalone loader checks.
The device gate passed eight runtime-control cases, four packed arithmetic
cases each in FP32 and FP16, eager queue checks and60 Ascend C readiness cases.
Checks cover changing inputs/limits, continuation, capacity refusal, physical
parallel edges, missing/zero messages, exact int64 above2^55/near its maximum,
empty/right-boundary work and malformed live coordinates. One physical device
was leased and exposed as logical npu:0.

The separate msprof gate recorded60 `tide_closure` tasks on AI_VECTOR_CORE.
All20 AI_CPU tasks were `OnesLike` construction initializers. The complete trace
contains390 device tasks and includes setup and CPU assertions; task-time sums
are not throughput. No host CPU fallback warning appeared. The eager int64
queue sorting gate reports device-local AiCPU sorting; the separate tensor
scatter-reduce closure explicitly refuses NPU because that operation falls back
to host CPU on this stack.

These are separately tested building blocks. Readiness and bounded control
execute on device; complete graph consume/emit/selection/module dispatch,
continuous state, peer progression, backward, optimizer and full workloads are
not certified here. The AIV readiness implementation uses a scalar pipeline,
not an optimized parallel tiling. No performance or resident-training claim
follows. Failure-lifetime injection remains a separate pending gate.
