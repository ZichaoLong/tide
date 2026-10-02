# Public native host execution controls

Tested source: `2222d9db25a234eb0c739baf11bdccfffa2c5f5e`, clean immutable
snapshot. All four build/gate jobs passed: CPU11, NPU10, zero skips.
[Audit and artifact identities](consumer-host-execution-20261002.json).

The actual consumers now expose the existing native node worker pool through
`--workers`, independently of ATen `--threads`. `--packed-sources` and
`--batch-next` select existing packed source transport and batched Next/reset.
The Python native client and standalone LibTorch entry record effective choices
in `host_execution`. Defaults are unchanged. The independent Python scheduler
and resident device scheduler explicitly reject nondefault host controls before
model construction; no request is silently ignored.

Sixteen directed complete-training trajectories (CPU8/NPU8) compare all recorded
states, history, routes, pending messages, outputs, connected/disconnected
gradients and parameter updates against independent Python streaming. Cases span
three families, both schedules, Add/Attention, clear/retained state, SGD/AdamW,
workers2/3 and separate/combined transport switches. CPU includes FP32/FP64;
NPU uses FP32. Every trajectory retains two windows across each of two updates.
CPU additionally checks early refusal and two real CLI paths with unaligned
arrivals. NPU additionally checks both resident clients' two-card default CLI
paths and explicit workspace/head-budget refusals.

The core, resident ABI and CANN kernels are unchanged. CPU recompiles the small
consumer. NPU qualification reuses11 development objects only after comparing
source/header/compile options, then links the new installed consumer and checks
its loader. Both retain byte-verified qualified core/backend dependencies;
this is not a new full-core build. Development CPU11/NPU8 also passed. No new
failure was hidden or tolerance changed.

Records: `TASK/runs/build-host-execution-{cpu,npu}-clean01` and
`TASK/runs/host-execution-{cpu,npu}-clean01`, where
`TASK=/mi/data2T/zlong/tide-execution-flows`. Status, zero exit, JUnit, complete
comparison receipts, source and binary identities are audited. The unaffected
8,954-test CPU suite was not rerun. The earlier one-worker representative results
remain scoped to their original configuration. Bounded host-policy performance
screening is separate; this gate does not establish full-size readiness or a
throughput benefit.
