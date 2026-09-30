# Shared device forward memory — 2026-09-30

Clean `523b323` passes the full standalone build, four CPU CTests, all29 device
cells and independent CANN profiling. The [manifest](device-memory-20260930.json)
records exact source, raw artifact hashes, allocator observations and failures.
This qualifies the declared single-device FP32 HARD inference profile only.

The forward planner reserves static tables, queues/state/journals and all six
module minima before growing physical chunks. CPU profile validation precedes
device uploads. Conservative/aggressive policies reserve25%/10% of surplus,
respectively; unused allowances remain available to subsequent modules and CANN.
The program queries operator workspaces before allocation and uses one maximum-size
arena for its single ordered stream. An oversized request poisons construction
before allocating that workspace; an incomplete prefix cannot execute.

The new gate passes72 complete mixed-module feedback windows and seven workspace/
refusal checks. It combines slot-affine emission, SwiGLU, LH Full, tanh, fiber/event
attention and Add/EMA, with widths3/33/257,192/512MiB budgets, both policies and both
schedules. Exact-cap serial reuse, view alias accounting, release, deferred upload,
impossible minima and invalid policy are checked. All complete comparisons retain
rtol1e-5/atol1e-6 and exact discrete observations. The earlier29-cell scope includes
separately qualified FP16 components; it does not make the complete flow FP16.

Allocator peaks include construction transients; each fixture starts with zero
allocated TorchNPU bytes. Width257 observations (same for both schedules):

| Budget | Policy | Peak allocated delta | Emission rows | SwiGLU rows | Fiber/event query rows |
| --- | --- | --- | --- | --- | --- |
| 192MiB | conservative | 46.35MiB | 6 | 2 | 1 / 1 |
| 192MiB | aggressive | 47.10MiB | 8 | 3 | 1 / 2 |
| 512MiB | conservative | 142.70MiB | 70 | 24 | 8 / 9 |
| 512MiB | aggressive | 159.47MiB | 86 | 30 | 10 / 10 |

These finite observations calibrate the conservative estimates. They do not certify
all shapes, tighter estimates, vendor/driver HBM, allocator fragmentation, caller
tensors, training or communication memory. No free-HBM guarantee or speedup follows.
Actual row counts and planned buffer/workspace/headroom remain visible in results.

The trace contains62,714 AIV and4,291 AI Core tasks, with no AiCPU or CPU fallback
diagnostic observed. It includes construction, CPU assertions and exports; it is
placement evidence rather than synchronized throughput timing.

Retained failures matter: the first fixture had invalid external positions. A later
symmetric three-channel fixture missed strict parity in a near-zero SiLU/LayerNorm
output by1.45053e-6; it remains an unqualified reproducer. The allocation fixture now
uses asymmetric channels without changing tolerances; dedicated LH conditioning
checks pass separately. The first wide allocation run then hit the explicit workspace
cap because serial operator lifetimes were summed; shared workspace reuse fixes that
case. All original failed snapshots/logs remain, including the diagnostic follow-up.

Node-time attention batching, the remaining public matrix, peer progression and
resident backward/VJP/optimizer remain separate work. F1–F7 are not complete.
