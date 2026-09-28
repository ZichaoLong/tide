# Historical topology placement benchmark

This standalone consumer is being developed against the installed Tide C++ core.
It does not extend the public single-device runtime or import checkpoints.
Development small-tensor parity has passed on CPU and one/two NPUs. Immutable
qualification and full-size performance results are pending; consult `STATUS.md`.

The targets are the historical D2048/B512/V50304, 465-node topology:
17,269,426,339 Attention parameters and 9,468,020,899 Add parameters. The latter
is the historical "8.8B" count in units of 1024^3. FP32 weights, CPU initialization
order/seed, edge identities, two body ticks/token, clear, full-domain softmax,
four-head Attention and `norm-fp64-v1` Read are preserved. The measured modes
are no-grad and grad-forward, with no backward, optimizer, or history detach.

`--transport resident` keeps node state, KV caches, Aggregate/Full and messages
on assigned devices. The benchmark's own sealed-window scheduler reuses public
local programs. Same-device messages remain on that device; cross-device inputs
use Torch device-to-device copies when the destination consumes the fiber.
CPU computes exact FP64 Read from the required vectors and maintains count-only
region histories; controls return to each node. This is single-process model
sharding, not data parallelism or HCCL collectives. Device-copy byte counters
describe explicit tensor transfers, not independently measured fabric traffic.

`--transport host` retains the earlier comparison: CPU state/messages/Aggregate
and device State/Full through adapters. It necessarily transfers same-card
messages through CPU. Neither transport uses CSR pooling. Event pooling is the
declared historical baseline; CSR capability is independent of message placement.

The local TorchNPU 2.10/CANN9.0/driver25.3.rc1 stack crashes in cross-device
event waiting with its default asynchronous task queue. The same tiny two-device
value/VJP gates pass with `TASK_QUEUE_ENABLE=0`. The benchmark wrapper and verifier
therefore expose `--npu-task-queue` and default to 0, recording that condition.
The bare C++ executable also defaults an unset `TASK_QUEUE_ENABLE` to 0 before
initializing NPU; it preserves an explicitly supplied value for diagnostics.
This changes dispatch synchronization, not precision or the message payload path.
Other stacks and queue modes require their own gates; do not hide the failure.

`--placement memory` uses largest-node-first balancing. `locality` starts from
that mapping and strictly reduces physical cuts by deterministic moves/swaps,
bounded by 110% of mean node parameter load plus one indivisible node (or the
initial maximum). Embedding/head go on the lightest remaining devices afterwards.
Placement preserves every parallel edge. Static cut fraction and actual message
bytes must both be reported; cut reduction alone is not a speedup claim.

Build with `scripts/build_accelerator_scale.py --core-build CORE --build-dir CLIENT`.
This verifies the core source/binary identity, installs it, and links the consumer
against its exported target and exact Torch dependency. NPU needs the qualified
standalone SDK. It must not load Python or build-time stub libraries.

`scripts/verify_accelerator_scale.py` compares both transports and both placements
against the independent CPU scalar-slot schedule. It checks complete traces,
states, routes, histories, pending messages, logits and isolated nonzero/zero
VJPs, including None connectivity, as well as resident tensor devices. The
historical forward-only slot fixture's views are rebound to their original CPU
owners for this gradient oracle without changing values or initialization order.

The default `--vjp-policy strict` retains componentwise FP32 rtol=1e-5,
atol=1e-6 for quadratic-root VJPs. An explicit `basis-conditioned` check also
supports cancellation-sensitive quadratic contractions of roots with at most64
coordinates. If the strict comparison fails, it reports that failure, checks
every coordinate's VJP (the complete Jacobian) with the original componentwise
tolerances and exact None connectivity, and reconstructs each quadratic VJP on
CPU in FP64. Each direct VJP must agree with its reconstruction within
`atol + rtol * sum(abs(J_ij * cotangent_j))`. A changed Jacobian, nonfinite
value, connected-zero discrepancy, oversized root or failed reconstruction still
fails. All forward values, discrete routes and zero VJPs keep the strict rule.
This is a named benchmark verification policy, not a change to the core runtime
contract or a claim that the strict quadratic test passed. It changes no model
operator, parameter, precision, seed or timed computation.

Why this distinction matters: the wide Add D8/B1 seed0 fixture's first Full
is SiLU/RMS. Its squared norm is nearly constant, so backward subtracts large
terms. The same local formula without graph scheduling or transfers reproduces
CPU/NPU max error7.7188e-6; CPU FP32 itself differs from CPU FP64 by9.20483e-6.
The complete local Jacobian passes the original tolerance (maximum ratio0.0313).
An explicit FP32 RMS expansion did not change the failure. Historical strict
failures remain failed; evidence must name which policy qualified a workload.

The standalone process finalizes the NPU SDK after local tensors and workers
are destroyed and before static teardown. This resolved an observed intermittent
8-device exit crash in development. The installed core's later shutdown hook
can print a repeated-finalize warning; a zero exit and all gates are still
required. The earlier failed run is retained.

Formal runs use `scripts/benchmark_accelerator_scale.py`, a clean frozen source,
explicit device/count, copied topology, bounded time/RSS and fresh output paths.
There are 12 growing-context tokens: four warmup, eight measured. Every timed
token includes embedding, body, vocabulary head, transfers and device barriers.
Setup, IDs, previous-logit release and validation/metrics are outside timing.
Raw JSONL, atomic run/summary records, binary/input identities, physical mapping,
per-device allocator memory and an optional local Trackio projection are retained.
Independent concurrent processes and one model spanning devices are separate
experiments; comparisons must record contention and physical assignments.
