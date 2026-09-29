# Historical topology placement benchmark

This standalone consumer links the installed Tide C++ core. It does not extend
the public single-device runtime or import checkpoints. Immutable CPU/2/8-NPU
small-tensor qualification passed, including the465-node topology with the
explicit numerical policy below. See [qualification evidence](evidence/accelerator-scale-20260928.md).
The [bounded full-size performance assessment](evidence/accelerator-performance-20260928.md)
is complete: 54 cells cover inference, placement, complete training, matched
repetitions and a concurrent workflow. This does not qualify arbitrary scales
or hardware/software combinations.

The targets are the historical D2048/B512/V50304, 465-node topology:
17,269,426,339 Attention parameters and 9,468,020,899 Add parameters. The latter
is the historical "8.8B" count in units of 1024^3. FP32 weights by default, CPU initialization
order/seed, edge identities, two body ticks/token, clear, full-domain softmax,
four-head Attention are preserved. The default Read remains `norm-fp64-v1`. The measured modes
are no-grad, grad-forward and explicitly requested complete training windows.
The historical no-grad/grad-forward modes keep their original window behavior.

`--transport resident` keeps node state, KV caches, Aggregate/Full and messages
on assigned devices. The benchmark's own sealed-window scheduler reuses public
local programs. Same-device messages remain on that device; cross-device inputs
use Torch device-to-device copies when the destination consumes the fiber.
The default computes FP64 Read on CPU; controls return to each node. This is
single-process model sharding, not data parallelism or HCCL collectives. Device-copy
byte counters describe explicit tensor transfers, not independently measured
fabric traffic, and exclude implicit scalar extraction inside the selector.

Read and control placement are independent of payload precision:

| Configuration | Read | Softmax controls | Intended use |
| --- | --- | --- | --- |
| `--read-device cpu --read-dtype float64 --control-device cpu` | CPU FP64 | CPU FP64, controls cast to FP32 | Historical default/reference |
| `--read-device cpu --read-dtype float32 --control-device cpu` | CPU FP32 | CPU FP32 | Same-precision CPU reference |
| `--read-device model --read-dtype float32 --control-device model` | Node device FP32 | Fixed region-owner device FP32 | Resident inference/grad-forward candidate |

Mixed Read/control placement is also explicit. `model` means the node's assigned
device for Read and the lowest region member's assigned device for controls.
The default remains unchanged. FP32 uses the benchmark-owned custom Read profile
`scale-norm-fp32-v1`, changing graph identity while keeping the L2-norm formula.
NPU FP64 and model-device scoring with host transport fail explicitly.

`--ranking-device cpu|model` and `--event-device cpu|model` independently select
CPU or tensor implementations. Defaults stay CPU. Tensor ranking uses exact stable
lexicographic order: selected count ascending, affected count descending for LH,
descriptor descending, node ID ascending. Histories remain host-owned int64 maps;
node selection decisions run on the fixed region-owner device. Equal-length
candidate buckets batch the nondifferentiable ranking only. Softmax autograd stays
independent per region, preserving disconnected versus connected-zero VJPs.
NPU ranking requires FP32 Read; FP64 descriptors are never silently cast.

The tensor event queue owns scheduling keys on the first shard and computes next
arrival time, ready membership and canonical order there. It preserves all physical
edge identities and differentiable payload handles. The host consumes returned
indices to dispatch C++ local programs and assemble results. This is an explicit
device scheduling candidate, not an entirely device-resident control loop.
Snapshot export may sort host metadata outside the forward scheduling path.
Integer keys remain int64; on the local CANN9.0 stack ArgSort reports on-device
AiCPU execution for int64, distinct from host CPU and from AiCore. Explicit metadata
uploads/downloads have separate byte counters; other implicit validation/scalar
synchronizations are not covered by those counters. Both candidates require their
own exact-route/history/pending and isolated-VJP gates before performance use.

`--check 1` uses an independent CPU scalar schedule with matching Read precision.
`--reference-read-dtype float64` explicitly compares against historical FP64,
reporting descriptor error and route mismatches. Only descriptor dtype metadata
is normalized for that cross-precision comparison; discrete routes, all other
observables and gradients retain their checks. Near-tie route changes are failures,
not silently accepted numerical noise. Read and control roots have isolated VJP
checks, including connected-zero versus absent gradients. The CPU/two-NPU extension passed [immutable qualification](evidence/accelerator-scoring-20260928.md);
the new dispatch and complete-training candidates also passed CPU/two/four/eight-NPU
[immutable qualification](evidence/accelerator-dispatch-training-20260928.md).

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
Setup, IDs, previous-logit release and final-logit validation/metrics are outside
timing. The executor's own descriptor/device guards remain inside the body timer.
Raw JSONL, atomic run/summary records, binary/input identities, physical mapping,
per-device allocator memory and an optional local Trackio projection are retained.
Transfer byte counters cover explicit client requests and metadata paths;
autograd reverse copies, scalar extractions and internal vendor transport are not
included in those counters. Remote message bytes are logical forward payload
volume, not measured fabric traffic. Such work still contributes to synchronized
timing when it occurs within the timed phases.
Independent concurrent processes and one model spanning devices are separate
experiments; comparisons must record contention and physical assignments.

All requested device contexts are initialized before CPU weight construction so
reserved chips remain visible as occupied during that phase. Cooperative queue
locks do not prevent unrelated/non-cooperating processes from entering a device;
record such contention separately from a model capacity failure.

## Complete training windows

`--training-steps N` enables a complete training benchmark with `--grad 1` and
`--warmup 0`. Each optimizer update processes `--steps` tokens from an empty graph
state. Every token within that window retains full autograd/KV history: there is
no implicit detach or shortened window. Parameters and optimizer states continue
across updates. This explicitly models independent training sequences; it does not
claim continuous-stream training across optimizer boundaries.

The synthetic objective is the mean token cross-entropy over the entire batch and
window, with targets `(input_id + 1) % vocab`. HARD signaling preserves the historical
benchmark profile. `--optimizer adamw|sgd` selects AdamW (betas0.9/0.999, eps1e-5) or
SGD (momentum0.9), both weight decay0.01; learning rate defaults to1e-4. None gradients
remain absent and skip optimizer updates/decay. `--training-warmup` counts complete
optimizer updates, separately from the historical inference token warmup.

Timing includes zero_grad, empty-window executor initialization, IDs, embedding,
body, vocabulary head, cross-entropy, backward and optimizer update, with all-shard
barriers between phases. Model construction, previous-window loss/graph destruction
and terminal loss validation/metrics are excluded. Executor descriptor guards
remain inside forward timing. Phase times, total update time, normalized
ms/sample-token, loss, gradient-owner inventory and optimizer-state inventory are
recorded. These are throughput measurements of synthetic full training steps, not
training-convergence evidence. Training phase timings are not interchangeable with
the historical grad-forward token timings.

`scripts/verify_accelerator_training.py` checks both optimizers and models against
an independent CPU scalar-slot schedule for three updates: complete observables,
exact discrete decisions, loss, every parameter gradient including None/zero,
updated weights and optimizer slots. The ordinary isolated-root VJP check also runs.
Training/dispatch correctness passed the
[immutable CPU/two/four/eight-device gates](evidence/accelerator-dispatch-training-20260928.md).
The [completed full-size assessment](evidence/accelerator-performance-20260928.md)
records separate inference and training choices, process variance and retained
capacity failures. Public defaults remain unchanged.

## Explicit FP16 extension

`--dtype float32|float16` selects parameter/state/cache/message and projection
precision. FP16 requires resident transport. Initialize in the original CPU FP32
RNG order, then quantize; parameter bytes use the actual element size. Read keeps
its independent FP32/CPU-FP64 policy, controls are FP32 and event keys remain int64.
This changes floating representation, not graph topology or logical schedules.

FP16 training uses FP32 cross-entropy, persistent FP32 master parameters and
SGD/AdamW slots. `--loss-scale` is static (default128 for FP16,1 for FP32), gradients
are unscaled in FP32 before updates and parameters are copied back to FP16.
None owners skip update/decay; connected zero remains defined. Gradients are
checked before any update. Nonfinite gradients or updated payloads explicitly
fail; no dynamic skip or silent fallback. Timers include these checks, master
copies and unscaling for the new implementation. Model construction and
optimizer/master setup are excluded and recorded separately. Full-size FP32 comparisons must
use this same implementation, not old timings with a different guard policy.
This is mixed precision, not pure FP16 or AMP; there is no automatic cast policy.

`--check 1 --dtype float16` compares against the independent CPU scalar-slot
FP16 oracle. `--reference-payload-dtype float32` instead selects a CPU FP32
oracle with FP16-quantized initial parameters/constants as a cross-precision
diagnostic; it can fail because rounding changes a route. The default numerical
policy is atol1e-3, rtol2e-2, configurable with `--check-atol/--check-rtol`; FP32
keeps its original tolerance. Full trace, edge identity, routes, histories,
pending membership and None/zero remain exact. Isolated-root VJPs use FP32 losses
and static scale128. Low precision may change near-tie decisions even if both
implementations use the same formulas; those differences remain failed parity.
This finite qualification does not establish convergence or general FP16 stability.

The public Python/session FP16 API and its explicitly owned optimizer are described
in [precision](precision.md); the standalone scale consumer still owns no checkpoint
format or general heterogeneous public runtime.

The D8 half-precision consumer gradient probe shows a small packing-dependent
rounding difference even on CPU: max absolute error0.001953125, maximum default
tolerance ratio1.02827 for one embedding VJP. Preserve this strict failure.
The follow-up finite qualification explicitly uses `--check-atol 0.002` with
rtol0.02; both verifier scripts expose these options. This is a declared relaxed
numerical gate, not bitwise equivalence. The public API's separate half gate
retains its own recorded tolerance.

The [completed full-size FP32/FP16 pairs](evidence/accelerator-fp16-performance-20260929.md)
use this same implementation for both dtypes. They retain fixed allocation per
pair, measured timing scopes, memory, work counters and resource failures. These
shared-load observations do not replace independent small-tensor correctness gates.
