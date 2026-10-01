# Optional resident inference backend

The CANN backend is an optional library alongside the portable Tide core. It
owns the actual online queue, readiness, node-time batches, selection and
recursive advancement on one NPU. It accepts legal positive-delay feedback
as well as DAG/Settle encodings. Host code submits a sealed window; it does
not consume per-event scalars or decide the next event.

This backend exposes **single-device FP32/FP16 inference**, defaulting to HARD.
The [control extension](resident-control-vjp.md) adds explicit HST/SOFTP for
broadcast emission in FP32; FP16 currently requires HARD. Peer progression
remains separate work. Current build and device verification status is recorded
in [STATUS](STATUS.md).
The separate [explicit C++ training owner](resident-training.md) composes the
restricted graph VJP, optimizer and retained-window lifecycle; it does not change
this inference session's ownership or autograd contract.
The broader delivery contract remains [execution-flows.md](execution-flows.md).

## Python client

Build the portable core with `scripts/build.py --backend npu --npu-runtime python`.
Then build this backend with `scripts/build_device_control.py --core-build CORE
--build-dir NEW --ascendc-soc ACTUAL_SOC`. The explicit backend directory must
contain its build manifest and matching `_tide_resident`/`libtide-resident` binaries.
The loader checks their hashes and the exact native core. No standalone NPU SDK
is linked into a Python-owned runtime. CPU imports do not load this plugin.

```python
import torch
from tidegraph import GraphRuntime, ExecutionOptions, ExecutionPlacement, ResidentLimits

runtime = GraphRuntime(
    config, device="npu:0", native_library=core_build,
    resident_library=device_build,
    options=ExecutionOptions(
        implementation="native", schedule="greedy", mode="hard", trace=False,
        placement=ExecutionPlacement(preset="resident"),
        resident_limits=ResidentLimits(chunk_policy="aggressive"),
    ),
)
with torch.no_grad(), runtime.session(batch_size=2) as session:
    window = session.advance_device(external, stop=20, sealed_until=20)
    # All three tensors stay on the NPU. Invalid rows are unused capacity.
    values, valid, coordinates = window.values, window.valid, window.coordinates
    # Consume here, or clone before the next advance reuses these buffers.
    next_window = session.advance_device(next_external, stop=40, sealed_until=40)
    checkpoint = session.snapshot()  # Explicit CPU export, when needed.
```

`streaming` and `greedy` select the same online device algorithm's single-action
or legal batch mode. `resident_limits` holds queue/workspace/KV bounds and physical
chunk sizes. These do not truncate logical visibility. Irrecoverable capacity
refusals poison the owner; restore an earlier complete cut into a new session.
Invalid input coordinates, dtype/shape or finite-value checks fail before device
state execution. A closed session cannot advance or export; `close()` is idempotent.

`advance_device()` returns borrowed, read-only packed output buffers. Coordinates
are int64 `[sample, node, time, kind, output-port, position]`; `valid` distinguishes
an absent output from a present zero. Counts and work statistics also remain on
device. CPU/NPU input payloads are accepted, on one device per call. The boundary
adapter sorts host input metadata, packs payloads once and checks the whole batch;
recursive messages are generated and placed by the captured device program.

`advance()` additionally materializes a normal CPU `Result` for compatibility and
correctness consumers. `result()`, `snapshot()` and the `continuation` property are
explicit CPU exports. They never feed the owner's internal continuation. Settle
accepts `[batch, positions, width]` inputs; its normal Result is projected to the
body graph, while device coordinates remain in the documented physical encoding.
No output head, masking algorithm or loss is imposed by this library.

Construction validates and freezes parameter values. Normal in-place updates and
Python parameter replacements are detected before advance; recreate a session
from an explicit complete cut to use new weights or another schedule. Writes
through `.data` or external pointers are outside this ownership contract.
`save()` and `load()` use the existing checkpoint schema and reject an optimizer
argument. Saving rejects changed weights. `load()` validates the CPU checkpoint,
parameter aliases and a complete family boundary before closing the old owner;
it restores weights and continuation together. It then releases the old device
buffers before constructing new ones, so a device failure at that point leaves
the session closed. Loading shared weights invalidates other frozen sessions.
`reset()` explicitly starts a fresh sequence with current weights and limits.
Both operations require `torch.no_grad()`. Weight-only initialization remains
`runtime.load_weights(path)`; a caller-owned continuation may also be passed to
`runtime.session(..., continuation=q)`. Checkpoint exports may be changed
by callers without altering the device owner. No hidden detach or implicit
`no_grad()` context substitutes for training support.

## Independent C++ client

Build the core with `--npu-runtime standalone`, then use the same backend build
script and install both CMake packages. Link `tide::resident` and `tide::runtime`
after `find_package(TideResident CONFIG REQUIRED)`. Include `tide/resident.h`.
Declare `portable_torch::RuntimeSession` before tensors and resident owners, then
close all owners before closing the standalone runtime.

`tide::ResidentSession(graph, model, continuation, device, limits)` accepts declared
built-in modules, FP32/FP16 parameters on CPU or the target NPU and an imported complete
cut. It freezes values at construction. Normal in-place parameter changes are
refused; replacing tensors in the original C++ model does not replace captured
owners. `advance(inputs, stop, seal)` returns `ResidentWindow`; snapshot/result
are explicit CPU exports, with the same buffer and failure rules as Python.

Supported local modules are those already qualified by the device content flow:
sum/mean/weighted-mean/active/all-softmax Aggregate; identity/EMA/Add-repeat/event
attention and five fiber pooling profiles; linear/FP32 norm Read; count/positive
selection; adopt/clear Next; identity/tanh/LH/SwiGLU Full; broadcast/slot-affine,
phase-aware HARD emission and supported state clocks. HST/SOFTP use broadcast;
non-HARD slot-affine combinations are explicitly rejected. Custom or unavailable
module declarations fail explicitly. FP32 scoring and all four placement stages
must remain on the selected NPU. Host scheduler switches unavailable to this
backend are rejected rather than ignored.

FP16 state, messages, KV and output buffers retain the configured dtype. Read,
normalized Aggregate and attention normalization/weighted accumulation use FP32;
attention QK and projections use the payload dtype. Explicit half scoring,
HST/SOFTP and training requests fail. Diagnostic journals widen payloads on device
and restore payload/control fields to their public dtype at export; they do not
drive recursive execution. A dtype
change does not change int64 metadata, stable selection or physical-edge identity.
See [precision.md](precision.md) for rounding and verification limits.

The Python wrapper is a client of this C++/CANN implementation. Its results do
not certify an independent pure-PyTorch device scheduler. Neither import nor a
successful build establishes live-device qualification or throughput.
