# Explicit payload precision

`GraphConfig.model.dtype` and the qualification CLI accept `float16`, `float32`
and `float64`. NPU rejects FP64; BF16 remains unsupported. Public single-device
Python and native schedules keep their own independent implementations. Support
is finite and target-specific; implementation is not device qualification.
CPU FP16 CSR pooling is explicitly unsupported (the local Torch2.10 CPU kernel
rejects Half sparse-dense matmul); choose `fiber_pooling="event"` explicitly.
NPU CSR pooling remains unsupported at both payload precisions. No implicit
promotion or pooling-algorithm substitution is used to satisfy these requests.

FP16 model initialization is the FP32 seeded fixture cast to FP16. Parameters,
inputs, state, KV slots and messages use the declared dtype; callers supply
matching inputs. Local ATen operators keep the same formulas. Floating reductions
follow their declared operator/backend policy; explicit FP64 Read remains FP64
and is still rejected on NPU. The separately owned full-size consumer additionally
exposes Read/control placement and precision, as documented in accelerator-scale.md.
The eager public scheduler remains host-owned on all devices. The optional
[resident backend](resident-library.md) independently advances online device
queues and batches; its FP16 inference contract is described below.

For training, construct `FP32MasterOptimizer(runtime.model.parameters(),
optimizer="adamw", loss_scale=128, lr=...)`, compute a scalar FP32 loss, call
`optimizer.backward(loss)`, detach a completed session when required by the
existing training boundary, then call `optimizer.step()`. `zero_grad()` clears
both payload and master gradients. SGD and AdamW options are passed to PyTorch.
The caller owns accumulation, clipping policy, learning-rate schedules and task
loss; this helper does not create a training loop or hide dynamic loss scaling.
Construct the optimizer after weight-only initialization; recreating it is required
after replacing model weights outside its update/checkpoint path. Each unique
trainable FP16 leaf has one FP32 master. None gradients remain absent;
connected-zero gradients retain normal optimizer decay/state behavior.

Static loss scaling is explicit and checkpointed. Nonfinite gradients fail before
any weight update. Overflow while casting updated weights fails explicitly and
requires restoring a checkpoint; no implicit retry, skip or fallback occurs.
This policy is mixed precision with FP32 masters/slots/loss, not pure FP16 or AMP.
Passing an ordinary optimizer over FP16 leaves retains that optimizer's own dtype
behavior; it does not acquire FP32 masters automatically.

Python Session checkpoints retain schema v5 and add an optimizer-owned
`tide-fp32-masters-v1` payload when using this helper. They store/check master
precision, shape, owner order, SGD/AdamW state, static scale and correspondence
between quantized masters and model weights before mutating live owners.
Same-stack continuation and CPU handoff need their own qualification. C++ native
execution through Python uses this same Python checkpoint boundary. The separate
standalone C++ owner checkpoint/NamedOptimizer interface remains FP32/FP64; the
full-size consumer supplies its own FP32 masters and does not import checkpoints.

`qualify(..., dtype="float16", atol=..., rtol=...)` runs the ordinary independent
CPU schedule, complete observables, isolated gradients, chunking, optimizer
trajectory, and fresh-process checkpoint checks. FP16 defaults are atol1e-3,
rtol2e-2; they are explicit in the result. Discrete events/routes and None/zero
connectivity are exact, regardless of tolerance. Diagnostic losses/gradient
scaling use FP32 to avoid introducing avoidable FP16 reduction underflow into the
test itself. A route mismatch still fails. Same-formula FP32 success supports
the design, but cannot certify FP16 numerical stability or training convergence.

Immutable local [FP16 qualification](evidence/fp16-qualification-20260929.md)
records the exact supported scopes and retained failures. The standalone
[full-size FP32/FP16 comparison](evidence/accelerator-fp16-performance-20260929.md)
records inference/training timings and allocator peaks under shared load. It does
not establish a universal speed or training-memory advantage. On another stack, run
`scripts/qualify_accelerator.py` with explicit `--device` and `--dtype float32`,
then `--dtype float16`, for both `--implementation python` and `native` with its
matching `--native-library`; follow [accelerators.md](accelerators.md) for builds.

The resident C++/CANN owner and its Python client accept FP16 **HARD inference**.
State, parameters, inputs, messages and KV retain FP16. EMA/Add and fiber-bias
updates round at each declared operation/tick, including within a node-time batch.
Read and normalized Aggregate use FP32 on the stored payload values. Attention
QK/projections use the payload dtype; softmax, weighted accumulation and merging
physical key tiles use FP32, then cast the completed attention result. This avoids
overflowing an unnormalized half partial when the normalized output is finite;
it does not prevent overflow in the half QK product itself.

Device journals keep FP32 diagnostic storage and restore payload fields to FP16
at the explicit CPU export. Scores stay FP32; exported Region controls round to
the public payload dtype after FP32 softmax. FP16 HST/SOFTP and complete resident
training remain unavailable and fail explicitly. Internal
[FP32-master publication](resident-optimizer.md) has separate component
qualification. The [state](resident-state-vjp.md), normalized
[Aggregate](resident-aggregate-vjp.md) and identity/tanh/LH/SwiGLU
[Full](resident-full-vjp.md) components preserve actual half forward rounding
while accumulating adjoints in FP32; their tests use the corresponding quantized
forward/FP32-adjoint reference. This is not a claim of bitwise equivalence to
pure-half backward accumulation. Component support does not enable public
training, retained-window training or checkpoint resume. The
local [attention adjoint](resident-event-vjp.md) separately preserves half QK
rounding with FP32 global softmax/adjoints; event/fiber cache integration remains
outside that half component scope. The
complete-flow gate compares an independent CPU streaming schedule with exact
discrete/identity checks and FP16 atol2e-3/rtol2e-2; FP32 retains its original
thresholds. Build, device qualification and performance evidence remain distinct.
