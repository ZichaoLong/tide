# Device parameter-owner adjoints

The internal `append_parameter_vjp` component reduces the physical partials
from [graph VJP](resident-graph-vjp.md) into `ParameterRegistry` owners. Its
single-NPU FP32 profile has the same built-in Aggregate/broadcast, identity/EMA/Add state
and identity/tanh/LH/SwiGLU Full limits. It is not an optimizer or public training API.

The caller supplies the original graph/model registry, optionally restricted to
trainable owners. Static alias metadata uses TensorImpl identity and canonical
names, as the portable registry does. Distinct TensorImpl objects sharing storage
stay distinct. A single owner can have differentiable and HARD Read aliases;
only its differentiable uses contribute to its gradient. Any later update must
also refresh its forward Read aliases. HST/SOFTP additionally reduce physical
Read partials and their connection bits through this same registry; see the
[control contract](resident-control-vjp.md). HARD retains its prior packed layout.

Normalized Aggregate coefficients have per-node/logical-slot partials and
connection bits. An all-source softmax logit can be connected without a message
on that slot. Physical scale owners remain separate unless the registry explicitly
shares them, including an owner shared across a coefficient and a physical scale.

Each owner has a device connection bit and a flat gradient slice. An offset of
`-1` denotes no differentiable use in this profile and avoids allocating unused
matrix gradients. Disconnected slices are zero with a false connection bit;
connected zero slices retain a true bit. A metadata task alone writes flags;
independent feature tiles sum present contributors in stable alias order.
Absent partials may contain NaNs and are never evaluated. The operator clears
outputs on replay and never downloads connection bits to drive reduction.

Explicit no-grad construction validates partial bank shapes, FP32 ownership,
alias shapes and tensor budget before recording the kernel. Empty registries
still require space for dummy kernel arguments. Capacity refusal is explicit.
The supplied registry must describe the same forward model; this internal API
does not infer sharing from equal values or recover it from frozen parameter copies.

The development gate compares 36 real graph/root combinations against CPU
Streaming autograd, with shared cross-node weights, bias/decay/Read aliases,
input/aggregate/output/edge/retention aliases, width1/3/257 and both schedules.
It checks None/zero, poison, replay, distinct storage-overlap owners, a trainable
subset, empty registry, bad shapes and insufficient budget. Profiling is separate
from throughput; immutable qualification is indexed in ROADMAP.
