# Adopted semantics and validation contract

Authority: `tide-core-3` at the revision in `upstream.json`. Local formulas below
instantiate the abstract interfaces; changing them is a versioned local change.

The experimental historical-workload bounded scheduler is a separate finite
representation of these formulas, described in [bounded scheduling](bounded-scheduler.md).
Its explicit first-order VJP carries structural connectivity independently of
numeric derivatives. Dense padding alone must not turn absent gradients into
connected zeros. It does not replace the ordinary eager/autograd interfaces.

## Common spine

Finite fixed multigraph; strictly positive integer edge delays. Complete fiber
-> Agg/Upd/Read -> region SelStep -> comparison snapshot/Next -> active Full ->
messages and outputs. Candidate iff fiber nonempty. Empty nodes/regions preserve
stored state/history. Candidate state adoption may observe all or active only;
Next can clear selected state without erasing the snapshot used by Full.

Logical time, token position, observation count and execution time are distinct.
Graph indices/budgets and all execution coordinates/counters must be exact Python
`int` values in signed int64 range at Python entry points. Bool, float (including
integral float), scalar tensors and out-of-range integers are rejected before
execution or native conversion. Field-specific nonnegative/clock/owner bounds
still apply. This also covers imported state/history, pending messages and input
ledgers before checkpoint restoration changes live owners. Native C++ fields
already use int64; malformed Python records are not coerced into valid C++ ones.
Local program input/output slots have validated physical edge/port mappings;
see `local-ports.md`. Layout is part of graph identity, outside shared weights.
Complete-cut continuation is `(cut, node states, region histories, pending)`;
pending contains every message sent before cut and arriving at/after cut.
Runtime identity, sample count and input-position ledger are also validated.
The sealed-window API explicitly declares complete external inputs in `[a,b)`.
Current native structural identity is **v13**; single-graph checkpoint payload is
**v5**. The separate two-clock application bundle is **tide-token-application-v1**
(`token-application-checkpoint.md`); CPU qualification is in
`evidence/token-checkpoint-coordinates.md`.
Per-port positions start at zero and are contiguous; their times strictly increase.
It does not yet implement independently advancing per-port online watermarks.

The optional [online greedy schedule](greedy-scheduler.md) certifies complete
region-time prefixes from actual pending work and positive-delay closure. It
accepts feedback without a numerical prepass or whole-window event expansion;
the scalar reference remains independent. Its live-fiber capacity is an explicit
refusal boundary, separate from future byte-budgeted chunking/device scheduling.

Native cursor ownership changes materialization, not finite-valued event
semantics. `advance` keeps queues/state native; snapshots explicitly clone tensor
storage and preserve gradients in ordinary grad mode. Input rejection is
retryable; execution exceptions require restoring a prior snapshot into a new
cursor. See `streaming-cursor.md` for ownership and recovery details.

## First local profile: `ema-ffn-v1`

For tagged values x_i and learned source weights w_i, `h=sum_i w_i*x_i`.
External ports and internal edges have different parameter identities.
`proposal = sigmoid(decay)*old + h`; descriptor is a learned linear read of
the proposal. Region selection orders candidates by history count (ascending),
descriptor (descending), node ID (ascending); a profile can omit count priority.
Controls are softmax descriptor probabilities over the complete candidate set.
Histories count selections. Decisions/integers are not differentiated.
Comparison adopts proposal for observe-all or active nodes, otherwise old.
Next returns comparison, or connected zero for clear-on-active. Full computes
`g = h + tanh(comparison @ W + bias)`; selected Emit sends its result through
source-specific learned edge/output scales. No autonomous idle decay is implied.

HARD returns g. SOFTP returns `h+p*(g-h)`. HST has exactly g forward and local
VJP `(bar_h=0, bar_g=u, bar_p=zeta*sum(u*(g-h)))`. This is an explicit surrogate,
not the derivative of the hard forward. Only selected nodes execute Full;
unselected descriptors can receive gradient through softmax denominators.

`full-programs.md` extends Full to an optional auxiliary value and a sparse family
of local output-slot payloads. Only present coordinates are delivered; numerical
zero still sends. Broadcast retains the first profile; slot-affine projection and
explicit logical-time phases instantiate distinct payloads and absence.

`aggregate-programs.md` adds mean, positive weighted mean, active-source softmax
and all-source softmax. Aggregate sees tagged atoms, time, local input slots and
physical scales, and returns summary content plus optional per-source content.
Only all-source softmax differentiates absent-source logits via its denominator;
missing messages and present zeros remain distinct. Source slots
use the graph's logical `SourceDomain`, which can give exclusive physical aliases
one source coefficient. Duplicate logical arrivals fail; physical tags, ledgers
and explicit parameter ownership remain observable. See `source-domains.md`.
Existing memory profiles consume summary content; `content-programs.md` propagates full source information
to custom state/Read/Full programs. `read-programs.md` separates Read and implements
region content/old/proposal modes. `next-programs.md` implements complete Next
requests with comparison-identity state-prefill guards. `region-programs.md`
adds region-owned typed histories and independent SelStep programs, including
empty selection and tensor controls.

The [fiber attention packing policy](attention-packing-policy.md) is an execution
choice outside graph/checkpoint identity. Exact buckets and one padded query
batch preserve event visibility, logical state and the declared public VJP.

## Equality and training

Python/native `full_autograd` is an execution policy outside semantic/checkpoint
identity. Its optional batched affine VJP retains per-row undefined/connected-zero
gradients under the [declared first-order contract](full-batched-autograd.md).
Python/native `aggregate_autograd` independently selects replay or isolated batched
source VJPs under [its contract](aggregate-batched-autograd.md), also outside
semantic/checkpoint identity. Normalization retains per-event Jacobians before
shared-owner accumulation, including near-zero-gradient optimizer behavior.

Logical content/state observability does not fix Tensor object count, physical
layout or operator granularity. [Optional packed transport](packed-transport.md)
preserves source-aware content and Next transitions through batched storage and
explicit first-order replay. Trace and snapshot contracts still apply; physical
reuse counters are distinct from canonical logical event counts.

Additional local state profiles are specified in `state-programs.md` (SSM),
`matrix-memory.md` (Linear, gated `delta`, ungated `delta-rule-v1`) and `attention.md` (aggregated-event GQA/window).
Their clocks, clear behavior and batching contracts are explicit; sharing the
generic executor does not make these different profiles interchangeable.
`state-clocks.md` adds explicit periodic local state ticks, delegating the local
program's batching contract while retaining global continuation/message clocks.
`lazy-add.md` defines explicit tick-repeat decay, encoded versus physical state,
cut decoding and the fixed-parameter boundary of its LH inference interpretation.
`lh-full.md` specifies post-selection activation/normalization and the per-edge
signaling mapping, with Tide's explicit HARD/SOFTP/HST training contract.
`fiber-attention.md` defines separately named same-fiber visibility, sum pooling,
tick-repeat KV log-bias decay and complete cache continuation.
`fiber-pooling.md` adds separately named post-attention mean/linear/softmax profiles
and a graph-domain coefficient vector, without changing sum Aggregate content.
`token-window.md` defines an explicit application clock boundary for readout and
norm-only Full profiles; it does not alter logical edge delays or input seals.

- Exact comparison: graph/atom/edge identities, time, positions, candidates,
  routes, history, validity and pending-message membership.
- Tensor comparison: FP64 atol=1e-10 rtol=1e-8; FP32 atol=1e-6 rtol=1e-5.
  A route mismatch fails; report score margins rather than hiding it by replay.
- Trace: fibers, content, proposal, descriptor, control, active, comparison,
  next, histories, Full values, emitted messages and external outputs.
- Objectives separately root outputs, final state and pending messages. Compare
  input, parameter and differentiable initial-state VJPs. In-memory chunking has
  no implicit detach. Serialized continuation is a declared gradient boundary.
  `Continuation.detach()` explicitly truncates state, region history and in-flight messages.
  Checkpoint v5 stores all node/region tensor slots and validates parameter-alias
  topology, shared values and named optimizer ownership/order/class before
  changing weights. Reconstruct the same sharing and optimizer groups when
  restoring; see `checkpoint-ownership.md`. Older schemas are rejected. Checkpoints do not
  claim to restore a full training controller, data cursor or framework RNG.
  The standalone C++ owner layer additionally defines `TIDENCK1` schema v1 for
  CPU FP32/FP64 named values and built-in SGD/AdamW state. It is a separate
  little-endian, checksummed format with identity and alias preflight and
  exclusive no-overwrite publication; it is not interoperable with Python torch
  files and does not restore graph continuation or a training controller.
- `None` versus connected-zero is observable for optimizer parameter groups;
  do not silently normalize away a missing parameter gradient. For a single
  packed input tensor, zero entries are the ordinary tensor VJP contract.
- Shared parameter ownership must survive packing, parallelism and checkpoints.
- Eager [packed transport](execution-placement.md#packed-eager-transport) groups
  only already-produced messages. It preserves each public row's undefined or
  connected-zero cotangent, physical edge identity and order; numerical zeros
  do not determine whether a row is present or differentiably connected.
- Isolated public roots must preserve structural gradient absence. Packed
  execution currently uses local semantic autograd replay with an explicit
  training cost; see `packed-autograd.md`. Numeric zeros cannot identify absence.
- A specialization owns its loop/dependency order. Common kernels alone do not
  validate formulas; tiny hand-computable cases supply independent anchors.

Python/native streaming and legal block policy interfaces are specified in
[cross-family-policies.md](cross-family-policies.md). Policies and schedule
switches do not alter checkpoint identity or the first-order contract.

## LH and other references

LH contributes C++ inference and implementation ideas, not training semantics.
Its candidate updates, count-priority selector and selected clear can fit the
spine. The bounded equal-width, fixed-weight two-clock composition maps
unit-delay ticks, eager decay, same-fiber attention and token-window Pronounce;
actual original-C++ whole-model inference is qualified in
`evidence/lh-iocortex.md`. The bounded single-PDG inference encoding is qualified
in `evidence/lh-single-graph.md`; Tide training/single-graph resume separately in
`evidence/single-graph-training.md`. The two-clock bundle preserves its actual occurrence ledgers and cross-graph
owners (`token-application-checkpoint.md`). Reconstructing those ledgers from the
single-PDG projection remains a separate obligation; token index is not an
occurrence counter.

`fractal-latcarf` supplies validation design examples (eager/packed/specialized,
chunk/state/gradient checks). Its historical results do not certify this tree.

## Accelerator execution boundary

The standalone historical-topology consumer additionally owns optional tensor
node ranking/event queues and explicitly bounded training windows; see
[its contract](accelerator-scale.md). Those implementation choices preserve exact
discrete order, integer histories, edge identities and payload VJPs. They do not
broaden the public single-device library's placement contract. Training window
resets are explicit application boundaries, never an implicit runtime detach.

The single-device eager extension preserves graph identities, schedules, HST VJP
and checkpoint schemas. CPU FP32/FP64 remains the reference. CUDA FP32/FP64 and
NPU FP32 require explicit backend capability; NPU FP64 is rejected. Graph tensor
payloads and parameters stay on the selected logical device; graph metadata and
discrete scheduling remain host-owned. Qualification alone moves deterministic
CPU fixtures to the candidate device and copies observables back for comparison.
Numerical tolerances never relax routes, owner identities or None connectivity.
The optional [resident library](resident-library.md) owns device-side online
progression and continuation for its declared FP32 inference profiles and explicit Emit mode,
and FP16 inference with FP32 Read/normalization. Payloads and cache slots
retain the configured dtype; discrete scheduling and identities are unchanged.
Resident owners remain on their constructing host thread; CANN submission from
another thread fails before a runtime task is issued. The internal peer transport
submits all devices asynchronously before waiting at the window boundary.
Borrowed device output windows do not export state; snapshot/result are explicit
boundary materializations. Runtime parameter updates invalidate its frozen
inference program until an explicit reconstruction. This separate backend does
not broaden eager training evidence into resident backward or optimizer support.
The separate [explicit resident training owner](resident-training.md) defines
retained-window root tokens, parameter generations, alias-aware device updates
and complete-cut training exports for its declared FP32/FP16 payload profiles,
with FP32 cotangents and optimizer masters.
[HARD physical-slot projection adjoints](resident-emission-vjp.md) use the actual
unscaled emission journal, preserving zero delivery-scale and absent-slot
connectivity. Their qualified scope is separate from compact projection placement.
The internal [Full-sharded training composition](resident-peers.md#canonical-owners-and-complete-internal-training-steps)
reduces aliases into one canonical FP32 master owner, reaches an all-device
finite/representability decision before committing, and publishes rounded values
to every used forward alias. State/KV remain on the coordinator. Its independent
qualification does not certify public multi-device training/checkpoint clients
or complete model sharding; graph identities and checkpoint schemas are unchanged.
The separate internal [compact state/Read/KV placement](resident-peers.md#compact-state-read-and-kv-owners)
keeps complete fibers and global region selection, then adopts state/cache proposals
only after a common device decision. It has no coordinator state/KV replica and
explicitly refuses monolithic reverse APIs. Its separate compact reverse path
packs actual device journals, retains owner-local KV adjoints across windows and
keeps complete-region HST/SOFTP normalization on the coordinator. Canonical alias
reduction/publication consumes these local partials; development and immutable
qualification are recorded separately in STATUS. Public multi-device training
and throughput are not inferred from this internal implementation.
Internal FP16 state, normalized Aggregate and identity/tanh/LH/SwiGLU Full adjoints retain the actual
quantized forward operands/results and use FP32 cotangents/accumulation. Their
cast VJP is the ordinary first-order identity, not a derivative of rounding's
staircase. [Precision scopes](precision.md) distinguish component qualifications
from the complete public training/master/checkpoint lifecycle; no graph identity
or eager checkpoint schema is changed.
The half local attention adjoint similarly preserves QK rounding and global
normalization. Event and same-fiber cache/projection components preserve actual
half operands,including physical source-product rounding,and use FP32 cache
adjoints and boundary sums. [Retained-window checks](resident-retained.md)
separately exercise whole-graph and cache continuation integration. Public half
training exports half named values with FP32 master/slots and validates exact
rounded correspondence on restore; qualification remains separately indexed.
[Resident control adjoints](resident-control-vjp.md) add HST/SOFTP, complete-frame
softmax and linear/FP32-norm Read under the same mathematical contracts.
Their half component preserves payload/control/difference rounding while
retaining FP32 cotangents and internal frame probabilities. This component
and half HST/SOFTP inference alone do not certify complete half graph training.
Its supported modules and qualifications are separate from resident inference.
Its Python client uses a separate `tide-resident-training-v1` CPU checkpoint
containing updated graph parameters, optimizer state and complete continuation;
it does not change the eager checkpoint schemas or restore retained tapes.
Cross-device checkpoint handoff is tested separately from same-device new-process
continuation; no cross-vendor RNG or bitwise optimizer trajectory is promised.
The optional resident `accumulate()` policy sums multiple independently detached
backward groups under one parameter generation, then performs one optimizer step.
Its detach is explicit; it cannot replace a retained cross-window VJP. The
[training contract](resident-training.md) defines capacity, connection flags and
checkpoint refusal; graph identities and checkpoint schemas are unchanged.
Opaque resident device continuations save detached numerical state and can switch
independent streams within their originating live owner. They preserve complete
pending/KV/history/input-ledger semantics while sharing current parameters and
any accumulated parameter gradients. Save/restore refuse retained differentiation;
this is not a disk checkpoint or an implicit cross-stream gradient connection.
Optional device row compaction preserves valid pending identities and complete
cache prefixes, including zero-valued entries. Dynamic allocation synchronizes
at the explicit detached snapshot boundary; live scheduling remains on device.
The explicitly FP64 `norm-fp64-v1` Read cannot compute on NPU. The optional
[placement adapter](execution-placement.md) permits CPU Read/control/ranking
with NPU payloads and retains autograd across those explicit transfers. Each
Read kernel declares its descriptor device; default placement remains with the
payload. The separate `norm-fp32-v1` Read converts visible values to FP32 before
reduction and returns an FP32 descriptor. Its declared conversion/norm VJP and
graph identity are independent of the FP64 profile; see `read-programs.md`.
NPU CSR fiber pooling is outside the supported execution policies;
callers select `event` explicitly. No implicit CSR conversion or host execution
is used to make an unsupported request pass.

The [eager payload-owner extension](execution-placement.md#eager-payload-owners)
keeps node state/KV and delivered messages on their declared owners, gathers each
complete region frame for selection, and preserves differentiable per-message
copies and canonical parameter aliases. Ownership and schedule do not change
graph identity, message identity or checkpoint v5. Checkpoint loading may remap
owners; it remains an explicit autograd boundary. This extension is separate from
device-resident progression and from packed multi-device performance acceptance.


## Configurable low precision

The explicitly selected FP16 extension retains the same graph, delay, scheduling,
HST and owner semantics. FP32/FP64 remain the original required references. FP16
rounding, overflow and underflow can change values or discrete near-tie decisions;
FP32 success alone is not FP16 equivalence evidence. Qualification records its
floating tolerances and never relaxes discrete or None/zero comparisons.
See [precision.md](precision.md) for API, master-optimizer/checkpoint boundaries,
and [accelerator-scale.md](accelerator-scale.md) for the separately configured
Read/control/dispatch consumer. BF16 and automatic mixed precision remain outside
this extension. NPU FP64 and CSR pooling stay explicitly unsupported.
CPU FP16 CSR pooling is also explicitly unsupported; selecting event pooling is
a caller-visible policy choice, not a runtime fallback or implicit promotion.
