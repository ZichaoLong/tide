# Adopted semantics and validation contract

Authority: `tide-core-3` at the revision in `upstream.json`. Local formulas below
instantiate the abstract interfaces; changing them is a versioned local change.

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

Native `full_autograd` is an execution policy outside semantic/checkpoint
identity. Its optional batched affine VJP retains per-row undefined/connected-zero
gradients under the [declared first-order contract](full-batched-autograd.md).

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
- Isolated public roots must preserve structural gradient absence. Packed
  execution currently uses local semantic autograd replay with an explicit
  training cost; see `packed-autograd.md`. Numeric zeros cannot identify absence.
- A specialization owns its loop/dependency order. Common kernels alone do not
  validate formulas; tiny hand-computable cases supply independent anchors.

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
