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
Local program input/output slots have validated physical edge/port mappings;
see `local-ports.md`. Layout is part of graph identity, outside shared weights.
Complete-cut continuation is `(cut, node states, region histories, pending)`;
pending contains every message sent before cut and arriving at/after cut.
Runtime identity, sample count and input-position ledger are also validated.
The sealed-window API explicitly declares complete external inputs in `[a,b)`.
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
missing messages and present zeros remain distinct. Existing memory profiles
consume summary content; `content-programs.md` propagates full source information
to custom state/Read/Full programs. `read-programs.md` separates Read and implements
region content/old/proposal modes. `next-programs.md` implements complete Next
requests with comparison-identity state-prefill guards. `region-programs.md`
adds region-owned typed histories and independent SelStep programs, including
empty selection and tensor controls.

## Equality and training

Additional local state profiles are specified in `state-programs.md` (SSM),
`matrix-memory.md` (Linear/Delta) and `attention.md` (aggregated-event GQA/window).
Their clocks, clear behavior and batching contracts are explicit; sharing the
generic executor does not make these different profiles interchangeable.

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
  Checkpoint v4 stores all node/region tensor slots and validates parameter-alias topology before changing any
  weights; reconstruct the same sharing when restoring. Checkpoints do not
  claim to restore a full training controller, data cursor or framework RNG.
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
spine. Compatibility still requires a precise mapping of unit-delay ticks,
eager decay, same-fiber attention and token-window Pronounce. No numerical LH
equivalence is claimed until an original-C++ comparison is executed.

`fractal-latcarf` supplies validation design examples (eager/packed/specialized,
chunk/state/gradient checks). Its historical results do not certify this tree.
