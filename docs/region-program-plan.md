# Region programs and complete history gate

Complete content, Read and Next are qualified; see their contracts/evidence.
This remaining gate refines ROADMAP. STATUS is the only current handoff.
Keep LH read-only and treat it as an inference oracle, not a training authority.

## Typed history and program boundary

Introduce a region-owned History record with named scalar int64 metadata, named
per-node int64 maps, named floating tensor slots and last_time. Counts/affect
history live here, never in node state. Preserve functional updates and lazy
allocation: empty candidate regions call no program and do not create history.
Validate ownership, int64 ranges and profile-specific layout. Default selected
counts remain nonnegative; generic int metadata can be signed. Check increments
for overflow, including the related built-in node observation counters.

Replace hardcoded selection with registered Python RegionProgram and native
RegionKernel. The request contains old history, logical time and canonical
(node ID, descriptor) pairs for the complete nonempty candidate set. Graph-owned
region membership/budget/policy is immutable configuration, not shared mutable
module state. Return active subset (possibly empty), exactly one control per
candidate and next history. Validate subset/capacity, control domain, tensor
metadata/finiteness and resulting history before committing.

Keep scalar descriptors in the first increment. Controls may be general tensors
for custom local programs; existing Emit/control-blend profiles require scalars.
If named heterogeneous control fields prove necessary, add one explicit record
through Next/Full/trace rather than hiding them in state or closures. LH's norm
Read needs an explicit FP64 descriptor policy and conversion of default softmax
controls to payload dtype, including packed execution; do not silently cast a
norm computed in FP32 and call that FP64 accumulation.

## Profiles and independent anchors

The default count/descriptor/ID selector must preserve all existing behavior,
including singleton softmax connected-zero VJPs. A tensor-history example should
make history affect later candidate scores/controls, with a learned registered
region parameter. One useful scalar recurrence is history'=alpha*history+sum(d),
score_v=d_v+history*learned_bias_v, followed by top-k and softmax. Pass graph-owned
local membership layout instead of capturing physical node IDs in shared weights.
Use hand-computable two-event histories, routes and VJPs, including initial-history
leaves and disconnected samples. A custom selector that selects none tests
passive candidates, missing Full and absence of generated downstream messages.

Later LH selector profile: prioritize fewer selections, then more prior affects,
then larger FP64 norm, then stable node order. Increment affects for every actual
candidate and selections only for active nodes. Dedicated regions encode forced
activity outside base hubs. Lock the original C++ branch/config when comparing
heap versus tensor ranking; large counter overflow is not an implicit equivalence.

## Integration checklist

- Continuation.history becomes the complete typed record; fork preserves tensor
  graphs, detach truncates all tensor slots, checkpoint v4 encodes plain records.
  Validate checkpoint identity/alias layout before touching any weights.
- Native cursor import and snapshot clone history tensor storage; detach truncates
  it. Sparse advance must not rescan all stored regions or clone full history.
- Trace exports the complete post-selection history and controls; comparison and
  objective helpers include appropriate tensor-history roots and None/zero checks.
- Model owns registered region modules/parameters, native adapter exports supported
  profiles and rejects Python overrides. SettleGraph embedding reuses body region
  programs and creates separate boundary selector programs.
- Python reference and native schedules remain independent. Python specialization
  must stop assuming singleton always-active: evaluate the same local selector,
  skip Full/delivery if inactive, retain its independent topology loop.
- State prefill still follows the exact State + comparison-identity Next contract;
  history/control recursion stays causal across frames, with Full batching after.

Acceptance: default-regression parity; analytic tensor-history values/initial-state
and parameter/input VJPs; empty selection; serial/parallel, batch and prefill/
causal paths; generic and specialized cycles/DAGs/SettleGraph; cut composition,
native cursor ownership, detach, sharing, optimizer/checkpoint restoration and
malformed output rejection in FP64/FP32. Add a standalone native custom selector.
Qualify an immutable implementation commit, then archive evidence separately.
After this seam, implement LH profiles/Pronounce with an immutable snapshot of
actual dirty C++ sources and compare complete mapped inference state/messages.
