# Next LH gate: Add state, idle clocks and complete inference projection

Region programs and descriptor precision remove the selector obstacle. Continue
from the original C++ snapshot identified in STATUS/`lh-selector.md`; never edit
the LH worktree. The selector component oracle is not a whole-model oracle.

## Establish the clock mapping before optimizing it

Original `BaseAL::forward` calls hidden decay on **every tick**, including absent
inputs. `TensorHidden::decay` and the batch cache multiply by `(1-decay_rate)`;
Add then aggregates the complete incoming fiber and adds it to the decayed hidden.
Chosen nodes can clear afterward, while this tick's output snapshot survives.

Implement a lazy Add state storing the post-candidate value and last tick. For
a candidate at theta, decay from last_time through theta, then add content.
Initial last_time=-1 means tick zero includes one decay. At a cut b, the original
physical hidden is obtained by decaying the stored value through tick b-1.
This decode is for comparison/readout; it must not manufacture idle graph events
or scan the whole graph during sparse advance. Include initial nonzero state,
idle prefixes/suffixes, missing samples, clear and token-cut continuation.

Finite precision requires an explicit policy: repeated per-tick multiplication
matches the original order, while power/scan regrouping may round differently
and change hard routing near ties. Start with an explicitly named repeat profile
as the original-code oracle. A separately named power profile can target sparse
long gaps, with numerical/route qualification and disclosed limits; do not
silently substitute power and claim original bitwise semantics. Retention can
be a learned scalar initialized from the original computed `(1-decay_rate)` in
payload dtype. Tide defines its VJP independently of LH's in-place/custom AD.

Local state APIs should carry this profile through scalar, independent-batch and
optional exact sequence contracts. Counters still count observations, not ticks.
Old-mode Read must have a declared interpretation of the stored representation;
LH's compatibility path reads the proposal. Complete cut roots and checkpoint
records must preserve the clock needed to decode physical hidden state.

## Original component comparison and later whole-model bridge

Extend the optional oracle to link untouched AccumulateLocal, Hidden, BatchHidden,
Confluence, ModuleUtils and their configuration dependencies. Test both original
single-sample and batch Add paths using the same frozen weights/tagged fibers,
then compare mapped hidden at every tick/cut and selected pre-clear outputs.
Keep ordinary Tide training anchors in Python: hand-computed values, input,
retention and initial-state VJPs; include cuts/detach/sharing and all executors.

Then map source signaling and CHAL activation/normalization into per-slot Full
programs. The current projection Emit alone does not implement original signaling.
Confluence source normalization must preserve source IDs and missing versus zero.
Only after these pieces fit should a tiny actual IOCortexNet/Pronounce run compare
weights, input occurrences, messages, hidden, selection history and logits.
Pronounce consumes a token window on its own clock; empty-output input domains
and its retained state need explicit handling. Same-fiber attention is another
profile, not the already-qualified aggregated-event attention.

## Remaining performance boundary

Current selectors copy/validate a touched history's full maps. Sparse cursor
advance avoids other region owners, but a huge single region's history can still
dominate. Preserve the functional baseline and add a validated history-patch or
persistent-map path for built-in profiles before claiming large sparse workload
performance. Do not expose mutable cross-owner state or lose immutable trace/cut
snapshots. This optimization belongs to ROADMAP's measured performance gate.
