# LH inference compatibility work item (M7)

Reference: `~/llm/lh`, HEAD `5fd237d40c9880ccb6e511e4bf20799c7022fd1e`,
inspected 2026-09-21. It contains user modifications to `BatchHidden.cpp`,
`bench-lh-small.cpp`, test graph data, and untracked build/data directories.
Do not modify or clean that tree. Numerical runs record the actual dirty-source
snapshot, not HEAD alone. Original-selector equivalence is qualified in
`evidence/lh-selector.md`. Tick-repeat Add and its original-C++ component oracle
are qualified in `evidence/lh-add.md` (`lazy-add.md`). Whole-LH inference
comparison remains pending.

Only the C++ interpreter is relevant. Python graph generation is useful;
the old Python interpreter is not an inference authority. LH supplies no
training contract; use this project's explicit state/VJP/truncation rules.

## Semantic mapping to implement and test

| LH operation | PositiveDelayGraph profile requirement |
| --- | --- |
| `CortexNet.cpp:think_single_step` reads prior activations through four adjacency blocks | Unit-delay edges; source signalling can move into the prior event Full for fixed inference weights |
| Candidate CHAL updates before selector | Observe-all proposal adoption; candidates are not restricted to selected nodes |
| Base-hub selector count/affect/norm/ID priority and other-hub forced activity | A dedicated region selector and explicit region partition; `lh-count-affect-v1` now provides the ordering; full-capacity regions provide forced activity |
| Clear after selection | Next resets selected memory but Full retains the pre-clear comparison snapshot |
| Tensor hidden decays every tick, including idle | Lazy `(memory,last_tick)` interpretation; repeated multiplication vs exponentiation needs a numerical policy |
| KV decay subtracts decay rate from attention log biases | A separate decay law; not tensor multiplicative decay |
| All same-fiber K/V are appended before queries attend | Same-event visibility is all-to-all; ordinary triangular order within the fiber changes LH |
| `Pronounce` gathers `n_layer` output occurrences once per token | Explicit window/readout clock and source-occurrence tags; not an ordinary next-tick output node |

For Pronounce, a candidate fixed-graph encoding uses one phase-tagged edge per
tick phase, with a constant delay per edge so all occurrences reach the token
readout event together. Full selects the phase edge from its logical clock.
This is a design to verify, not an established numerical compatibility result.
Empty-output windows need an explicit valid-input/readout protocol matching the
original C++ behavior. Preserve Pronounce's own hidden state and decay clock.

Map LH activations to in-flight transformed messages, hidden to node state,
selector counters to region history and token-window readout to its own state.
State projection must include all of these. Comparing logits alone is inadequate.

Further source audit: signaling produces one distinct vector per outgoing edge
(`CortexNet.cpp:EmitToEdges`), so replicated scalar-scaled emissions alone do not
instantiate LH. Confluence distinguishes active-only softmax normalization from
all-source softmax, preserving incoming source indices. The attention `block_size`
configuration is stored but does not evict KV entries in the inspected C++ code;
do not map it to Tide's new window setting. Add decay occurs before each tick's
update, including idle ticks. Selector norm scores explicitly accumulate in FP64.

## Implementation reuse and next experiment

Reuse dual CSR/CSC indices, sample IDs, packed signal offsets and segmented
attention ideas. Audit old in-place/custom-autograd routines separately. The new
native thread pool already propagates Torch thread-local state and coordinates
commits; preserve these properties when borrowing kernels. Avoid nested OpenMP
and ATen oversubscription.

For the remaining whole-model comparison, build a read-only adapter from
an immutable snapshot of LH C++ sources using this project's CPU build setup.
Start with Add hidden and a tiny graph, then attention, multiple samples, idle
ticks, selected clear, token continuation and Pronounce. Save weights/inputs and
compare all mapped state/messages/counters as well as outputs in FP64/FP32.
If a true semantic mismatch survives an explicit encoding, document the minimal
counterexample and discuss the choice with the user before changing Tide semantics.

## Next bounded gate after region programs

Typed region histories and SelStep extension interfaces are qualified in
`region-programs.md`. The LH selector profile with separate selected/affected maps
and graph-owned canonical membership is implemented (`lh-selector.md`).
Selection compares **prior** counts; all actual candidates then increment affect,
and only active nodes increment selected. Forced-activity hubs use other regions.

The explicit FP64 norm Read and payload metadata/conversion policy are also
implemented; see `lh-selector.md` and STATUS for qualification. The generic
region request carries dtype/device separately from descriptors/history.

Re-reading `Selector.cpp` and `Selector.h` confirms that original LH count storage
is **int32**, while ranking uses double. The heap compares lexicographically;
the tensor path builds a composite double score with successive multiplications.
Adding one to every candidate affect count preserves ordering only while counts
and composite arithmetic stay in a safe range. Do not claim equivalence at
int32 overflow or beyond exact composite-score precision.

Both original C++ selector paths are qualified against an immutable actual-source
snapshot with ragged batches, ties, empty candidates, sparse local IDs,
forced activity and small counters. Active sets and both counter maps agree;
see `evidence/lh-selector.md`. This is not whole-LH inference parity. Add's
encoded/physical clocks are specified in `lazy-add.md`; same-fiber attention,
source signaling and Pronounce still need the complete projection above.
