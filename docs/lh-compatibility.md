# LH inference compatibility work item (M7)

Reference: `~/llm/lh`, HEAD `5fd237d40c9880ccb6e511e4bf20799c7022fd1e`,
inspected 2026-09-21. It contains user modifications to `BatchHidden.cpp`,
`bench-lh-small.cpp`, test graph data, and untracked build/data directories.
Do not modify or clean that tree. A future numerical run must record the actual
dirty-source snapshot, not identify it by HEAD alone. No original-LH numerical
comparison has been executed in this project yet.

Only the C++ interpreter is relevant. Python graph generation is useful;
the old Python interpreter is not an inference authority. LH supplies no
training contract; use this project's explicit state/VJP/truncation rules.

## Semantic mapping to implement and test

| LH operation | PositiveDelayGraph profile requirement |
| --- | --- |
| `CortexNet.cpp:think_single_step` reads prior activations through four adjacency blocks | Unit-delay edges; source signalling can move into the prior event Full for fixed inference weights |
| Candidate CHAL updates before selector | Observe-all proposal adoption; candidates are not restricted to selected nodes |
| Base-hub selector count/affect/norm/ID priority and other-hub forced activity | A dedicated region selector and explicit region partition; current count/descriptor profile is insufficient |
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

## Implementation reuse and next experiment

Reuse dual CSR/CSC indices, sample IDs, packed signal offsets and segmented
attention ideas. Audit old in-place/custom-autograd routines separately. The new
native thread pool already propagates Torch thread-local state and coordinates
commits; preserve these properties when borrowing kernels. Avoid nested OpenMP
and ATen oversubscription.

After M5 makes content/state programs extensible: build a read-only adapter from
an immutable snapshot of LH C++ sources using this project's CPU build setup.
Start with Add hidden and a tiny graph, then attention, multiple samples, idle
ticks, selected clear, token continuation and Pronounce. Save weights/inputs and
compare all mapped state/messages/counters as well as outputs in FP64/FP32.
If a true semantic mismatch survives an explicit encoding, document the minimal
counterexample and discuss the choice with the user before changing Tide semantics.
