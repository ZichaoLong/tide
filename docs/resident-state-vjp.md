# Device state-chain VJP

`tools/device_online/state_vjp.h` is an internal first-order component for the
resident training path. It accepts identity/EMA/Add-repeat state, HARD adoption,
observe-all/selected adoption, and selected clear. Other state kinds refuse with
device error 12. This is not yet a public autograd session or complete graph
backward/optimizer implementation.

The forward owner exposes a borrowed `StateTape` through `state_tape()`. It uses
the actual NPU journal and immutable module tables; requesting it requires
recorded forward values. Its lifetime ends at the next advance or close.
Capturing a reverse program does not preserve an earlier tape across overwrite.
Callers must retain a separate tape to differentiate multiple windows together;
the existing inference interface does not implicitly detach or promise that
training lifecycle.

The cotangent interface has separate values and connection bits for content,
old state, proposal, comparison, next state and the final state. A false bit is
an absent gradient, including when its unused storage contains NaN. A true bit
with numerical zero stays connected. Clear multiplies a connected gradient by
zero without disconnecting it. Adoption follows the recorded active decision;
HARD selection has no differentiable decision surrogate in this component.

A device metadata task validates bounds and int64 owner/time continuity and
builds predecessor links for each sample/node. Parallel feature tiles then walk
these reverse chains. No event count, predecessor or adoption decision is sent
to the CPU to drive reverse execution. Empty and shorter windows reuse the same
captured program and clear old output rows. Tensor-workspace admission precedes
output allocation; caller inputs and the enclosing `CannProgram` operator
workspace have separate budgets. Malformed metadata prevents the numerical reverse kernel from
writing a partial result; the caller must inspect the error at its boundary.

Returned content and initial-state adjoints retain connection bits. Parameter
adjoints are separate sample partials, before alias-aware parameter-owner
reduction; the public parameter registry remains responsible for ownership.
Disconnected raw parameters must not receive fabricated zero `.grad` tensors.

Add-repeat uses the declared periodic int64 state clock. A reverse step replays
the actual literal multiplications from the saved old state into a bounded
per-core scratch chunk, then differentiates those multiplications in reverse.
It never divides by retention or substitutes a power formula. Zero, one and
negative retention therefore keep their ordinary gradient connections.
`repeat_chunk_ticks` changes physical storage and prefix recomputation, without
changing the logical state interval or truncating its gradient. The requested
tensor budget limits concurrent feature tiles before allocation. A gap beyond
the forward owner's explicit tick-work limit refuses with device error 8.
Retention adjoints are sample/feature partials; reduction includes both axes
before accumulating into the scalar parameter owner.

The standalone gate compares the component with independent CPU FP32 and FP64
autograd, using mixed identity/EMA/Add nodes, selected/unselected adoption, clear,
connected zeros, poisoned absent values, padding, nonaligned feature widths,
int64 clocks/counters above 2^55, periodic Add clocks, zero/negative retention,
small/large replay chunks, empty replay and malformed-tape/budget/work refusals.
Separate integration cases consume a real resident forward journal in streaming
and greedy modes. Their passing status belongs in the immutable evidence record;
this document states the contract only. Profiling is separate from timing.

Full and Aggregate VJPs, message dependencies, HST/SOFTP, other state modules,
alias reduction, retained multi-window graphs, optimizer updates and multi-card
reverse progression remain required before claiming complete resident training.

## Low precision state adjoints

FP16 forward tapes retain FP16 decay/retention banks and the exact FP32-widened
forward journal. Cotangents and returned adjoints use FP32. EMA uses the sigmoid
coefficient computed in the forward parameter dtype; Add reconstructs every
literal tick with a round to FP16 before proceeding. Reverse products and sums
remain FP32. Changing the physical replay chunk never removes a rounding point.
This is a mixed precision adjoint, not bitwise PyTorch half-gradient accumulation.
The cast has its usual identity first-order VJP; no derivative of the rounding
staircase is asserted.

The half gate compares independent CPU quantized forward with FP32 autograd
adjoints at rtol=2e-3/atol=2e-5, retaining exact connection checks. Literal products
and sums round to half; cast VJPs pass FP32 cotangents, and the sigmoid VJP uses
its saved half coefficient. The initial CPU pure-half-backward comparison is
retained as a failed precision-contract mismatch, not relabelled passing. Its
maximum reported error was about5.05e-5 in a cancellation-sensitive case. The
corrected oracle keeps the same forward inputs and the original half tolerances. FP32/FP64 reference tolerances remain 1e-5/1e-6. A
separate 1024-tick rounding-sensitive CPU autograd anchor lifts quantized forward
products to FP32 adjoints and uses the strict tolerances. It must distinguish an
incorrect whole-forward FP32 replay by more than one in the scalar retention
adjoint; chunks17/1024 must both pass. Real half forward journals cover both
schedules. These are component checks; whole-graph FP16 reverse and public
training continue to refuse until the remaining adjoints/lifecycle are qualified.
