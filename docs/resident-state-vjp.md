# Device state-chain VJP

`tools/device_online/state_vjp.h` is an internal first-order component for the
resident training path. It currently accepts identity/EMA state, HARD adoption,
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

The standalone gate compares the component with independent CPU FP32 and FP64
autograd, using mixed identity/EMA nodes, selected/unselected adoption, clear,
connected zeros, poisoned absent values, padding, nonaligned feature widths,
int64 clocks/counters above 2^55, empty replay and malformed-tape/budget refusals.
Separate integration cases consume a real resident forward journal in streaming
and greedy modes. Their passing status belongs in the immutable evidence record;
this document states the contract only. Profiling is separate from timing.

Full and Aggregate VJPs, message dependencies, HST/SOFTP, other state modules,
alias reduction, retained multi-window graphs, optimizer updates and multi-card
reverse progression remain required before claiming complete resident training.
