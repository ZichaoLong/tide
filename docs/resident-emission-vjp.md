# Resident physical-slot projection adjoints

The HARD resident training path accepts `slot_affine` emission alongside broadcast,
including mixed graphs, physical parallel edges and explicit emission phases.
Implementation and fixed-source qualification are distinguished in [STATUS](STATUS.md).
This extends the existing public training/session APIs; it introduces no new
scheduler and consumes no reference event trace.

For a present selected slot, `y = g @ W + b` precedes physical delivery `z = s*y`.
Given connected cotangent `u` on that delivery, the device computes:

- Full cotangent `(s*u) @ W.T`, summed over connected present slots;
- weight cotangent `outer(g, s*u)` and bias cotangent `s*u`;
- delivery-scale cotangent `dot(u, y)` from the actual unscaled forward journal.

A zero scale still connects the projection with a numerical zero gradient and
may have a nonzero scale gradient. An absent slot has no connection. No division
by scale reconstructs `y`. Actual node/sample/int64-time/local-slot keys link the
candidate's emission journal to its consumed, pending and output messages on
device. Incoming messages produced before the retained cut remain boundary leaves.

Device-controlled reverse stages pack only connected affine rows into bounded
matrix batches. Broadcast rows bypass projection arithmetic. Distinct scratch
indices pad matrix batches; ordered owner reductions avoid conflicting scatter.
Static aliases merge before one SGD/AdamW update and publication to every actual
forward bank. Shared or unused owners retain the existing None/zero contract.
Retained tapes preserve projection parameters and unscaled slot values. Checkpoint
restore and continuation retain the same public schema and generation guards.

Public training retains one immutable copy of the physical projection weights
and biases for all windows in an update. Dynamic journals/state/cache still get
independent window copies. This is a frozen copy, not a borrowed forward bank;
the owner prohibits parameter publication while those windows are live. Backward,
detach and close release the shared snapshot, and the next update captures the
new parameter version. The general internal tape-copy API keeps its original
independent-copy contract unless the guarded training owner supplies this cache.

Retained admission charges projection snapshot bytes once, plus per-window
reservations. A window that would exceed the declared budget is rejected before
device progress. Backward statistics expose `retained_projection_bytes`,
`retained_window_bytes`, `retained_windows` and `retained_bytes`; these are tape
reservations, not total allocator peaks. This removes repeated projection copies,
but does not partition the projection bank or its physical gradients. Current
implementation and qualification status remains in [STATUS](STATUS.md).

FP16 keeps actual half forward operands and stored slot values. Cast VJPs and
parameter accumulation use FP32, as with other resident modules; this does not
claim equality to half-gradient accumulation. CPU FP32/FP64 autograd and an
independent rounded-forward reference check the declared first-order semantics.

Full/state/cache execution and canonical optimizer owners may span devices.
**Projection forward banks and their physical partial gradients currently remain
on the coordinator.** Compact projection placement and a total scale budget are
still needed before full-size resident delivery. HST/SOFTP slot-affine emission
remains explicitly refused: mixing before projection would mishandle bias/control
adjoints. Existing broadcast HST/SOFTP behavior is unchanged.

Affected trajectory gates use `--emission` on the existing standalone public
sharded-session and FP16 training checkers. They compare continued windows, all
parameter and boundary gradients, zero/None roots, optimizer state, updated
parameters and restored suffixes. The Python public gate is
`tests/test_resident_emission_training.py`. Profile and throughput runs remain
separate from these correctness checks.
