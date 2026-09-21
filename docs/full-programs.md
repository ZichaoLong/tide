# Full programs and sparse per-slot emissions

The interfaces and built-in profiles are [qualified](evidence/full-programs.md).
`FullKernel` (C++) and `FullProgram` (Python) evaluate selected events after Next.
They receive the complete read-only comparison State (value, slots, clock and
observation count), logical time, content, control and outgoing slot-domain size.
They cannot mutate state/weights or re-evaluate Next.

Return an optional auxiliary `value` and a sparse family of `(local_slot,payload)`.
The auxiliary value preserves the previous Full trace for built-in backbones;
it does not determine delivery. Slots must be sorted, unique and in range, with
payload width/device/dtype matching content. Missing slots produce no record.
Zero tensors are present and create candidates at their destinations. Delivery
visits present slots through the flat inverse index, applies existing physical
edge/output scales and preserves canonical public record order.

Next/history updates persist even if Full emits nothing. Conservative TimedDAG
frontiers may include downstream frames with empty actual fibers; these create
no node event or state/history update.

## Built-in profiles

Tanh/SwiGLU backbones retain `g = h + FFN(comparison.value)`.
`Node.emission="broadcast"` replicates the existing HARD/HST/SOFTP result.
Identity boundary adapters copy content and ignore Emit mode, with no parameters.

`Node.emission="slot_affine"` owns independent `emit_w_j` and `emit_b_j` parameters
in the node's extra map for every local output slot j:

```
held_j = h @ emit_w_j
fresh_j = g @ emit_w_j + emit_b_j
payload_j = Emit(held_j, fresh_j, control)
```

The declared HST VJP applies per present coordinate; cotangents into a shared
control sum normally. Never-used slot parameters remain disconnected. The
auxiliary Full value uses unprojected Emit, independently of slot weights.

`emit_period` is a positive integer. Empty `emit_phases` means all slots present;
otherwise one integer per slot specifies -1 always, -2 never, or a residue
matching `logical_time % period`. Identity adapters remain unconditional.
Logical time is not silently interpreted as token position.

Profiles and phase policy enter graph fingerprints and native format v6.
Checkpoint payload stays v3; incompatible graph identities fail before weight
changes. Shared modules require compatible policies/parameter slot domains,
independently of physical wiring. SettleGraph preserves body-local slots and
program modules when converting ports to edges.

## Batch and backward

Built-in batches evaluate the FFN over selected events, then group eligible rows
per slot for projection. They may span samples/times; unselected events are absent.
`batch` defaults to scalar evaluation. Programs advertise `joint_batch`, and
`full_scalar_fallback_steps` counts fallback work separately from batch API calls
and semantic replays.

Packed training uses local semantic autograd replay. Packed/reference slot
presence must agree; disagreement fails. Each packed tensor keeps its exact
forward value and binds to the corresponding scalar graph. Aliases to the
auxiliary value retain that dependency. No replay runs in no_grad/inference_mode.
Deterministic, functional scalar/batch equivalence is a program obligation;
custom inference programs are not double-evaluated to check it. First-order
qualification scope and training costs remain those in `packed-autograd.md`.

## Extension and navigation

C++ clients supply `NodeWeights::full_kernel`; its `validate_weights` checks the
parameter/slot domain. Synchronous `FullInput` borrows comparison only for the
call; do not retain/mutate that pointer. Native execution has no Python callbacks.

Python programs subclass the parameter-owning `FullProgram` nn.Module, declare
a versioned `profile` matching `Node.emission`, and are supplied through
`Model(..., full_programs={node: program})`. Python-only programs cannot silently
cross the native adapter; provide a corresponding C++ implementation.

Interface: `cpp/include/tide/full.h`, `python/tidegraph/full.py`.
Kernels/replay validation: `full_kernel.cpp`, `full_evaluate.cpp`.
Native delivery: `delivery.h`. Schedulers retain independent execution orders.
`cpp/test/custom_full.cpp` combines custom State/Full programs, reads pre-clear
slots/clocks and checks isolated native gradients. Tests: `test_emit_programs.py`,
`test_emit_schedules.py`, `test_full_contract.py`. Reviewed evidence remains under
`docs/evidence/`.
