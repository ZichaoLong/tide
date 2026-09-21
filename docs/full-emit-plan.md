# Next increment: per-slot Full/Emit programs

Status: planned; stable local ports are qualified in `evidence/local-ports.md`.
This document specifies the next implementation gate, not current API support.

## Public boundary

Add a native immutable `FullKernel` program, injectable by a C++ client like
`StateKernel`. A request contains the complete read-only comparison State,
logical time, aggregated content and selection control. Next has already run;
Full cannot modify its persistent result or call Next. Graph wiring stays outside
the parameter module. A program sees the node's outgoing slot domain, not physical
edge/output IDs.

Return a sorted, unique sparse vector of `(local_slot, payload)` plus an optional
auxiliary `value` for compatibility with the existing Full trace. Missing slots
are absent messages/outputs; zero tensors are present. Validate slot bounds,
uniqueness and payload metadata. Delivery resolves each present slot with the
flat outgoing index, applies existing physical send/output scales and queues or
publishes the result. Preserve public canonical record order even with permuted
slot layouts. Trace records the tagged emitted family separately from the old
auxiliary Full value.

Programs expose scalar evaluation and an optional packed selected-event batch.
The default batch is an explicit scalar fallback. Built-ins use batched FFN
computation across samples/times; report packed work and scalar fallback/replay
work separately. No Python callback belongs in the native execution path.

## First concrete profiles

Keep the existing tanh/SwiGLU backbone and broadcast Emit numerically compatible.
Separate fresh Full computation from its HARD/HST/SOFTP combination where needed.
Add a slot-affine profile with independent parameters per outgoing slot:
`held_j = h @ W_j`, `fresh_j = g @ W_j + b_j`, then declared Emit on that pair.
Use separate parameter tensors so a never-used slot can remain structurally
disconnected. The profile is an example, not the native extension limit.

Add a declared integer-time phase policy as a small absence example: positive
period, one phase per slot, -1 for always and -2 for never. Empty phase metadata
means all slots present. Persist static policy/profile identity in graph
fingerprints. Identity boundary adapters remain unconditional stateless copies.
Logical time is explicit; do not silently reinterpret it as token position.

The exact policy/config names can be refined while implementing, but their
meaning and checkpoint guards must be explicit. Shared modules are valid only
when their program/parameter slot domains are compatible; mappings may differ.

## Autograd and schedules

Use the qualified semantic replay approach for packed Full families. Packed and
scalar paths must return the same slot set; fail on a disagreement, never replace
packed routing by reference routing. Bind each packed tensor to the corresponding
scalar graph. Preserve alias relationships to the auxiliary value where practical.
No replay runs under no_grad/inference_mode. Retain isolated-root None/zero tests.

Python reference, frontier/Settle blocks and independent fixed-topology schedules
must all consume per-slot outputs. Native streaming/cursor, frontier and native
fixed schedules must do likewise. Empty emissions stop propagation without
undoing the candidate's Next/history update. Conservative potential-event
frontiers may still include now-empty frames; those create no node event.

## Acceptance and file boundaries

Separate program interface, built-in kernels, packed/replay helper and delivery
from schedulers (`full.h`, small native source files, a Python Full module).
Retain complete comparison State privately until Full returns; Python trace
currently keeps only its value/slots, which is insufficient for a custom Full
reading clock or observation metadata.

Verify hand-computable distinct edge payloads, parallel edges, absent versus zero,
phase routing in a cycle, sparse continuations, all Emit modes, isolated gradients
and optimizer behavior. Exercise Full reading pre-clear comparison, output roots
from only one slot, shared modules with permuted layouts, direct/encoded
SettleGraph, serial/parallel/batched/step schedules and standalone custom C++ Full.
Keep full FP64/FP32 qualification tied to a frozen implementation commit.
