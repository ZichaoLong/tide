# Interfaces, state and autograd

Use the [architecture map](../../../../docs/architecture.md) for current symbols,
[semantics](../../../../docs/semantics.md) for observable behavior and versions,
and [prefill lessons](../../../../docs/performance-lessons-prefill.md) for concrete
positive and negative results. This is a design method, not a second API contract.

## Follow the batch through every boundary

Inspect scheduler output, local input preparation, numerical operation, saved
state, backward, validation, transfer and result publication. Record B, logical
time ranges, ragged offsets, owner, dimensions, dtype/device and gradient roots.
Preserve a legal batch through these boundaries instead of repeatedly unpacking
and rebuilding it. General local programs may need smaller batches; capability
fallbacks must express that fact, not retain incidental historical restrictions.

Within a legal, capacity-admissible window, aim for major submissions and reverse
assembly organized by nodes, regions and stages rather than individual B×L events.
Attention still has required query/history work; feedback, state-dependent
selection and capacity can require real splitting. Metadata describing such
dependencies must be available without a reference trajectory prepass.

## Give stored data an owner

A unique KV bank with range/index queries can avoid materializing every history
prefix. Preserve original int64 time, visibility, expiry, zero writes and the
Read/Full snapshot distinction. Reuse projections only for the same parameter
version and input identity; equal values do not establish shared gradient roots.
Private node state remains private even when parameters or computation are shared.

Distinguish persistent state, retained differentiable state, readonly input,
scratch storage and exported snapshots. Define parameter-publication and window
lifetime rules. Reusing scratch requires completion of all readers, including
asynchronous work and backward. Explicit snapshot materialization is an API choice;
it need not force one mutable Tensor allocation per internal field per event.

## Design the transpose with the forward

No-grad buffers are useful for numeric caches and explicit VJPs; detaching a
cache without reconnecting its input derivatives loses training semantics.
Batch the transpose of gather/scatter/segmented operations and accumulate every
shared use. Preserve disconnected versus connected-zero inputs where declared.

Inspect whether a custom Function actually avoids fine-grained graph work. It
may instead reenable autograd for thousands of private leaves and later call
nested `grad`. Ordinary autograd is appropriate for a few large GEMMs/Attention/FFN
operations; custom boundaries should remove the proven structural cost.

Packing many rows into one Tensor is not sufficient. A per-row view transpose
can repeatedly allocate and zero a full packed gradient. Use a block transpose
or segmented accumulation; account for its memory, ordering and connectivity.
The [rejected query-column trial](../../../../docs/evidence/fc16-cost-attribution-20261110.md)
is a specific example, not a prohibition on packed adjoints.

## Expose the observations the program needs

Specify ordinary output, trainable boundary state, intermediate observations,
complete trace and checkpoint export. Intermediate roots with causal relations
cannot become independent sibling outputs merely because they share a Function.
Decode actual candidate data; do not fill missing records by running a reference.

Readonly views, explicit copies, owner handles and requested root sets can make
the common path cheaper. Changing public mutability/aliasing or return shape is
an explicit software-contract change with migration. The fixed mathematical
capabilities remain testable through the new representation; old per-field
cloning or wrapper shape need not become a permanent architecture constraint.

## Select meaningful checks

Exercise isolated and combined roots, repeated and aliased inputs, frozen/shared
parameters, missing/zero messages, short/long and empty histories, window expiry,
large logical times, retained cuts and fresh-process restore. Include gradients
from independent losses and all shared users. Check optimizer slots as well as
parameter values. Follow affected paths across the actual clients and backends.

Keep syntax, memory/layout, mathematical identities and numerical trajectories
separate in the tests. Update tests tied solely to an intentionally replaced API;
retain independent formulas and prior strict numerical witnesses with their
original scope. A mechanism-level change can be adopted after relevant checks;
representative whole-path measurements establish its integration cost.
