# Same-fiber attention oracle design

Scalar sum/tick-repeat baseline is qualified in `evidence/lh-attention.md`;
contract: `fiber-attention.md`. Real batch/sequence packing and CROSSBATCH are
qualified in `evidence/fiber-packing.md` (`fiber-packing.md`). Post-attention
Confluence is now qualified in `evidence/fiber-pooling.md` (`fiber-pooling.md`).
Pronounce and whole-model two-clock inference are qualified in
`evidence/pronounce.md` and `evidence/lh-iocortex.md`. Current backlog: ROADMAP.

Use the immutable snapshot in STATUS. `AccumulateLocal.cpp` projects every source
row, appends all current K/V, evaluates every current query against the entire
cache including all current rows, then applies Confluence and `c_proj`. The
existing aggregated-event attention is a different program. A triangular mask
inside one fiber changes semantics; a prefill mask must compare **event** order.

## Bounded profile and independent anchors

Start with explicitly named same-fiber **sum Confluence** and tick-repeat log-bias
decay. Preserve raw source tags/local slots and scaled input rows; do not collapse
the fiber to its Aggregate summary before Q/K/V projection. A candidate contributes
one observation but potentially many cache rows. First implement readable Python
and native scalar paths, then independent-batch and exact packed event-sequence
paths. Retain the scalar path as the VJP/reference anchor.

State holds key/value rows and per-key log bias in payload dtype, plus existing
last_time and observation count. Before appending, subtract the learned decay
scalar from previous biases once per elapsed logical tick, matching original
subtraction order. New keys start at zero bias. Explicit cut decoding advances
bias to cut-1 without idle events or whole-graph scans. Do not apply Add's
multiplicative retention to KV bias. Empty cache structure and clear/detach have
declared gradient connectivity; a numerically zero cache is still present.

Original `block_size` does not evict cache in the inspected code. Reject or
separately name any eviction policy. Query/key heads, qkv/output projection layout
and optional biases need explicit validation/import mapping. An initial cache's
row count is not bounded by the observation count. Complete state/checkpoints
must retain every memory slot and the tick clock.

Packed prefill should flatten real source rows with event/sample offsets, project
them in batches, and group compatible sample/cache shapes. Allow all current-fiber
keys, mask only later events and other samples, and preserve per-event output
pooling. Avoid one cross-sample quadratic score matrix. Compare scalar/packed
values, routes, all caches, final/pending/output roots and first-order VJPs;
test initial caches, clear, idle gaps, ragged fibers, source permutations, cuts,
sharing, serial/parallel, cyclic streaming, frontier and SettleGraph embedding.

Extend the original oracle to actual `Attention` and KVHidden paths. Start with
explicit per-sample packed/cached attention and add CROSSBATCH after matching its
cache/bias projection. Run original custom-autograd machinery only under no-grad;
Tide's training VJP is independent. No all-mode claim follows from one LH mode.

## Post-attention Confluence is a separate extension

For normalized/learned Confluence, weights act **after** attention. Multiplying
input rows by those coefficients before Q/K/V is not the same function. Existing
physical edge/input scales still act on arriving payloads. Preserve all-source
versus active-source denominator domains and missing versus zero rows.

Choose an explicit post-pooling program or typed coefficient content; do not try
to recover coefficients by dividing weighted contributions by input values.
Graph-owned slot count and canonical source order must remain available, including
missing sources. Original gathering orders internal CSC sources before appended
bridge/token inputs; an adapter must map local slots and numerical reduction
order deliberately. This extension must be separately qualified against original
Confluence routines before claiming those whole-model configurations.

The IOCortexNet adapter and Pronounce token clock have their own original-C++
gates; Add/Selector/Full component checks alone would not establish those results.
