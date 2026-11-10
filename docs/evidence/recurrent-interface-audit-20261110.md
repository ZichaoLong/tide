# Recurrent, convolution and Describe interface audit

Review source: `06b242eef2a203a950e30ced9b50d5c6d1cb8a73`.
This is a source/contract review, not a performance or device qualification.
The [hash-bound manifest](recurrent-interface-audit-20261110-reviews.json)
records the actual files and Python definitions reviewed. C++ files were read
manually; their selected symbol anchors are not an exhaustive parsed C++ census.
Later edits and runtime evidence must retain their own implementation identity.

## Review scope

The review followed five memory families through scalar formulas, raw state,
packed updates, query/Full batching, private query/state VJPs, fixed-logical-time
convolution and parameter rebinding. It also read the listed independent formula,
int64, packed-state and root tests. It did not inspect every repository function,
qualify arbitrary compositions, or measure new end-to-end throughput.

## Findings and implementation boundaries

1. **Describe capability was unnecessarily tied to KV query selection.**
   `dense_describe.evaluate` uses the same cat/ind/regional_total/context/cat
   features and three shared projections for recurrent and KV memories, but
   `query_execution.execution_view` selected it only with a KV dense query.
   The subsequent change separates memory-query and Describe capabilities:
   recurrent memory queries remain original, while auto can batch these heads.
   Explicit legacy still selects the scalar head. The contract and actual
   counters are in [weighted-dense-query](../weighted-dense-query.md).
   The new `test_weighted_describe_capability` checks all ten legal family/read
   pairs, private input/state roots, shared/frozen parameters, individual fields,
   repeated use, connected zeros and the actual three projection batch shapes.
   The C++ batched head publishes each field with one unbind rather than a select
   node per row. This does not by itself remove private VJP grouping.

2. **Packed forward does not imply a sequence-level recurrent VJP.**
   `recurrent_packed.packed` and native `packed::recurrent` prepare coefficients
   jointly and scan along time. `state_vjp.bind_sequence` / native
   `state_sequence_vjp` still advance through time depths and create private
   one-step graphs; their grouping signatures include actual timestamps and
   observation counts. Zero scalar-replay counters do not prove zero repeated
   private construction. A future causal sequence transpose can prepare legal
   input-only coefficients once, but must retain intermediate-state root
   causality rather than connect a root to sibling future states. A smaller
   candidate is a recurrent-specific grouping signature that omits metadata
   equality only after proving each row's calculation consumes that metadata.

3. **Convolution retains repeated validation and cache snapshots.**
   Query convolution validates each cache/query with device boolean reads.
   Packed convolution assembles an owner bank, then scans it with a fresh time
   mask for every returned cache, costing O(L × (old-cache + L)) index work.
   Batched guards and an owner/range representation can reuse existing lessons,
   but raw zero-receive retention and exact int64 time visibility remain required.
   The scalar formula is
   `C(t)=bias+W[0]*z(t)+Σ(j>=1) W[j]*(w(t-jδ)*z(t-jδ))`:
   the current tap is unweighted and stored past values are weighted. A vendor
   convolution on weighted inputs therefore needs an explicit tap-zero rule and
   dense-lattice applicability check. Sparse original times cannot be compacted.

4. **Some recurrent queries calculate unused branches.**
   `query_batch.effective` always constructs H/Z/W even though sum Read uses only
   Z/W and Full uses only H. Nonbasic state Read computes effective state again
   inside `matrix_read`. `features` also computes GDN key normalization, SSD key
   expansion and step coefficients that Read/Full do not use. Purpose-specific
   helpers can remove work while keeping the scalar formula's distinct RMS
   branches, None/zero structure and full raw checkpoint state.

5. **The matrix scan has a separate algorithmic cost.**
   Delta/GDN materialize dense K×K transition matrices. `matrix_scan` uses
   Hillis–Steele composition with O(log L) stages and K³ matrix products, whereas
   the one-step update is rank one. A bounded chunked/rank-structured comparison
   is warranted; replacing the scan is not the same low-risk change as removing
   an unused projection. Affine/SSD use scalar-transition affine scans.

6. **Singleton state boundaries warrant separate inspection.**
   Query/Aggregate can avoid a private nested backward for one row; generic State
   still constructs detached private leaves. Any alternative must retain numeric
   scan results, first-order/version guards and independent intermediate roots.

These findings are review observations, not another live backlog. Current
selection, progress and stop/restart conditions belong in [ROADMAP](../ROADMAP.md).

## Historical experience applied

Core-3 Full/Aggregate batching and LH's batched cache/gather/adjoint returns support
keeping work at a block boundary. Their evidence does not certify core-5: the
historical large LH measurement records grad-enabled forward, its clock is an
activation clock, and its K/V history is not a permanent single gradient owner.
The rejected private-query column packing also shows that a contiguous buffer
alone can cause expensive full-buffer transposes for every row view.
See [batching lessons](../performance-lessons-batching.md) and the
[performance index](../performance-lessons.md).

The Describe change preserves five named CPU FP32 near-radial evidence strict
failures instead of serializing projections. `diagnose_describe_rounding.py`
promotes the same FP32 parameters, raw states and input leaves to FP64: ten legal
family/read cases differ by at most 6.50e-15 across the checked FP64 roots; the
five FP32 evidence-weight/bias witnesses differ by 1.04e-6–3.49e-6. Each FP32
implementation is within 1.84e-6 of the FP64 formula at those witnessed owners.
This is finite local diagnostic evidence. Tests keep the original strict
thresholds, separately name these radial failures, and continue checking the
remaining fields and nonradial roots. Native/device/full-training qualification
is recorded separately by the integration work.
