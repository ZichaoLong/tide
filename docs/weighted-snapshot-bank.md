# KV snapshot storage and independent adjoints

This is an execution representation of fixed `0770593`
[KV §2–4](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md),
used by the default capability-selected [packed query](weighted-dense-query.md).
The mathematical state, int64 clocks, Read0/−/+ and Full snapshot choices,
graph/checkpoint versions and first-order gradient contract are unchanged.

`kv_packed` already writes one actual bank per sample/node window. The untraced
sequence VJP now returns immutable, potentially overlapping views of that bank,
instead of concatenating every prefix again. Distinct snapshot outputs still
have distinct gradient edges. Its existing adjoint reduction preserves an absent
future write's `None` and an expired but causally connected write's zero.
Complete tracing retains the intermediate-state chain.

`weighted/snapshot_bank.py` and `weighted_snapshot_bank.cpp` detect actual
contiguous storage overlap at the query boundary. Storage identity is used only
for lookup; banks are ordered by first appearance, not allocation address.
Equal numerical values or parameters alone never establish shared computation.
Independent or strided rows use ordinary packing. No public selector or topology
name is required; all requests keep their declared numerical execution identity.

For scalar/vector slot gathering, each unique interval is copied once before
the padded query gather. For Read's K/V projection, `Y_i = mask_i(X_i Wᵀ)` is
computed by projecting the union of visible unique entries once, then gathering.
Invisible entries are masked **before** projection; expired huge values must not
overflow a projection merely because another snapshot uses the same bank.
Visibility union uses int64 counts and original timestamps.

The custom first-order transpose returns each snapshot's independent
`dX_i = mask_i(dY_i) W`, and
`dW = Σ_i mask_i(dY_i)ᵀ mask_i(X_i)`.
It does not treat the shared numeric bank as a shared gradient root. The outer
query boundary still selects participating rows/fields, checks original versions
and sums all parameter users. Projection backward currently retains the padded
row matrix multiply; this increment does not claim a fully compressed adjoint.
There is no floating-point atomic reduction in this implementation.

Banks are call-local, hold actual computed values and do not cache parameter
versions across optimizer steps. State/checkpoint decoding reads actual slots;
it does not call a reference recurrence. Returned State slots may share storage:
treat them as immutable while a graph is live, and clone before external mutation.
Explicit detach preserves the existing truncated-gradient boundary and does not
promise storage isolation. Built-in query/region publication follows the
[read-only view contract](tensor-publication.md); explicit owned exports remain separate.

`tests/test_snapshot_bank.py` checks literal projection values/derivatives,
overlap/gaps/empty intervals, independent roots, partial/zero cotangents, ignored
huge entries, original version guards and Python/native queries. Existing
`test_kv_sequence_vjp`, `test_kv_prefix_roots` and query/consumer tests cover causal
roots, trace, general graphs, optimizer and exact continuation.

The additional FP32 cat-evidence scalar witness retains its strict failure:
maximum gradient difference `3.8445e-6`. Disabling bank reuse gives identical
gradients at that root. Same-material FP64 candidate/scalar difference is
`3.56e-15`; FP32 candidate/scalar distances to FP64 are `1.03e-6`/`2.83e-6`.
The separate witness test checks this bounded explanation; it does not relax the
strict test or certify full-scale numerical equivalence. Formal build/device and
performance evidence must be tied to the implementation commit.
