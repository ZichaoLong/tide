# Built-in tensor publication: read-only views

The built-in weighted batched Read, Full and regional VJP boundaries publish
read-only tensor views (`readonly-view-v1`). KV sequence State slots can also
share their actual owner bank. The API keeps its tensor structures and independent
gradient roots, but no longer promises separate storage/version counters for every
returned row or field. This is a deliberate software ownership migration.

For fixed `0770593` mathematics, observations are values, not writable storage
allocations. Returning `ref.detach()` from the custom boundary attaches the
declared outer VJP while retaining the private graph's original `SavedVariable`
version checks. It does not identify distinct gradient roots, multiply message
strength again, change `None`/connected-zero gradients or alter parameter sharing.
The candidate computes every returned value; no reference fills omitted records.

Treat returned tensors as immutable while using a result or its graph. A consumer
that needs a mutable, gradient-disconnected copy uses:

```python
mutable_value = returned_value.detach().clone()
```

Out-of-place tensor operations remain differentiable as usual. `clone()` without
detach retains the gradient edge when that is wanted. Do not use `.data` or
modify a returned view and expect other fields or future backward calls to be
unaffected. Saved-version failures detect invalid live-graph mutation where the
value is saved; inference views are not a runtime-enforced immutable tensor type.

This removes the default allocation/copy cost proportional to every published
field. The Python/native query and region boundaries implement the same ownership
contract. Scalar and custom program fallbacks can own extra storage, but consumers
must not depend on that incidental isolation. Public return-to-host packing is a
separate explicit owned-copy operation; [its isolation contract](complete-return-packing.md)
remains. Checkpoints still encode actual mathematical values; graph/checkpoint
versions do not change. Installation content/source hashes identify the new
software contract; active downstream locks are not silently upgraded.

`test_independent_copies` now verifies explicit clone migration and rejects an
invalid live private-graph mutation, alongside the opt-in owned-copy helper.
Query/region root tests still compare independent fields, partial and zero
cotangents, shared parameters, original versions and complete training/resume.

Row decoding inside batched queries also uses one `unbind` per field instead of
one tensor `select` per returned row. This composes the same linear view map,
including partial cotangents, with one transpose node. Native Full and Describe
and Python/native packed KV memory use this rule; it does not eliminate all row
objects or claim a fixed kernel count for arbitrary feedback graphs.
