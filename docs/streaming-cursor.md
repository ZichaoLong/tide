# Owned native streaming cursor

Functional `Streaming::run(continuation, inputs, stop, seal)` remains the simple
native oracle. It copies/validates the complete continuation and materializes
pending messages each window. `StreamingCursor` owns state/history/ledger and
the ordered event queue across windows. Graph adjacency, local programs and the
worker pool belong to a reusable `Streaming` engine.

```python
engine = Native(graph, model, packed=True, workers=3, trace=False)
cursor = engine.cursor(initial_continuation)
window = cursor.advance(new_inputs, stop, sealed_until=stop)
# window: cut, outputs, optional trace/messages and counters; no complete state.
saved = cursor.snapshot()  # Explicit export for inspection, loss or checkpoint.
cursor.detach()            # Explicit state-slot and pending-message truncation.
```

C++ interface: `cpp/include/tide/cursor.h`. The engine must outlive its cursors;
Python retains it. Calls sharing an engine serialize access to its pool, while
independent nodes within a call still run in parallel. Each cursor serializes
advance/snapshot/detach. Do not mutate model parameters concurrently with calls.

## Cost and ownership

Creation validates the full imported continuation once and clones its tensors,
preserving autograd under ordinary grad mode. Advances validate only new external
records and touched port ledger entries. They do not copy/scan all states,
reconstruct pending queues or compare full graph fingerprints. Scheduling visits
arriving fibers, relevant region histories and selected outgoing CSR edges.
Native execution invokes no Python.

`snapshot()` copies metadata, canonicalizes pending records and clones all
state/pending tensors. Mutating a snapshot cannot change cursor state. Its cost
is O(all retained state + pending), plus pending sorting. Clones retain VJPs in
ordinary grad mode; no_grad/inference_mode follow normal Torch rules.
`detach()` walks every state and queued message, making an explicit gradient
boundary. Checkpoints remain values-only autograd boundaries.

Treat inputs and `advance` result tensors as immutable while consumers need them.
Traces may expose views used by later computation; use snapshots or clones for
mutable exports. Local programs must be functional. Internally produced state
is trusted; arbitrary external in-place mutation is unsupported.

Extra counters: `external_records`, `cached_states`, `queued_times`. The latter
two are constant-time container sizes, not state-scan counts. A standalone custom
kernel counts validation callbacks: 512 imported states are checked once, while
subsequent advances touching two samples do not revalidate them. This establishes
a work-boundary property, not speedup or million-node workload performance.

## Failure and recovery

Input coordinate/tensor/position/seal checks finish before mutation. Rejection
leaves the cursor usable and unchanged. Exceptions during execution mark it
failed. Further advance/snapshot/cut/detach reject it; `failed` remains readable.
Recover by constructing a new cursor from an earlier snapshot/checkpoint. There
is no hidden whole-state rollback copy per successful advance. Workers finish
before execution errors return.

The functional API preserves its supplied continuation after an exception.
Both APIs share finite-valued event semantics. Failure timing is not an equality
guarantee for invalid custom kernels or numerical failures.

The cursor supports PositiveDelayGraph and TimedDAG streaming. Frontier requests
are not silently changed to streaming. Owned frontier execution, per-port online
watermarks, paged/ring attention caches and allocator optimization remain separate
work. Qualification status and evidence are in `STATUS.md`.
