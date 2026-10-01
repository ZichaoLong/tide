# Resident same-fiber attention training

The optional single-NPU FP32 training owner implements the existing
[same-fiber attention](fiber-attention.md) and five [pooling](fiber-pooling.md)
profiles. STATUS distinguishes implementation, development checks and clean
fixed-commit qualification. This contract implies no throughput advantage.

The candidate independently produces its source/event and old/proposed cache
journals. Device hash association matches those actual records. A stable device
sort orders each complete fiber by logical source slot while retaining physical
message and edge identities. It neither substitutes aggregate-as-token attention
nor uses CPU reference events. Static topology and parameter ownership may be
prepared on the host.

Within a reverse graph stage, device control advances the recorded predecessor
chain for each cache owner. Independent owners are packed into bounded batches.
Queries can be chunked and keys tiled, preserving complete current-fiber KV
visibility and the global softmax denominator/Jacobian. Source, QKV/bias, output
projection/bias, pooling and repeated-tick decay adjoints are batched. Direct
source adjoints add to ordinary Aggregate/Read/Full contributions; they do not
replace those independent paths. Physical source scales receive both terms.

Cache adoption and clear act separately from visible state. Rejected proposals
retain the old cache carry. Selected clear keeps connected-zero adjoints through
the empty slices. Key, value and log-bias roots have separate connection flags;
numeric zero never means None. Disconnected or padded payloads may be poisoned
and must be masked before arithmetic. Decay follows the declared local clock,
with a bounded repeated-tick adjoint; no gradient is fabricated for an empty old
cache or a zero-tick subtraction that was never executed.

The parameter registry reduces shared uses, including sharing across Full and
fiber projections. Device SGD/AdamW publishes into the forward owner's actual
banks. Grouped reverse snapshots are gathered copies and are never publication
destinations. A step detaches the generation while retaining numerical cache
state; old keys are not reprojected with updated parameters.

## Public roots and retained windows

The [training API](resident-training.md) uses the cache owner layout from the
[event contract](resident-event-vjp.md). Event groups precede fiber groups;
each collection is grouped by static head geometry. A fiber `CacheWindow` adds
`log_bias [owners,capacity]`. Event groups leave that field undefined.

`ResidentCacheCotangents` and Python per-group dictionaries accept `log_bias`
and `log_bias_connected [owners]` as an independent pair, alongside key/value
pairs. An omitted pair is disconnected; an omitted connection mask in Python
uses state presence. Event groups explicitly reject log-bias roots. Consumers
forming losses must mask cache padding using `lengths`, including log-bias
padding. Returned views must not be mutated.

`initial_cache` returns key/value/log-bias gradients and independent connection
flags for caller-owned states at the incoming generation boundary. Absent states
have no caller leaves. Present empty tensors may have connected-zero gradients.
Retained tapes clone the actual source/cache records and projection/pooling
banks on device. Bridges add the next window's initial cache adjoints to the
previous window's final roots without implicit detach, including empty windows.

This first implementation retains complete per-event cache journals. Forward,
retained tape, reverse tensor and CANN workspace budgets are separate and finite;
capacity refusal is explicit. It does not silently truncate KV, roots or the
differentiation horizon. Cache dependency loops can have more iterations than
forward prefill. Storage/recomputation optimization, FP16 and peer training are
separate work in ROADMAP, not claims established by this implementation.
