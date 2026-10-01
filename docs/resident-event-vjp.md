# Resident event attention and KV adjoints

This contract extends the optional single-NPU FP32 training owner with the
existing [aggregated-event attention](attention.md) formula. Implementation,
development checks and fixed-commit qualification are distinct in STATUS and
ROADMAP. Fiber attention has a separate cache/decay/pooling formula and is not
implicitly covered by this extension.

The candidate's actual forward journals retain old/proposed KV coordinates and
values. Reverse builds device hash associations to the candidate's actual
events in expected linear work, without a CPU numerical prepass or searching
every event for every cache row. Retained tapes freeze the projection banks and
cache journals that generated each window.

Visible state reverse sends attention proposal cotangents to the attention
adjoint. It does not connect the proposal to old visible `state.value`: that is
an EMA dependency, absent from event attention. Read of old visible state still
has its own declared connection. Adoption, selected clear and soft/hard control
retain their ordinary meaning.

Within each actual reverse graph stage, the device packs independent cache
owners into a batch and advances each owner's recorded predecessors. Query,
Q/K/V projections, shared GQA head reduction, output projection and parameter
partials are batched. Physical key tiles use one global softmax denominator
and Jacobian correction; independent per-tile normalization is not equivalent.
The current reverse cache dependency loop may use more iterations than forward
node-time prefill. No throughput advantage is implied without measurement.

Cache adoption, eviction and clear preserve structural connection separately
for keys and values. A final key-only root can connect Wk while Wq/Wv/Wo remain
None. A connected empty slice produces connected-zero gradients through the
concatenation, including evicted rows and empty initial tensors. Discarded
proposals do not acquire cache gradients from the adopted old cache. None is
never inferred from numeric zeros; absent and padded root payloads may be NaN.

Q/K/V/O parameter partials are flattened without padding shared KV heads. The
ordinary registry reduces all declared aliases, including sharing across nodes
or projection roles. Device optimizer publication updates every corresponding
forward bank. A step ends the differentiation generation but preserves actual
numerical KV continuation. Checkpoints retain that cache with updated parameters
and optimizer state; cached keys are not reprojected using newer weights.

## Public cache roots and initial gradients

`ResidentTrainingWindow.cache` is a list grouped by static head geometry. Each
group exposes a static `nodes` list and device `key`, `value`, `lengths`, `present`.
Event groups precede any [fiber groups](resident-fiber-vjp.md); their additional
`log_bias` field is undefined and event groups reject log-bias cotangents:

- Key/value shape: `[sample_count * len(nodes), capacity, kv_heads, head_width]`.
- Owner order: sample first, then the node order in `nodes`.
- Lengths are int64; presence is bool, both `[owners]`.
- Only the prefix given by each length is logical cache content. Present empty
  caches have length zero; they are different from absent states.

`ResidentCotangents.cache` supplies one key/value cotangent record per group,
or an empty list for all-disconnected cache roots. Each value/connection pair is
supplied together or omitted together. Key and value connections are separate
`[owners]` masks. They may select an empty cache but not an absent state. Shapes,
device, dtype and root ownership are checked before reverse submission.

Python `session.cotangents(window, cache=[...])` accepts per-group dictionaries
with `key`, `value`, `key_connected`, `value_connected`. Omitted masks default to
that group's state presence; omitted values mean disconnected roots. Consumers
forming a loss from padded views must mask padding according to `lengths`.
They must not mutate the retained window's borrowed tensors.

`ResidentGradients.initial_cache` carries the same node/owner convention, actual
initial lengths and separate key/value connections. Only states present at the
generation's incoming cut are caller-owned leaves. Each retained-window bridge
adds the later window's actual initial cache gradients to the earlier final
cache roots on device and validates matching lengths, including empty windows.
There is no implicit detach between physical batches or retained windows.

Forward KV/journal limits, retained tape bytes, reverse tensor and operator
workspace budgets are separate. Capacity errors refuse explicitly; no cache is
silently truncated to make training fit. This implementation retains complete
per-event cache journals and bounded batched reverse prefixes. Reducing that
training storage or adding key recomputation is a separate optimization with
the same roots, connections and continuation contract.

## FP16 local attention component

The packed attention adjoint also accepts FP16 Q/K/V, with FP32 additive bias
and public cotangents. It recomputes QK in actual payload precision before FP32
scaling and global softmax. Its adjoints and GQA head reductions stay FP32.
The softmax correction uses the complete unrounded weighted sum across all key
tiles; only the saved output for a downstream projection is half-rounded and
widened. A separate softmax per physical tile, an unrounded QK product or a
rounded softmax correction would change this contract.

The component gate uses independent CPU quantized-forward/FP32 and FP64 adjoint
references. Ordinary half tolerances are2e-3/2e-5; the original FP32/FP64 checks
remain2e-5/2e-6. Two strict half fixtures use different physical key tilings and
must distinguish missing QK rounding. Empty/short/full prefixes, connected-zero
roots, poisoned padding, GQA, memory refusal and replay reuse remain checked.
## FP16 event cache and projection components

Event reverse also accepts actual half forward cache/projection banks. The
candidate independently generates its event/state/cache journals from public
inputs. Reverse restores half operands and recomputes the QKV projection in
half, then uses the local attention adjoint above. The output projection VJP
uses the actual half-rounded attention result. Projection/content gradients,
cache roots, cache carry and boundary sums remain FP32; disconnected operands
are sanitized before projection, without evaluating unused parameters.

The event component gate compares its actual journals with independent CPU
quantized-forward FP32/FP64 autograd. It covers adoption, selection, clear,
window eviction, GQA, streaming/greedy, independent-owner reverse batches,
None/zero roots, empty incoming caches, poisoned padding and replay reset.
Half uses rtol2e-3/atol2e-5; FP32 retains rtol1e-5/atol1e-6. An isolated cache
boundary gate checks FP32 sums outside the half range, connected empty/zero
caches and rejection of incompatible lengths, invalid roots and small budgets.

These components do not enable half fiber reverse, complete graph backward,
retained-window graph training or the public training owner. Each integration
has a separate gate; isolated cache sums do not certify end-to-end continuation.
