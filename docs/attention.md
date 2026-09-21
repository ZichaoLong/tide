# Aggregated-event attention profile

Local profile `event-gqa-v1`, CPU FP64/FP32. This implements causal scaled
dot-product attention, GQA/MQA and optional sliding windows. Qualification
status lives in `STATUS.md`; this document specifies behavior, not performance.

## Equations and clocks

Node fields: `query_heads`, `kv_heads`, `window`. Query heads divide model width
and are a positive multiple of KV heads. Head width is `width/query_heads`.
Each nonempty fiber produces one content vector h through source-weighted sum.
Project Q=hWq, K=hWk, V=hWv, append one K/V observation, and attend with
softmax(QKᵀ/sqrt(head_width)), then Wo. Consecutive groups of
`query_heads/kv_heads` query heads share a KV head.

`window=0` keeps all accepted observations. A positive W retains the last W,
including the current one. Idle logical-time gaps consume no slots and cause
no decay. Proposals precede selection; selected-only adoption discards unselected
proposals. Window length, accepted observation count, logical time and token
position differ. No RoPE, absolute position embedding, dropout or per-message
K/V is implied. Zero-valued nonempty fibers still append observations.

Slots `key` and `value` are `[cached_observations,kv_heads,head_width]`. Read uses
the attention output. Selected clear preserves the comparison snapshot for Full,
sets the persistent read vector to connected zero, and empties K/V. Zero-valued
old keys must not remain in the softmax denominator. Clear preserves the last
observation clock/count. Empty slicing plus clone releases old storage in
inference while giving connected-zero cache VJPs.

LH appends every current fiber member before all current queries attend. That
requires a separate fiber-aware profile. Interpreting this profile's event
sequence as LH message order would change the semantics. Original-LH numerical
parity remains a separate milestone.

## Packed execution and limits

`python/tidegraph/packing.py` and `cpp/include/tide/packed.h` define nonempty
segments: flat contents, prefix offsets, `(sample,node)` owners, event times and
canonical tagged fibers. Each call has one node/program and unique owners.
Empty segments/unordered times fail validation. Padding never creates candidates.
Fibers preserve edge IDs/source positions; native views are borrowed during the
call and never stored in State.

Python step is an independent head-by-head oracle. Python/native prefill project
packed Q/K/V, group equal `(cache_length,sequence_length)` segments, and compute
`[B,Hq,T,K+T]` scores with causal and window masks. No cross-sample quadratic
matrix is created. Native streaming uses T=1. Native frontier `packed=True`
passes multiple samples to `packed_sequence`; `packed=False` passes one sequence.
Kernels without an override explicitly fall back to one sequence call per sample.

Region selection remains causal. Clear/selected-only adoption prevents state
prefill under the current contract; Full can still batch. Exact length grouping
can fragment ragged batches. Cache concatenation, head expansion and dense masked
scores are ATen correctness baselines. Ring/paged caches, length buckets and
specialized kernels need further work and measured evidence.

Final persistent read/KV tensors are compact clones, preventing narrow final
views from retaining a complete inference prefill allocation. Intermediate trace
states may share block storage. Autograd retains what backward needs until
backward/detach; compaction is not truncation.

`state_blocks` counts semantic sequences; `state_sequence_calls` counts actual
groups/calls. Native frontier reports `max_state_batch`, `max_state_sequence` and
`attention_score_elements` including masked entries computed. These counters do
not establish speedup. Streaming `update_calls` counts kernel entry calls, not
internal cache-length groups.

Graph fingerprints now include attention fields (native structural format v4).
Checkpoint payload stays v3; incompatible graph identities are rejected, without
implicit migration. Pretrained model support needs explicit position/RoPE,
normalization, layout and model-specific equivalence work. This representative
state program is not a complete open-weight model importer.
