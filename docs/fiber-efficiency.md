# Fiber execution and storage options

These opt-in execution policies preserve the canonical complete-fiber
Agg/Upd/Read/SelStep/Next/Full spine in [semantics](semantics.md). They do not
change graph identity, checkpoint values, logical clocks, routes or parameter
sharing. Defaults retain the previous implementation for controlled ablations.
They complement, independently, `attention_packing="exact"|"single"`.

| Native Python keyword / scale CLI | Default | Alternative |
| --- | --- | --- |
| `fiber_pooling` / `--fiber-pooling` | `event` | `csr`: one sparse pooling matrix per node batch/prefill |
| `fiber_cache` / `--fiber-cache` | `cloned` | `owned`: retain immutable per-sample concatenations |
| `attention_layout` / `--attention-layout` | `event` | `head`: head-major temporary attention batches |
| `defer_state_release` / `--defer-state-release` | false / 0 | retire old state containers in compact parallel cleanup |
| scale-only `--projection-layout` | `input` | `linear`: output-major physical QKV/output matrix storage |

The C++ fiber factory and configuration helper take `(packing, pooling, cache, layout)`;
defaults are `("exact", "event", "cloned", "event")`. The fiber options affect packed
batch/sequence execution; the independent scalar kernel remains the oracle.
Deferred release requires compact events and Streaming. Trace mode retains its
complete next-state records and uses the original publication path.

## CSR pooling

Event offsets are CSR row pointers, attention-output row indices are columns,
and post-attention source coefficients are values. The matrix has shape
`events × present source rows`; no dense node×batch or event×source matrix is
allocated. Explicit zero coefficients keep present rows and their K/V updates.
Sum/mean/linear/all-softmax use one pooling call. All-softmax normalizes the
complete logical incoming domain once per node batch, including absent slots.
Active-softmax still computes stable softmax separately on each event's active
domain before the single sparse multiplication; absent large logits cannot
cause active-domain underflow. This policy uses native CPU ATen CSR operations.

Exact attention buckets collect their output views into original event order
before pooling. This adds a concatenation but removes repeated tiny Tensor
index/softmax/matmul calls. Performance depends on shape and sparsity; fewer
calls alone do not establish improvement. Aggregate's logical result remains
visible, including source contributions. The independent [packed transport
option](packed-transport.md) can share scaled source storage with attention;
visibility does not require separate allocation or repeated scaling.

## Immutable KV ownership

Both attention paths already construct an independent, exact-length KV
concatenation per sample. `owned` saves prefixes of that allocation directly,
instead of cloning final KV out of temporary attention batches. The final
cache occupies only that sample's actual rows, with no other sample or padding
retained. Earlier prefill event states can share immutable prefixes belonging
to the same sample; there are no in-place writes to old or saved tensors.
Explicit cursor snapshots still clone storage. Training keeps the declared
public VJP and its existing scalar replay; this is not an optimized backward.

`attention_layout="head"` stacks temporary Q/K/V as `[sample, head, row, D]`.
The batch matrix multiply can then flatten sample/head with a view. The original
`[sample, row, head, D]` storage requires materialization when flattening these
dimensions after permutation. Single packing gathers query-owner caches after
stacking in head order, as in LH's batched cache layout. Persistent KV still has
the same `[row, head, D]` logical format and compact final storage. Scores, masks,
softmax, query counts and cache visibility are unchanged.

## Publication and matrix storage

Deferred release swaps the new state into its owner and keeps the displaced
container on the event until existing worker cleanup. Publication order,
message delivery and region decisions stay deterministic. No tensor value is
modified. Trace mode retains the complete recorded next state instead.

The scale layout option changes only QKV/output matrix strides. It preserves
their shapes, values, random-number consumption and single parameter owners;
there is no permanent transposed cache to invalidate or extra parameter copy.
The generic native kernel already accepts these strides. Emit and LM-head
weights retain their existing Linear layout.

Dense work counters remain comparable across all options, but do not count
pooling's sparse arithmetic, indexing or copies. Use the common optional timers
in [operator-profiling](operator-profiling.md) plus unprofiled paired end-to-end
runs. For CSR, `detail/pooling_calls` counts node batches; its timer includes
CSR construction and concatenation. With event pooling it counts sample events.
Head-major exact query stacking is included in `detail/input_pack`; KV stacking
remains in `detail/kv_gather`, with fewer implicit copies inside attention.

Directed checks live in `test_fiber_efficiency.py`, `test_scale_efficiency.py`
and `test_stream_optimizations.py`: independent values/routes/states and VJPs,
ragged prefill, no-grad ownership, snapshot/checkpoint policy switches, extreme
active logits, shared strided parameters and optimizer updates. The
[CPU qualification and ablations](evidence/fiber-efficiency.md) passed6719 tests
and measured12 large cases. The repeated all160 combination improved latency
by about1% in that short window; this is not a general speedup claim. Consult
STATUS for the current qualification and follow-up measurements.
