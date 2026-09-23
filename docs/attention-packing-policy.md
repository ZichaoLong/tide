# Configurable same-fiber attention execution

`attention_packing=exact|single` selects an immutable execution policy for the
built-in `lh-fiber-attention-*-repeat-v1` kernels. Default: `exact`. Graph v13,
checkpoint v5 and tide-core-3 semantics are unchanged. The independent Python
scalar/reference and Python exact packed implementations remain validation
anchors; this increment adds the alternative LibTorch implementation.

## Policies

- `exact`: existing `(initial KV rows, total query rows)` buckets, one QK/AV
  group per shape. KV is represented once per owner inside each bucket.
- `single`: flatten all real queries in the node invocation. Pad each owner's
  old+current KV to the participating owners' maximum length, then gather by
  query owner. Score shape `[sum_queries, heads, max_KV, 1]`. No query padding
  or cross-sample scores. This resembles LH CROSSBATCH grouping but retains
  Tide's functional compact state, per-event pooling and public training VJP.

Both project QKV once over all real source rows and output once over all real
events. Each query sees its owner's old keys and all keys in its current event;
future-event keys and padding have negative-infinity bias. Event clocks,
repeated decay subtraction, source-slot order, pooling and final compact KV
ownership are preserved. Neither mode changes node scheduling or selection.
Floating-point tensor equality uses the existing FP64/FP32 tolerance; routes,
identities and gradient presence must match exactly.

`single` can reduce dispatch at the cost of padding and temporary repeated KV.
For M participating owners, Q real queries and maximum KV length L, padded
owner KV scales as M×L×D; each gathered K or V scales as Q×L×D, besides the
Q×H×L scores. Exact buckets reuse each owner's KV across its queries. This
memory difference matters when local caches or per-owner query counts grow;
the number of global tokens alone does not predict it.
It does not imply LH's reserved in-place cache or CSR pooling. No automatic
selection or coarse interval buckets are implemented. Performance conclusions
remain specific to the measured workload and host.

## API and CLI

C++ clients construct `make_fiber_attention_kernel(node, input_slots, "single")`
and install it in `NodeWeights::kernel`. The returned program is immutable and
can be shared; clock adaptation still occurs during model configuration.
Alternatively, `configure_fiber_attention(compiled_graph, model, "single")`
selects all built-in fiber programs before general model configuration. It
rejects invalid modes or already configured fiber programs before mutation,
so it cannot silently overwrite a custom implementation. Other state profiles
are untouched. No process-global policy or thread-local mode is used.

Python native adapter: `Native(..., packed=True, attention_packing="single")`.
Streaming, owned cursors and frontier sequence/batch calls use that program.
Unpacked scalar steps remain the scalar reference; disabling `packed` does
not request the padded batch. Policies can be changed when constructing a new
executor from the same continuation/checkpoint; no checkpoint conversion is
needed. Training retains the existing packed-value/scalar-VJP replay boundary.

Both `tidegraph-scale-bench` and `scripts/benchmark_pdg_scale.py` accept
`--attention-packing exact|single`. The portable `run_pdg.py` exposes the same
option, retains all other current defaults, prints the selection and records
it in run configuration and native per-token event context. LH has no such
option and continues using its existing CROSSBATCH implementation.

## Validation and work accounting

`tests/test_fiber_single.py` covers ragged caches and source/event counts,
serial/parallel streaming/frontier, complete states/traces/routes/pending,
public-root gradients and absent paths, all five pooling policies, selection,
clear, shared optimizer updates, and checkpoint/cursor policy switches.
`tests/test_operator_work.py` checks that one packed single invocation has one
attention group and preserves all useful/projection work while counting padded
scores. Native factory validation and multi-event counts are also exercised.
Counters include all computed padded scores, including masked future scores
inside prefill. `max_state_batch` continues to count participating owners.
The clean [qualification and fixed workload report](evidence/attention-packing-policy.md)
covers 6553 CPU FP64/FP32 tests, relocated kit builds and a same-binary 17.27B
pair. In that single short-window run, `single` was 7.4% slower despite 84% fewer
attention groups. No general speedup or x86_64 verification is claimed.
