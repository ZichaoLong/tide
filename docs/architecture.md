# Implementation navigation

New model package: src/tide. Fixed mathematics: math/. New tests: tests/.
Topology generation, batched local math, instance evaluation and benchmarks have
separate responsibilities but no compulsory universal intermediate representation.
| Responsibility | Implementation | Independent checks |
|---|---|---|
| legal configuration and graph | config.py, topology.py, topology_hb.py | test_topology.py |
| exact parameter binding | parameters.py, model.py construction | test_parameter_binding.py |
| incoming message processing | reception.py | test_reception.py |
| KV history and selected Full | kv.py, layout.py | oracle_fc.py, test_fc_reference.py |
| SSD recurrence/convolution | ssd.py, ssd_memory.py | test_ssd.py, test_ssd_memory.py |
| encoded regional scores, role allocation | region.py, allocation.py | oracle_fc.py, test_allocation.py |
| three weighted Attention sites | attention.py | test_attention.py, device_attention.py |
| layer windows, Timed pending/flush | model.py | oracle_timed.py, test_timed_reference.py |
| training state/checkpoint | checkpoint.py | test_models.py |
| optional contiguous HB layer shards | placement.py | test_placement.py |
| explicit matrix/accumulation policy | precision.py | test_precision.py |
| non-equivalent compute comparisons | benchmarks/baselines.py | test_baselines.py |
| real incremental baseline KV and continuous timing | benchmarks/continuous.py | test_continuation.py |

Implementation paths in the table are relative to src/tide unless qualified;
test paths are relative to tests. A whole B×L window passes through each region:

```
for layer in topology:
    for region in layer:
        messages = gather_tensor_banks(preceding_layers, pending)
        received, content = reception(messages)
        descriptor_inputs, memory = update_all_receiving_candidates(content)
        controls = regional_scores_and_allocation(descriptor_inputs)
        output = full(pack_selected_queries(content, controls), memory)
        publish_tensor_bank(output, controls)
```

There is no B/L event loop in this evaluator. The finite region loop and fixed
Attention/SSD chunks remain actual costs. NodeLinear batches independent node
matrices; private adapters preserve mathematical ownership, including per-head
KV query adapters. Full packs selected queries once and publishes once. This
does not imply zero padding or zero host scheduling; timing/profiling must
identify their cost before claiming efficient execution at every graph size.

Region membership supplies static owner metadata. When every possible owner
uses the same base matrix, NodeLinear flattens node/B/position rows into one
GEMM; private adapters still use node BMMs. Mixed groups retain independent
BMMs. This dispatch does not inspect device owner values on the host.

Quad allocation sorts its clipped-linear response breakpoints, then rebuilds
the selected face from its remaining mass. Exact endpoints preserve zero
senders. Ill-conditioned rows retain the safeguarded bisection fallback;
Shannon uses that iterative solver. Both use the declared solution-mapping
first-order VJP, not differentiation through solver iterations.

KV cache entries already follow original time. A stable prefix-count partition
compacts valid entries without sorting int64 timestamps; invalid padding stays
first, preserving the previous stable ordering and finite-window retention.
