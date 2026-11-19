# Implementation navigation

Active model package: [src/tide](../src/tide). Fixed mathematics: [math](../math).
Current independent checks: [tests](../tests).
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
| training state/checkpoint | checkpoint.py, benchmarks/continuous.py | test_models.py, test_continuation.py |
| optional contiguous HB layer shards and AdamW ownership | placement.py | test_placement.py, test_capacity_placement.py |
| explicit matrix/accumulation policy | precision.py | test_precision.py |
| fixed-fixture and full-backward continuation diagnosis | scripts/qualify_precision.py, scripts/check_continuation.py, scripts/diagnose_restore.py | test_precision.py, test_restore_diagnosis.py |
| non-equivalent compute comparisons | benchmarks/baselines.py | test_baselines.py |
| four joint matches and actual touched-parameter accounting | benchmarks/matching.py, benchmarks/statistics.py | test_benchmark_contract.py |
| real incremental baseline KV and continuous timing | benchmarks/continuous.py | test_continuation.py |
| three Attention performance sites | benchmarks/attention_sites.py | test_attention_sites.py |
| frozen finite suites and process ownership | benchmarks/programme.py, scripts/launch.py, scripts/job.py | test_programme.py, test_job.py |
| bounded whole-model capture probe | scripts/probe_resident.py | test_resident_probe.py |

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

KV cache entries already follow original time. On NPU a stable prefix-count
partition compacts valid entries without sorting int64 timestamps; invalid
padding stays first, preserving stable ordering and finite-window retention.
CPU retains its measured stable timestamp-sort path. NPU KV queries with at least
four heads batch compatible projections while preserving the private adapter of
each head. CPU and placed wrappers retain the original projection path.

The [precision contract](precision.md) keeps FP32 parameters and inputs while
selecting FP32 or BF16 matrix arithmetic; KV payloads, scalar controls and SSD
state have explicit policies. The [numerical report](evidence/roadmap-numerics-followup-20261118.md)
records actual execution and retained deviations separately from implementation.
The [comparison contract](benchmark-contract.md) defines A/B/C/D targets, while
[benchmarking](benchmarking.md) owns timing and workload interpretation.

[HB placement](placement.md) is a single-process parameter-row/layer split.
`placement_adamw` preserves original Parameter ownership across disconnected
shards, and checkpoint format 2 retains partitions and transport choice.
Default cross-NPU transport uses host staging; its measured qualification does
not erase the direct transport's recorded backward event failure or establish
peer DMA, replicas or CUDA hardware qualification.

There is no device-resident scheduler. The [TorchAir probe](evidence/roadmap-resident-20261118.md)
stops before backend invocation at a guard over static role metadata. That guard
can be moved outside capture; it does not prove later data-dependent selection
or progression impossible. The lexical host-decision inventory is navigation,
not a measured cost table or a list of executed backend rejections.
