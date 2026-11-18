# Implementation navigation

Active model package: [src/tide](../src/tide). Fixed mathematics: [math](../math).
Current independent checks: [tests](../tests).
Topology generation, batched local math, instance evaluation and benchmarks have
separate responsibilities but no compulsory universal intermediate representation.

| Responsibility | Implementation | Independent checks |
|---|---|---|
| legal configuration, clocks and optional bindings | config.py, topology.py, topology_hb.py; scripts/support_catalog.py | test_topology.py, test_optional_bindings.py |
| exact parameter binding and optional static row views | parameters.py, model.py construction | test_parameter_binding.py, test_parameter_views.py |
| incoming message processing | reception.py | test_reception.py |
| KV history and selected Full | kv.py, layout.py | oracle_fc.py, test_fc_reference.py |
| SSD recurrence/convolution | ssd.py, ssd_memory.py | test_ssd.py, test_ssd_memory.py |
| encoded regional scores, role allocation | region.py, allocation.py | oracle_fc.py, test_allocation.py |
| three weighted Attention sites | attention.py | test_attention.py, device_attention.py |
| layer windows, Timed pending/flush | model.py | oracle_timed.py, test_timed_reference.py |
| training state/checkpoint | checkpoint.py, benchmarks/continuous.py | test_models.py, test_continuation.py |
| HB layer/whole-region ownership, transport and AdamW | placement.py, placement_ownership.py, region_placement.py | test_placement.py, test_region_placement.py |
| optional same-layer device tasks and serial audit identity | region_execution.py | test_region_concurrency.py |
| explicit matrix/accumulation policy | precision.py | test_precision.py |
| fixed-fixture and full-backward continuation diagnosis | scripts/qualify_precision.py, scripts/check_continuation.py, scripts/diagnose_restore.py | test_precision.py, test_restore_diagnosis.py |
| non-equivalent compute comparisons | benchmarks/baselines.py | test_baselines.py |
| four joint matches and actual touched-parameter accounting | benchmarks/matching.py, benchmarks/statistics.py | test_benchmark_contract.py |
| real incremental KV, continuation and fresh-process restored timing | benchmarks/continuous.py, benchmarks/restore.py | test_continuation.py, test_restore_benchmark.py |
| complete resident inference schedules, bounded activity and matched layer placement | benchmarks/inference.py, benchmarks/streaming_audit.py, benchmarks/transformer_placement.py | test_inference_benchmark.py |
| independent KV eviction, original-time and SSD/Conv metadata observer | benchmarks/cache_metadata.py, scripts/check_inference_cache.py | test_inference_cache.py |
| bounded common/budget searches and separate memory lifetime ledger | benchmarks/search.py, benchmarks/variants.py, benchmarks/memory_ledger.py | test_search.py, test_variants.py, test_memory_ledger.py |
| restored-anchor training cost, actual device trace and baseline candidates | benchmarks/costs.py, benchmarks/stages.py, benchmarks/cost_report.py | test_costs.py |
| fixed nonempty training-state path comparisons and first Attention mode witness | scripts/check_training_paths.py, scripts/diagnose_attention_modes.py | test_training_paths.py, test_attention_mode_diagnosis.py |
| named training settings, public-API example and actual resume | configs/recommended, examples/train.py | test_train_example.py |
| optional standalone regional scoring slice | native/regional | test_native_region.py; scripts/check_native_region.py |
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

Optional `region_batch_size > 1` groups same-layer regions with compatible node
roles/memory, edge padding and private-state shapes. `region_batch.py` batches
reception, preparation, selected Full and regional projections, retaining a
separate region axis for Attention and allocation. Publication restores original
node/region order. Shared bases and private adapters retain their original owners;
the ordinary one-region path remains an explicit comparison candidate.

`checkpoint_regions=True` uses non-reentrant activation recomputation of the
pure `_evaluate_region`/batched call. It retains grad/autocast mode and does not
detach cross-window state. This saves internal activations at the cost of repeated
work; regional input messages remain saved inputs. Static last consumers release
unused layer references; Timed pending owns a compact last-tick copy instead of
keeping full-window storage through a view. Autograd may retain other tensors.

`checkpoint_reception=True` recomputes only reception when the surrounding region
is not already checkpointed. Both single and batched regions use the same boundary.
`ssd_summary_backend="prefix"` uses parallel affine composition for short/long
vector summaries, preserving zero decay without logs or inverse decay. The
default block implementation and the separate SSD matrix state scan remain
available. [Training measurements](evidence/training-search-20261118.md) select
these options in named HB presets; ModelConfig defaults stay explicit and general.

`ssd_scan_backend="incremental"` directly evaluates a single-position matrix
recurrence; `"doubling"` additionally replaces the short block's sequential
Cumprod with pairwise products. Both preserve the original block candidate.
Doubling can retain more backward intermediates, so its speed and memory are
separate observations. Reception's fused QKV and structural padding mask are
also explicit candidates; see [Reception execution](reception-execution.md).

Region membership supplies static owner metadata. When every possible owner
uses the same base matrix, NodeLinear flattens node/B/position rows into one
GEMM; private adapters still use node BMMs. Mixed groups retain independent
BMMs. This dispatch does not inspect device owner values on the host.

Optional [static parameter views](parameter-indexing.md) carry a stronger exact
owner order from static region construction. Only consecutive parameter rows
use slices; dynamic Full selection drops that metadata and retains true gathers.
Placed wrappers map the static order to local rows without device-to-host reads.
Caches contain metadata rather than projected weights and are safe to initialize
under inference before switching to training. The default remains `gather`.

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

[HB placement](placement.md) assigns contiguous layers or complete regions to
parameter owners. The input projection has one owner; regional transport deduplicates
predecessor nodes per destination, and private state stays on its regional owner.
`placement_adamw` preserves original Parameter ownership across disconnected
shards; checkpoint format 2 retains partitions, transport and optional execution policy.
`region_concurrency="per_device"` joins one same-layer task per device before
publication, carrying grad/autocast/device/current-stream context. Caller-thread
instrumentation uses an explicitly labelled `serial_region_audit` witness.
Whole-region placement batches compatible regions within one parameter partition;
different partitions remain separate batches even when they use one device.
The caller prepares shard routing before concurrent workers start, and publication
restores the original region order. Batching composes with recomputation and
`per_device` execution; candidate qualification is recorded separately.
Default cross-NPU transport uses host staging; its measured qualification does
not erase the direct transport's recorded backward event failure or establish
peer DMA, replicas or CUDA hardware qualification.

There is no device-resident scheduler. The [TorchAir probe](evidence/roadmap-resident-20261118.md)
stops before backend invocation at a guard over static role metadata. That guard
can be moved outside capture; it does not prove later data-dependent selection
or progression impossible. The lexical host-decision inventory is navigation,
not a measured cost table or a list of executed backend rejections.
