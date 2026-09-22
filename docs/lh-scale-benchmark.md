# Large LH/Tide streaming comparison contract

Status: eight original-LH local scale runs are complete; see
[source, measurements and limits](evidence/lh-local-scale-pilot.md).
Historical data are references, not pass/fail targets. Exact historical
reproduction, large Tide/LH parity and a LH/Tide speed ratio remain unverified. Implementation backlog
belongs to [ROADMAP](ROADMAP.md); this file owns workload and comparison meaning.

The user subsequently identified a10fdb1 plus a few parameter changes as the
intended Attention baseline. The active follow-up uses that revision's original
C++ algorithms/test timer and explicit width/selector changes, with an expanded
160-core local budget; see [original test reproduction](lh-original-test.md).
The earlier Add pilot does not reproduce that Attention workload. The present
LH workspace has a modified BatchHidden.cpp, so this follow-up uses a separate
clone of the selected revision instead of the earlier dirty-source snapshot.

## User-reported historical references

Reported on 2026-09-22, for a 56-core CPU and batch 512:

| Reference | Approximate parameters | Hidden width | Nominal activation ratio | Reported static nodes | grad ms/token | nograd ms/token |
| --- | --- | --- | --- | --- | --- | --- |
| `lh-wide-2048` | 8.8 billion | 2048 | 1/32 | 224 | 3.6 | 3.3 |
| `lh-narrow-128` | 8.5 billion | 128 | 1/64 | 57,344 | 136 | 29 |

These numbers are user observations, not new project measurements. The node
counts may exclude hubs and describe one static graph reused by inet and onet.
The user confirmed that ms/token divides elapsed time by both batch and
measured generation steps. Thus the reported nograd throughputs are about
303.0 and 34.5 sample-tokens/second, and batch-512 step times are about 1.6896
and 14.848 seconds, respectively. These are arithmetic restatements of the
historical observations, not new measurements or per-sequence token latencies.

The CPU was an Intel model approximately eight years old; the exact model is
unknown. Historical dtype is remembered as LH's default, plausibly FP32 but
unconfirmed. Use explicit FP32 for the initial reconstruction and retain this
uncertainty. The user does not recall whether grad meant grad-enabled forward
or backward alone. Preserve that column as an unresolved stage; do not label
it end-to-end training or compare it to a newly chosen stage as if confirmed.
Topology, weights, selector budget, token count, cache age and reset policy also
remain unresolved. The user did not observe a substantial next-token slowdown
within a few hundred tokens; record that as a bounded historical observation,
not constant-time attention or an unbounded-context claim.

The reported node-count products (leaf counts in the candidate reconstructions)
are exactly equal:
`224 * 2048^2 = 57344 * 128^2 = 939524096`. This makes the pair useful for
studying node granularity near a fixed parameter budget. It does not establish
equal total parameters, degree, candidate work or selected work; the activation
ratios differ by two. Historical reproduction retains both ratios. A later
controlled granularity comparison needs a common ratio and actual work counts.
Include width 2048 in local exploration; reported parameter counts, graph sizes
and timings are reference scales, not exact acceptance thresholds. The user
explicitly authorized local experiments and requested measured results.

## Source clues and unresolved reconstruction

Read-only inspection of `~/llm/lh` on 2026-09-22 found:

- `test_Connectome.Graph.py` contains the active generation example
  `base_num_or_hpnums=[0,1,7,224], levelnum=2, localnum=32`.
  `PyConnectome/Graph.py` defines the entries as per-level counts, not cumulative
  offsets. This candidate has 224 leaves and 8 hubs: 232 static nodes.
- The same file contains two commented 57,344-leaf examples:
  `[0,1,7,56,448,57344]`, localnum 128, has 57,856 total nodes;
  `[0,1,7,56,448,3584,57344]`, localnum 16, has 61,440 total nodes.
  Neither is identified as the historical narrow run. A leaf count alone does
  not select a topology or establish nominal activation 1/64.
- `Connectome/cpp/test/test-cortexnet.cpp` uses batch 512, CROSSBATCH attention,
  `selectnum=1`, and 100 calls to `IOCortexNet::think`. The non-NOGRAD branch
  does not call backward or an optimizer. This is a clue for grad-forward only,
  not confirmation of the historical experiment's scope.
- The current `Connectome/cpp/test/cfg.json` uses width 512 and four heads;
  it is not either historical model configuration. The referenced large
  `Connectome/cpp/test/graph-data/cfg.json` is unavailable at that path.
- `bench-lh-small.cpp` is not an exact replacement timer: its `decode` includes
  prompt execution and its incremental timer keeps advancing mutable caches.
  `NoGradGuard` alone also does not specify LH's own hidden-state mode flags.
  A dedicated explicit mode/reset/timing wrapper is needed.

Keep the LH tree read-only. Snapshot its actual C++ bytes, dirty-source identity,
graph-generation Python/dependencies, model configuration, generated adjacency
and inputs into project-owned artifacts before qualification. Python is used
only for graph generation here; the LH execution reference is its original C++.
The dedicated preparation tool now snapshots graph-generation sources separately
from the existing C++ snapshot, with identities in each input record.

## Node and parameter accounting

Record leaf, hub, total static, inet, onet, PDG body and readout node counts
separately, with all four adjacency block edge counts and degree distributions.
For the current equal-width mapping, two cortex state namespaces require
`2 * static_nodes` body nodes and one readout node. Thus the 232-node candidate
would have 465 single-PDG nodes under that mapping. The two narrow candidates
would have 115,713 or 122,881. These are conditional construction counts, not
measurements of a built large model.

The [single-PDG encoding](lh-single-graph.md) replicates body edges by token
phase while aliasing their original parameters. Record this physical edge cost
and logical-to-physical identity mapping. Never multiply parameter count by
phase aliases or identify inet and onet merely because they share a topology.
Count distinct parameter owners, actual tensor storage and adapter-only
scaffolding separately, including embedding, four signaling blocks, memory,
normalization, Pronounce and vocabulary head. Report actual totals alongside
the approximate 8.8B/8.5B labels. Memory includes state, cache, messages, maps,
autograd metadata and temporary packing, not just weight bytes.

Record realized candidate and Full counts per sample/tick and per output token,
for each cortex and layer. Separate the nominal region budget from global
selected-node fraction: mandatory hubs and lead-point policy affect it. Also
record active edge/message counts, per-node packed batch distributions, cache
lengths, body ticks per token and load imbalance. Global batch 512 does not
guarantee 512 rows per node.

## Comparison scopes and modes

1. **Historical reconstruction:** recover the original configuration and timer
   where possible. Preserve unresolved fields explicitly. A new machine run
   is a reconstruction, not a promise to repeat old milliseconds.
2. **Matched inference:** original LH C++, Tide two-clock adapter and Tide
   single-PDG cursor use identical logical topology, weights, token IDs, module
   equations, selectors, clocks, cache/clear rules and readout. Compare native
   serial, node-parallel and packed paths with correctness outside timing.
3. **Comparable-scale Tide:** other PDG topologies/modules may match parameter
   budget, width, batch and realized work. Label them scale comparisons; they
   do not certify LH numerical equivalence or isolate the executor alone.

Keep nograd-forward, grad-forward, backward and optimizer timing distinct.
For grad-forward, require grad-enabled parameters, declare graph retention and
detach boundaries, and verify that the path actually builds an autograd graph.
Set both ATen grad mode and LH's own hidden-state flags before state creation.
Do not substitute inference_mode for no_grad without declaring another variant.
An enabled-grad forward comparison measures different AD implementations;
LH remains no authority for Tide's training semantics. Tide packed semantic
replay counters and costs must be visible. Tide backward/optimizer comparisons
use Tide's own independent semantic anchors.

## Timing, resources and records

For B samples and T measured token steps with total elapsed milliseconds M,
retain raw step times and report both `M/T` ms per batch step and `M/(B*T)`
amortized ms per sample-token, plus `1000*B*T/M` aggregate tokens/second. The
amortized number is not the latency observed by one autoregressive sequence.
Specify whether the timed region includes embedding, all body ticks, readout,
vocabulary head, sampling and state disposal. Measure construction, warm state
preparation, core advance and snapshots separately. If cache ages change with
steps, compare identical ages and report the time series; do not treat growing
caches as repetitions of the same fixed workload.

Use a common explicitly selected physical-core budget for same-host LH/Tide
runs, record CPU model, affinity and NUMA placement, and run heavy variants
sequentially. The earlier Add pilot used56 cores; the user authorizes up to160
for the a10fdb1 Attention follow-up. Record node-worker, ATen, inter-op, BLAS
and OpenMP settings; worker count times an independent BLAS pool is not the
physical-core budget. Budget roughly half this host's physical memory for
this workload; the former256-GiB bound is superseded. Record an address-space
limit separately from actual RSS. Serial is a correctness/overhead anchor;
parallel implementations may divide the common core budget differently.
Matching a core count does not reproduce another CPU's bandwidth or caches.

CPU FP32/FP64 remain supported project targets. Resolve the historical dtype;
label any chosen reconstruction dtype explicitly. FP32 parameter storage alone
is about 35.2/34.0 GB for the reported sizes, before extra copies and caches.
Inspect available RAM, cgroup limits, disk and cache-growth estimates before
allocation. Avoid serializing several full 8B weight copies just to transfer
between implementations; preserve reproducible input/weight identity and
measure one implementation's resident state at a time.

Use frozen sources, bounded durable jobs and the existing project-owned run
record/metrics protocol described in [streaming-benchmark](streaming-benchmark.md).
Each run needs finite token/repetition, wall-clock, memory and artifact bounds.
Retain failures and limits reached; partial runs do not qualify the full target.
Declare warmup and sampling only after a short staged pilot estimates cost.
No unrestricted parameter sweep is implied by these two fixed target cases.

## Acceptance boundary

The current whole-model oracle uses a fixed width-4, batch-4 fixture and
hand-built adjacency (`cpp/test/lh_iocortex_fixture.*`). Its constructor also
rewrites weights for deterministic small checks and clones imports. It cannot
serve as a large-model importer merely by changing constants. A reusable
four-block CSR/CSC importer must preserve original weights, arbitrary region
budgets, aliasing and source-port identity, with bounded memory overhead.

A completed large matched comparison requires that the reusable importer passes
small independent parity gates, topology/parameter accounting is checked, a
bounded scale ramp succeeds, and the chosen scale results have complete timing,
work and memory evidence. Check large output/state/route agreement outside the
timer without materializing an unbounded whole-run trace. Never relax routing
identity requirements or fabricate equivalence when different kernels choose
different routes. Unsupported cells and failed scales stay explicit.

The small shared-weight EMA pilot remains useful for runtime overhead; it
provides no qualification for these independent-weight 8B workloads. The
observed historical gap motivates profiling call granularity, CSR-row Emit,
indexing, memory traffic, autograd objects and packing, without assigning the
entire gap to hidden width before matched experiments.

## Local execution entry points

`scripts/prepare_lh_benchmark.py` snapshots Graph.py/BaseUtils.py, uses original
Python graph generation only, and retains historical Add-model configuration
bytes recovered from LH history. Width2048 and width128 Add configurations
exist in that history; they remain candidate reconstructions. The narrow
localnum128 recipe uses selectnum2 for nominal 1/64; the wide localnum32 recipe
uses selectnum1 for nominal 1/32. Actual selected fractions include hubs.

`scripts/build_lh_benchmark.py` builds `cpp/lh_bench/` against the unchanged
original C++ snapshot with OpenMP node parallelism. `scripts/benchmark_lh.py`
records one explicitly bounded nograd, grad-forward or backward run. Grad
forward retains the graph across the declared finite window; backward, when
requested, differentiates a summed squared-logit mean for that window. This is
a runtime probe, not a language-model training objective. Each repetition
resets state and reuses the same pre-generated token IDs. Warmup advances the
state; measurements retain token/cache age in each event. Finite-value checks,
logging and graph disposal are outside forward timing. Large matched Tide
imports and performance qualification remain separate work.

The local Torch CPU build links both OpenMP and a pthread OpenBLAS pool. Record
`--threads` and `--blas-threads` separately; pin the combined process to the
declared physical CPU set. The default BLAS count matches --threads for
reproducibility of the initial pilot. A bounded follow-up compares BLAS=1
with the initial BLAS=56 observation and uses common measured token ages
for nograd and grad-forward. This follow-up is complete in the linked evidence;
thread-count changes are explicit variants.
