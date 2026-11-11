# Comparisons and cost attribution

Choose comparisons for the question, using the public execution path and fixed
model/input/state/parameter identities where applicable. The current concrete
FC16 entry is [fc16-cost-attribution](../../../../docs/fc16-cost-attribution.md);
the [reproduction map](../../../../docs/reproduction.md) links other diagnostics.

| Comparison | Answers | Evidence boundary |
| --- | --- | --- |
| Hand calculation, independent formula/solver, directional derivative | Is the local forward/VJP correct? | Smooth branches, declared boundary VJP and finite precision are separate checks |
| Independent graph schedule | Are dependencies, controls, messages and state transitions correct? | Reference actions or routes cannot become candidate inputs |
| Same-function direct tensor computation | What does the core mathematics cost? | Declare omitted observations, roots, diagnostics and restore duties before interpreting the gap |
| No-grad, grad-enabled forward, actual backward, complete training | Where do graph construction, saved state, reverse assembly and updates cost? | Grad-enabled forward is not training; mode differences alone do not isolate pure engine overhead |
| Comparable native Transformer | How efficiently do projections, Attention and FFN execute? | Different function/topology; Tide Read/evidence/control costs remain real mathematics |
| LH/core-3 and current candidates | Which caching, batching, explicit VJP and parallel mechanisms transfer? | Match timer scope, modules, active work and CPU pools; parameter count alone is not equivalence |

## Measure structure and time separately

For autograd, inspect both outer roots and private graphs saved inside custom
Functions/Nodes. Deduplicate nodes and edges; count their types, saved tensors,
logical bytes and unique storage bytes. Then separately count reverse-node
executions and nested `grad` calls. Traversing only public `next_functions`
misses private graphs, and repeated execution counts are not unique node counts.

Record unique KV writes versus repeated prefixes, history projections, packing
bytes, output objects, metadata transfers and effective matrix shapes. Stage,
operator, custom Function and hardware-kernel counts are different quantities.
Calling-thread timers contain dispatch, math and waits; overlapping worker sums
and nested operator times cannot simply be added as wall-time shares.

Use separate profiling and uninstrumented measurements. Freeze initial weights,
optimizer, state and inputs; reset them after warmup when comparing one update.
Report actual candidates, Full calls, support and message volume alongside time.
Different routes/work counts preclude a strict same-work speedup claim.

Measure no-grad requests and training separately. Input-position throughput is
processed positions divided by the declared complete interval. Generated-token
throughput needs actual generated tokens. TTFT requires an actual prompt-to-first
returned-token measurement; training time divided by B or L does not provide it.

## Report coverage and incremental cost

When summarizing implementations, show which modes were actually measured at
each source/configuration: no-grad request, grad-enabled forward and complete
training. Leave missing cells explicit; another scale, historical revision or a
training forward phase cannot supply a missing standalone measurement. This is
a reporting discipline, not a requirement to run every mode or combination.

Give elapsed seconds and the throughput numerator. For ragged graph inputs,
count actual distinct sample/logical-position pairs represented by input; B×L
may be only a capacity. Port messages, node updates and Full calls are work
counters, not additional input tokens. Keep TTFT separate from throughput.

For matched complete-update records, subtract corresponding forward/backward
phases and show the remaining whole-interval delta. This localizes a measured
gap without establishing its cause. A one-sided small profile cannot allocate
the full-scale difference to regional allocation, scheduling or autograd.
Map identified costs to module/interface symbols and distinguish required
mathematics, representation choices, repeated implementation work and unresolved
cost. Quantify only the increment actually isolated; otherwise name the next
discriminating comparison. Compact formulas are attainable references, not
proof of a minimum achievable time.

The [FC16/LH accounting example](../../../../docs/evidence/performance-accounting-20261111.md)
derives these distinctions from existing records without launching new runs.

## Resource and scale choices

Use available CPU/memory/idle devices to finish useful work quickly, as specified
by [execution-flows](../../../../docs/execution-flows.md). Record actual ATen,
BLAS, scheduler pools, affinity/NUMA and device mapping. Small-test thread counts
are not performance caps. Shared NPU use follows the current prompt's explicit
scope; historical shared measurements do not grant new access.

Use small/moderate cases for frequent attribution and relevant numerical checks.
Use large representative cases at stage boundaries or where they can change a
decision. Reuse valid prior measurements. A slower local case or finite numerical
difference does not require a broad sweep searching for a favorable result.

Complete training includes required preparation, scheduling, packing, loss,
backward, checks, communication, optimizer, cleanup and synchronization.
Construction, compilation, capture, warmup and disk restore are separately timed
unless the reported interval deliberately includes them. Review every internal
result rather than deriving acceptance from an outer supervisor's exit status.

## Concrete historical anchors

- [LH Add and core-3](../../../../docs/evidence/add-scale-comparison.md): matched
  9.468B scale, CPU56; no-grad and grad-enabled forward, no large backward.
- [LH Attention](../../../../docs/evidence/lh-a10-attention.md): 17.269B, B512,
  CPU160 budget and about53 observed occupied cores; a different parameter count.
- [Full VJP](../../../../docs/evidence/full-batched-autograd.md) and
  [Aggregate VJP](../../../../docs/evidence/aggregate-batched-autograd.md):
  remove per-event replay; the latter also records small true training.
- [FC16 attribution](../../../../docs/evidence/fc16-cost-attribution-20261110.md):
  public graph, independent FC formula and Transformer cost; complete-scale
  numerical qualification and scoped small checks are distinct.

Keep these original scopes and source identities. New stack/source measurements
are new evidence, even when dimensions and benchmark names are unchanged.

Repeatable local entry: [FC comparison commands and record adapters](../../../../docs/fc16-cost-attribution.md#repeating-the-investigation)
provides explicit old/default/formula/different-function pairs without fixed job
names. When moving a numerical boundary, keep profiler ranges aligned with the
same duties (layout/bind, mathematics, publication and transpose); moved work must
not masquerade as reduced math or increased interface overhead. New workload
adapters preserve source/configuration/actual-work identity and failures; they do
not confer independent correctness or complete-root qualification.
