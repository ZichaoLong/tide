---
name: optimize-tide-execution
description: Review and optimize Tide scheduling, local modules, batching, caches, autograd and device execution using repository performance lessons, independent comparisons and scoped regression evidence.
---

# Optimize Tide Execution

Optimize the whole data path: graph readiness, local computation, state,
backward, validation, transfers, publication and optimizer. The shared contract
is [execution-flows](../../../docs/execution-flows.md); task state is in
[STATUS](../../../docs/STATUS.md), not in this skill.

## Apply experience at the right granularity

Start from the [mechanism-based lessons](../../../docs/performance-lessons.md).
Use established local techniques when their preconditions hold: remove repeated
work, preserve batches across layers, reuse bounded storage and aggregate
adjoints by block. Relevant semantic and module checks justify adopting such
changes. Do not require a global benchmark for every helper or enumerate the
configuration Cartesian product. End-to-end measurements detect integration
problems and establish the scope of performance claims.

Before improving a function, inspect the producer/consumer interface. A batch
wrapper around row-wise work, repeated prefix materialization or an obligatory
mutable output per field may need interface redesign. Mathematics and required
capabilities remain; API form and storage contracts can migrate. Keep costly
historical adapters outside the default path or replace them with migration.

## Find the source of work

Frame a finite engineering question before profiling: the affected capability,
representative cases, resource/time budget and decision to make. Use additive
baselines, subtractive profiling and established local techniques as appropriate;
there is no prescribed ratio or universal ordering. Read
[additive comparisons](references/additive-comparisons.md) when a large unexplained
gap persists or a new numerical/interface module is being designed. Repeated
small gains without explaining that gap are a reason to reconsider the algorithm
or interface, not to extend the same optimization loop indefinitely.

Read [comparisons and attribution](references/comparisons.md) when planning a
diagnostic or performance claim. Reuse existing observations with their original
identity; generate new evidence only for the current question.
Authorized shared-device timing is valid evidence when its contention and scope
are recorded; follow the resource guidance in that reference rather than waiting
for exclusivity or repeatedly requesting already granted conversation-wide access.

Read [interfaces, state and autograd](references/interfaces-and-autograd.md)
when changing batching, caches or backward. A low outer call count does not
show that the private graph or hardware submissions are small. Inspect actual
matrix shapes, graph objects, bytes and ownership before choosing a mechanism.

Prioritize ordinary batch ATen/LibTorch and backend operators. Add fusion or
special kernels when they address a demonstrated capability or cost. Device,
schedule, placement, dtype and operator implementation are separate axes.
Defaults dispatch by graph/module capabilities, not benchmark names or a
hard-coded favorable shape. Genuine feedback may constrain time batching.

## Implement and verify

Select a coherent boundary and its callers; define the observations and roots
it must preserve. Use [development checks](../../../docs/verification.md),
independent formulas and relevant training/restore tests. Record finite numerical
differences after bounded diagnosis without blocking unrelated improvements.

Use [representative regressions](../../../docs/prefill-regression.md) to choose
small frequent checks and larger stage-boundary measurements. FC16, wide FC,
many-node HB/LH, sum/qry and recurrent chains answer different questions;
1B/9B and multi-device/resident runs are selected by affected work and resources.
No single case certifies every composition. Avoid maintaining a separate benchmark
algorithm that bypasses the public path without explicitly declaring that role.

## Retain reusable results

End each candidate with adopt, reject, or defer plus the evidence that would
justify reopening it. Audit coverage and implementation scope are separate:
finding an opportunity does not make it mandatory in the current round. Budget
diagnostics as well as measurements; preserve unresolved costs and strict failures.

Keep new lessons at the narrowest demonstrated scope: observation, mechanism,
change, positive/negative result, preconditions, counterexample and source/test
link. A smaller graph, fewer copies or calls can justify a local change; report
wall-time gains only where measured. Keep required mathematical work, removable
duplication, amortizable execution overhead and unresolved costs distinct.

Update the [lesson index](../../../docs/performance-lessons.md) and affected
contract, implementation and test navigation. A new default also needs public
client, identity and continuation checks. Current unresolved work stays in
ROADMAP, and dated evidence keeps its original implementation identity.
