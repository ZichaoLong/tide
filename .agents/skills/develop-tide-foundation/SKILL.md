---
name: develop-tide-foundation
description: Develop or verify Tide foundation modules, graph execution, training and continuation; connect fixed mathematics, implementation, directed tests and delivery evidence in this repository.
---

# Develop Tide Foundation

Use this workflow for a concrete implementation, correctness fix or downstream
foundation request. Start with [AGENTS](../../../AGENTS.md) and current
[STATUS](../../../docs/STATUS.md); the backlog remains
[ROADMAP](../../../docs/ROADMAP.md). Skills do not create a second task ledger.

## Locate the behavior

Use [semantic-map](../../../docs/semantic-map.md) to identify the fixed definition,
domain, parameters, state snapshot, dependencies and returned observations.
Use [architecture](../../../docs/architecture.md) and `rg` for implementations,
callers, configuration checks and independent tests. Read only affected contracts.

Distinguish mathematical behavior from API shape and implementation detail.
Interfaces serve performance, generality and replaceable local modules. API,
layout and serialization changes may need a version and migration; expensive
historical compatibility need not remain in the default path. Preserve required
observations and gradient dependencies through a documented binding, not through
accidental old object layouts. [semantics](../../../docs/semantics.md) owns graph
and checkpoint versions.

For a downstream request, follow [downstream-contract](../../../docs/downstream-contract.md):
inspect its fixed dependency, actual configuration, reproducer and acceptance
conditions. Foundation delivery and consumer acceptance remain separate facts.

## Implement at the affected boundary

- Keep graph representation, scheduling, local mathematics, placement, state
  ownership and comparison separate. Retain independent Python scheduling and
  standalone C++ execution; a wrapper is not an independent correctness anchor.
- Check the actual capability and configuration dispatch across affected clients.
  Same-shaped tensors need not have the same source, parameter owner or gradient
  root. Metadata such as logical time and counts remains int64.
- When changing backward or shared computation, follow all consumers of the
  state/result: unused versus connected-zero gradients, repeated inputs, parameter
  aliases, original version guards, optimizer updates and retained continuation.
- For batching or performance work, use
  [optimize-tide-execution](../optimize-tide-execution/SKILL.md).
  For broad interface or dependency changes, use
  [maintain-tide-foundation](../maintain-tide-foundation/SKILL.md).

## Select evidence by the change

[verification](../../../docs/verification.md) defines directed development checks
and immutable qualification; [development](../../../docs/development.md) gives
the build and installed-consumer commands. Select affected checks, rather than
rerunning every backend, configuration or historical experiment.

Use literal formulas or an independent solver for local mathematics, directional
derivatives for smooth branches, and independent schedules for graph dependencies.
Declared hard-function VJPs need their own checks; do not gradcheck a surrogate
against a different hard forward. Keep independent roots and shared contributions
in targeted tests, then check complete observables, multiple optimizer steps and
new-owner/process restore where affected.

Finite discrepancies receive bounded first-difference, higher-precision or
connectivity diagnosis. Preserve the original failure and report unclassified
causes; continue independent work and declared numerical exploration. Do not
convert causal errors, non-finite results or failed exact restore into tolerances.
Device build, device execution, training and scale evidence are distinct.

## Deliver a reviewable change

Run the relevant short development checks, commit the implementation, then test
that immutable source for durable qualification and record evidence separately.
Use existing launchers and [execution-flows](../../../docs/execution-flows.md)
for resources and long jobs; stable sources allow independent development to
continue while jobs run.

Use the [change-maintenance map](../../../docs/core5-study.md#修改时同步维护什么)
and [reproduction map](../../../docs/reproduction.md#一次调查结束后维护什么).
Update the affected contract, source/test navigation and reviewed evidence;
record reusable positive and negative lessons in the
[performance index](../../../docs/performance-lessons.md). Keep raw builds,
logs and checkpoints in ignored output directories. Report implemented,
verified and outstanding scope separately.
