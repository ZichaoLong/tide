---
name: maintain-tide-foundation
description: Review and improve Tide architecture, interfaces, source navigation, maintenance mappings and consumer usability, including systematic file/function audits and deliberate API migrations.
---

# Maintain Tide Foundation

Use for architecture/interface changes, broad source review or developer/consumer
usability work. Start from the [branch purpose](../../../README.md),
[architecture](../../../docs/architecture.md) and
[public consumer contract](../../../docs/downstream-contract.md).
Current state and backlog remain STATUS and ROADMAP.

## Design for the foundation's purpose

Interfaces serve general graph execution, trainable local modules, online batching,
continuation and independent correctness evidence. Performance, clear ownership
and module replacement are design requirements. Historical internal or public
API compatibility is a migration concern; it must not dictate a fragmented
default computation or accumulate wrappers in the core.

Separate fixed mathematics and required observations from the software form that
exposes them. Batch layouts, result mutability, handles, parameter containers and
checkpoint schemas can change deliberately. Document version/identity changes,
update consumers/examples and test the new contract. Keep old experiment locks
and historical evidence intact; do not silently upgrade active installations.

## Audit both modules and their connections

For a broad audit, inventory relevant files and functions at an exact source
revision. Record which symbols were actually reviewed, observations, proposed
changes, related checks and unresolved findings. An automatically collected symbol
list is an inventory, not review coverage. Keep one review artifact as evidence;
link actionable remaining work to ROADMAP instead of creating a competing backlog.

Generate the inventory with [review_inventory.py](../../../scripts/review_inventory.py)
using `--revision FIXED_SHA --output artifacts/NEW_AUDIT.json`. It reads committed
Git bytes, not working-tree edits. Python symbols use AST; C++ symbol inspection
requires `--cpp clang` with the matching include/compile arguments. Skipped or
failed parsing stays pending. `--reviews MANIFEST` attaches actual, source-hash-bound
review claims; see the [review report](../../../docs/evidence/implementation-audit-20261110.md)
and [manifest](../../../docs/evidence/implementation-audit-20261110-reviews.json)
for the distinction between inventory, reviewed code and qualification.

Inspect responsibilities, dependencies, duplicate implementations and public
entry points. Follow calls across graph representation, scheduling, local
mathematics, placement, autograd, parameter ownership and persistence. A clean
individual function can still participate in repeated cross-layer packing.

Ask of each affected boundary:

- What mathematical inputs, outputs and roots does it own? Which are metadata?
- What batch/ragged layout and state lifetime does it accept and produce?
- Can a replacement local implementation use it without duplicating scheduling,
  configuration resolution, validation or result assembly?
- Which public/default callers exercise it, and which independent implementation
  can detect a shared error?
- What migration or removed option would make the path simpler?

Refactor by responsibility and dependency. File-size guidance in AGENTS prompts
inspection; arbitrary splitting, forwarding-only layers and a registry for every
historical exception do not establish a better architecture. Preserve independent
reference/specialization schedules while sharing non-oracle infrastructure where
that does not couple their evidence.

## Make maintenance discoverable

Use [semantic-map](../../../docs/semantic-map.md) for definitions and independent
tests; [architecture](../../../docs/architecture.md) for implementation ownership;
[core5-study](../../../docs/core5-study.md#修改时同步维护什么) for change impact;
[reproduction](../../../docs/reproduction.md) for failures and generators.
Update these existing owners when moving symbols or changing a public contract.

Keep model, workload, execution and measurement identities distinct. Requested
and effective configuration must be inspectable; a capability declaration is not
a qualification result. Readable API/CLI defaults should select suitable behavior
by capabilities, with explicit failures for unsupported requests.

Experience is maintained by mechanism in the
[performance index](../../../docs/performance-lessons.md), linked to fixed
evidence and checks. Only project-independent, demonstrated methods should be
considered for later account-level migration; ordinary repository work does not
require editing account skills.

## Validate navigation and usability

Use concrete developer journeys: from a clean checkout install a public consumer;
locate and replace a local candidate; diagnose a slow path; reproduce a recorded
failure; save and restore a configured run. Check changed links and actual entry
points, not only directory names or prose.

A fresh-context review can check whether the repository alone supplies the needed
definitions, implementation, tests, commands and update obligations. Match the
check to the change: documentation reorganization does not require hardware or a
full numerical matrix. Substantive API and runtime changes use
[develop-tide-foundation](../develop-tide-foundation/SKILL.md) and affected
installed-client checks. Report remaining confusing or unreachable paths as
findings rather than declaring governance complete from link validation alone.
