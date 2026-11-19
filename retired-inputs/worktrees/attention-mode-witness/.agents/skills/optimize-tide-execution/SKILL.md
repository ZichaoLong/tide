---
name: optimize-tide-execution
description: Diagnose and improve Tide instance tensor flow, autograd, numerical cost and device performance with finite, comparable measurements.
---

# Optimize Tide instances

Optimize the selected bindings in [project scope](../../../docs/project-scope.md).
Use [current lessons](../../../docs/performance-lessons.md) as applicable
mechanisms and observations; task priorities belong in [ROADMAP](../../../docs/ROADMAP.md).

Inspect loops, tensor shapes, copies, lifetime and gradient assembly. Carry B/L
and compatible node axes through module boundaries. Count selected and padded
work separately. Treat historical KV, reception and regional Attention as
different shape families when choosing native fusion, ordinary batch math or
tiling. Preserve original-time visibility, weighted derivatives and private state.

Use a compact direct computation when it clarifies an attainable cost, and
static/profile observations when they distinguish the remaining causes. Choose a
small number of plausible changes, check affected semantics, then measure useful
whole-path cases. Keep, redesign or defer with a reason. A local improvement can
be adopted on relevant evidence; a full combination sweep and a second complete
implementation are not required for every change.

Read only the method needed:

- [Interfaces and autograd](references/interfaces-and-autograd.md): tensor
  boundaries, state ownership, packed derivatives and reusable buffers.
- [Additive/subtractive diagnosis](references/additive-comparisons.md): direct
  baselines, static loop analysis and bounded investigations.
- [Comparisons and attribution](references/comparisons.md): measurement modes,
  actual work, resource observations and the limits of timing conclusions.

[Benchmark contract](../../../docs/benchmark-contract.md) owns the four joint
Transformer matches and total/active/work definitions.
[Precision](../../../docs/precision.md) owns FP32/BF16 numerical policies;
[verification](../../../docs/verification.md) owns correctness acceptance.
Report effective precision/operator choices and stability costs, not just a
payload dtype label. Existing measurement is evidence only at its recorded scope.
