# Additive and subtractive diagnosis

Use this method when a substantial performance gap is unclear, or a new
high-frequency module/interface needs an attainable baseline. The depth of the
investigation follows its risk and expected use.

Start with a compact implementation of the selected function in a declared
domain. Batch both samples and sequence positions where the mathematics allows
it. Inspect this baseline too: a Python batch loop, repeated history expansion
or quadratic saved workspace can make a supposedly direct implementation slow.
Its measured time is attainable evidence, not proof of optimality.

Add meaningful duties in controlled steps: initial state, shared parameters,
actual backward, optimizer, continuation and public composition. Keep inputs,
weights, precision, return scope and timing boundaries fixed for same-function
comparisons. This distinguishes mathematical work from the cost of expressing
that work through an interface. A useful direct implementation can become the
production path after appropriate independent checks; it need not remain a
permanent duplicate.

Use subtraction to test specific explanations: no-grad versus grad-enabled
forward, forward versus complete training, selected versus padded rows, or one
layout versus another. These differences identify where to look; they do not
alone isolate a single cause. Stage timing, device traces and autograd inspection
are optional instruments chosen for the unresolved question.

Static inspection complements measurements. Follow both implementations' actual
loops over layers, regions, nodes, samples, positions, memory chunks and solver
iterations. Distinguish setup from each invocation and host metadata iteration
from tensor arithmetic. Record where B/L axes remain batched and where an
interface creates copies, gathers or repeated dispatch.

Count meaningful operations with an explicit unit: contractions, gathers, bytes,
autograd nodes or kernel submissions. Use expressions or bounds where runtime
support or iteration count decides the work. A Tensor API call, ATen dispatch
and device kernel are different quantities. Fewer calls can still mean more
padding, memory traffic or arithmetic.

Classify the remaining gap as mathematical work, duplicated/padded work,
interface or autograd overhead, or unresolved cost. If compact computation fits
poorly, redesign the affected boundary around the current instance requirements.
The separately scoped general foundation creates no fallback obligation here.

Finish a bounded investigation with an adopted change, a rejected approach, or
a concrete remaining opportunity and reopening condition. New evidence can
justify another investigation; routine edits do not require a complete census
or a new direct baseline. Comparison definitions and scope are owned by the
[benchmark contract](../../../../docs/benchmark-contract.md).
