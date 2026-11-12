# Additive and subtractive comparisons

For a new high-frequency numerical, state, autograd or interface feature,
consider three levels together: compact mathematical computation, its public
batch-module boundary, and a small real producer/consumer composition. Thin
wrappers can reuse existing references. Depth follows risk and expected use;
do not require a second implementation for every helper.

Start an additive investigation with a compact executable in a checked domain.
It is a measured attainable reference, not proof of optimality or zero redundant
work. Inspect its own algorithm (history expansion and quadratic sequence
workspace are common traps). Preserve independent formulas/schedules when
promoting code into production; do not turn the reference into a second framework.

Add responsibilities deliberately: nonzero left state, sharing, real backward,
optimizer, continuation/checkpoint, public module boundaries, composition and
default scheduling. Hold mathematics, inputs, parameters, return requirements,
precision and timing boundaries constant where possible. Measure the increment
at each meaningful boundary. Separate a new mathematical obligation from extra
work implementing the same obligation through an interface.

Ordinary output training and complete records/intermediate gradient roots have
different duties. Label projected equivalence, complete-record equivalence and
different-function comparisons. Transformer/Mamba comparisons illuminate shared
operations but cannot certify the whole Tide function. An eliminated dependency
with numeric derivative zero is not automatically equivalent to connected zero.

Specify both batch and sequence axes, causal/ragged dependencies, ownership and
materialization, block adjoints and validation granularity in module interfaces.
A compact formula that cannot fit naturally is evidence for redesign, not a
reason to silently bypass general graph semantics or hard-code a benchmark shape.
Capability dispatch must retain legal PDG/TimedDAG/SettleGraph fallback and expose
the effective path. Keep expensive compatibility outside the default computation.

Combine incremental measurements with subtraction: no-grad versus graph-building
forward versus true backward/full training, operator submissions, autograd nodes,
bytes, synchronization and actual work. Use these to choose the next experiment,
not an endless mandatory profile matrix. A baseline can be useful even when it
is not promoted; record why, the remaining gap and a concrete reopening condition.

When the gap is unclear, inspect both implementations statically before choosing
another optimization. Describe their input/output and observation contracts,
then expand the relevant call paths: loops over regions, nodes, physical atoms,
samples, positions, fields, parameter owners and solver iterations. Separate
construction from each invocation, metadata iteration from tensor work, and
forward from explicit/nested backward. Record where the batch axes survive and
where an interface forces gather, copies, materialization or separate calls.

Count meaningful operations with a declared vocabulary (matrix contractions,
normalization, source transforms, solver calls, gathers or bytes). Include loops
and branches: exact counts where static, expressions or bounds where actual
support, ragged lengths or convergence decide them. A Tensor API call, an ATen
dispatch, an autograd node and a device kernel are different units. Do not invent
one exact kernel total from source syntax, or rank algorithms by call count alone;
padding, redundant arithmetic, memory traffic, concurrency and synchronization
can outweigh fewer calls. Use measurements to test the static explanation.

Classify the resulting differences as required mathematical work, required public
observations, avoidable recomputation, or an explicit layout/algorithm tradeoff.
Show the smallest alternative boundary that would carry the compact computation
with the needed semantics. Reuse the analysis for later module replacements;
do not turn it into a mandatory full-repository census for each small edit.
