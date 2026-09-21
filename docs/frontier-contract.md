# TimedDAG frontier contract v1

The input window is sealed and finite. The node graph must be acyclic; the region
quotient may have cycles. Propagate potential arrival times in node topological
order, preserving every possible path/edge. These domains certify the source
events that must finish before each target fiber is complete. Actual candidates
still require actual messages; a potential event never creates a zero message.

At each stage, take the longest prefix of each (sample, region) time stream whose
potential message predecessors finished in earlier stages. Group these prefixes
by region across samples. The block computes causal Upd/selection/Next, then
packs Full over selected (sample,time) events per node. Complete blocks publish
messages or certify their absence before the next stage.

This is maximal **relative to the declared potential-source completion
certificate and region-block contract**. It is not a claim of globally minimal
kernel count, or maximality over every more permissive P/S/U/F contract in the
upstream notes. Inactive-source absence and separately closed node fibers could
enable finer scheduling; those are later contract extensions.

For observe-all nodes without selected clear, the EMA state recurrence has an
exact affine-prefix contract. The implementation uses a logarithmic-depth
associative tensor scan, followed by packed Full. Selected-only adoption and
clear use causal state steps inside the region block; Full is still batched.
State block/step counts and Full block counts are reported separately. A scan
can change floating reduction order and remains subject to route/tensor tests.

Regular rank-aligned graphs obtain one region block per available region stage,
including a batch/sequence Full call per active node. Unaligned delays or region
cycles can split these blocks. Planning enumerates potential events; an explicit
limit fails rather than silently changing algorithm. Use sparse streaming for
huge graphs/windows where speculative domains would be expensive.
