# Persistent Next programs and causal state preparation

[Qualified](evidence/next-programs.md). This instantiates upstream
Next(old, comparison, time, content, active, control) without changing Full's
snapshot or introducing updates for empty nodes.

`NextInput` contains both complete states, logical time, complete Content,
selection flag and control tensor. Python `NextProgram` is a parameter-owning
module supplied by `Model(next_programs={node: program})`; native clients supply
`NextKernel` in `NodeWeights.next_kernel`. Programs cannot mutate inputs/weights
or call/read the current Full. Custom profiles must have a versioned name in
`Node.next_state`. The Python adapter rejects unmatched native overrides.

Every candidate executes Next, including passive candidates. Full executes only
for selected candidates and reads comparison, even if Next has cleared or
replaced persistence. Empty nodes/regions remain unchanged. Native Next calls
run in the node pool with Torch thread-local state; results commit in canonical
order. Streaming may compute each node's Full after its Next inside the same
worker job; all records are delivered after the pool completes. Frontier keeps
causal Next commits between frames, then batches Full across closed frames.

## Profiles and static clear policy

`adopt-v1` returns comparison. `control-blend-v1` computes
`(1-control)*old + control*comparison` for the value and matching named slots,
and retains comparison clocks. This example requires matching slot keys/shapes;
it explicitly rejects growing attention caches rather than inventing a cache
interpolation. Its current selector supplies scalar softmax probabilities.

After the program, graph-owned `Node.clear` applies the state kernel's reset to
selected nodes. This static wrapper is part of the node's Next function and is
not captured in a shared weight module. Existing nodes may share parameters
while using different clear policies. A custom conditional reset can set clear
to false and express its transition using the six inputs.

Nonidentity programs validate resulting value/slot metadata and finiteness,
state-kernel layout and clocks before applying this wrapper. last_time must be
in [-1,current time], observations must be a nonnegative int64. The explicit
comparison-identity capability permits an unchanged-input fast path. State-kernel
reset must preserve its valid state contract, as in the existing clear API.

## Prefill and training

Programs default to `comparison_identity=False`; state prefill requires true,
exact state-sequence support, observe-all and no selected-clear policy. This
capability promises to return comparison with all slots/clocks unchanged.
Control blend and default custom Next therefore prepare state, select and commit
causally. Full can still batch closed frames. `next_steps` counts candidate
transitions; `state_prefill_blocked_next` counts events blocked by this capability
when the other prefill conditions hold.

By default Next uses independent scalar event graphs, with native node parallelism.
Optional native Streaming [batch Next](packed-transport.md) batches adoption/reset
and binds training results to scalar semantic replay.
Batch/sequence packing of Aggregate, Upd, Read and Full remains available under
its contracts. The default path has no joint Next batch. This preserves isolated
public-root first-order VJPs without additional Next replay; packed backward
optimization and performance measurement remain separate gates.

Current graph/checkpoint versions are in `semantics.md`. Profile changes
reject incompatible graph identity before changing weights. Shared custom Next
parameters use normal optimizer/checkpoint alias tracking. Representative checks
are in `test_next_schedules.py`, `test_next_contract.py` and the standalone
`cpp/test/next_programs.cpp`. Region integer/tensor history and structured controls
are in `region-programs.md`; LH profiles and comparisons are in `lh-compatibility.md`.
