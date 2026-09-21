# Next program gate

Complete content and independent Read are qualified in their corresponding
contracts/evidence. This remaining plan refines ROADMAP; STATUS is the only
current handoff. Replace this plan with the implemented contract after validation.

## Interface and static policy

Expose NextInput(old, comparison, time, content, active, control) to registered
Python NextProgram and native NextKernel. Content is complete and state includes
all slots/clocks. The function cannot call/read this event's Full, and cannot
mutate states or weights. Only candidates run Next, including passive candidates;
absent nodes preserve state. Full reads comparison regardless of Next's result.

Use Node.next_state as the versioned profile selector. The default adopt-v1
returns comparison. Preserve Node.clear as graph-owned static policy: apply
selected reset in the Next evaluator, after the program. Do not capture clear
inside a shared parameter module: existing tests share NodeWeights across nodes
with different clear flags. A custom reset policy can set clear=False and handle
its own transition using the six inputs. Keep Python/native schedules independent.

Control-blend-v1 example: next=(1-control)*old+control*comparison on value and
matching named slots; keep comparison clocks. Reject mismatched slot shapes (in
particular this is not an attention-cache blend). Validate custom resulting state
metadata, finite values, state-kernel layout and clocks compatible with the next
cut. Known comparison-identity programs may use a validated-input fast path.

## Prefill and execution

Next defaults to no comparison-identity contract. State prefill requires an
explicit identity guarantee, exact state sequence support, observe-all and no
selected clear. Control-sensitive Next uses causal preparation/selection/Next;
Full can still batch closed frames. Count causal Next work and capability fallbacks.
Native independent node Next calls can run in the pool; commit canonically. All
selected Full requests retain pre-Next comparison snapshots. Preserve the existing
public-root first-order VJP and None/zero contract, including shared parameters.

## Independent analytic anchor

For control blend with scalar EMA, sigmoid(decay)=1/2, content Read weights zero,
two candidates, controls=1/2, observe-all, no clear, initial states (2,4), first
inputs (1,3), second inputs (2,0): final states=(2.5,3.375), sum=5.875. Expected
per-node gradients to initial states are 0.5625, first inputs 0.375, second inputs
0.5, decay parameters (0.4375,0.9375), Read weights (1.4375,0.5625). Full consumes
the pre-blend comparison. An absent third node and independent sample must remain
disconnected from this root. Check declared connections even when coefficient zero.

Add a custom content/clock/active/control-dependent Next that uses all six inputs
and complete slots. Cover passive candidates versus absent nodes, selected clear,
full trace/state/pending/history and VJPs across streaming/frontier, native serial/
parallel/batch, fixed-topology anchors, SettleGraph, cuts/cursor/detach/checkpoint,
sharing and optimizer state in FP64/FP32. Assert zero state-prefill blocks for
control blend while later Full batching remains present. Exercise a standalone
native custom program and adapter rejection of unmatched Python overrides.

Graph identity must change; existing value/slot checkpoint payload need not.
General integer/tensor region history and structured controls follow this gate,
with checkpoint/detach/VJP obligations, then LH selector/profile/Pronounce and
original-C++ inference comparison. Do not encode region history in node state.
