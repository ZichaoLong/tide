# Next gate: complete content, Read and Next programs

Planned, not an implementation/support claim. This refines the corresponding
`ROADMAP.md` gate; `STATUS.md` remains the only current handoff. Replace this plan
with the implemented contract after qualification.

## Semantic boundary

The locked PositiveDelayGraph note, equations (5), (11)--(15), defines region
descriptor mode tau as content, old state or proposed state. Upd still computes
the candidate proposal for every nonempty fiber. Comparison adoption depends on
the region's observe-all policy and selection. Next receives old, comparison,
time, content, active and control; it cannot read/recompute this event's Full.
Noncandidates call none of these programs. Full always reads comparison even when
Next clears or otherwise replaces persistent state.

## Implementation sequence

1. Expose complete content to local programs: summary, program-visible source
   atoms with local slots, and optional Aggregate contributions. Resolve physical
   scales/slots before applying source-origin views. Preserve raw physical fibers
   separately for routing/trace. Reuse one graph-owned projection implementation;
   do not make custom State/Read/Next/Full invent their own embedding adapters.
2. Extend State step, independent batch and packed sequence metadata to carry
   complete content. Built-ins continue consuming summary. Provide Python custom
   state-program injection with registered parameters and native adapter guards;
   a Python override must never silently become a built-in native kernel.
3. Separate Read from state formulas. Put `read_mode` on Region, default proposal;
   put versioned readout profile/program on the node. A Read request receives only
   its mode-appropriate state (none for content) and content/time. Built-in linear
   Read uses existing `w.read`; identity boundary Read stays zero. Custom programs
   get the complete selected state, including slots/clocks. LH's FP64 norm is a
   later profile with an explicit descriptor/control precision policy.
4. Separate Next from scheduling. Native/Python requests expose all six inputs.
   The initial adopt/clear profile preserves previous behavior. A control-blend
   example and custom clock/content-dependent example exercise the full seam.
   Validate resulting states with the state kernel and continuation constraints.
5. Gate state prefill on an explicit Next identity/comparison contract in addition
   to observe-all and exact state sequence support. Default custom Next must fall
   back to causal state/selection/Next. Full can still batch across closed frames.
   Never precompute a recurrence that ignores controls consumed by Next.

Keep local kernels, graph fields, scheduling, replay and tests in separate files.
Native Next work can compute independent node results in the pool and commit in
canonical order. Python schedules remain independent. Add capability/fallback
counters where an API call can hide scalar work.

## Replay and validation

Split proposal evaluation from Read so semantic state replay no longer calculates
and discards a descriptor. For every prefilled event, retain the correct previous
bound state, including under no_grad, before evaluating old-state Read. Packed
Read must use bound proposals and per-event content, not reconnect independently
owned leaves through a common stack. First-order isolated-root scope stays the
same; arbitrary internal adjoints and higher-order AD are not implicit promises.

Acceptance requires independent formulas and route tests for all three Read
modes, control-sensitive Next recurrence and gradients, selected clear with Full
reading pre-clear slots, passive candidates versus absent nodes, and a custom
source-aware state under direct/encoded SettleGraph. Compare generic streaming,
frontier prefill/fallback, serial/parallel, batch packing, fixed topology anchors,
cursor/cuts/detach, sharing/checkpoint and optimizer state in FP64/FP32. For a
control-dependent Next, assert zero state-prefill blocks while later Full batching
remains valid. A standalone native client must exercise the new seams.

Graph/profile changes require new identities. Existing value/slot checkpoints
need no new payload schema unless their stored state structure changes. General
integer/tensor region history and controls are the following gate, with explicit
checkpoint/detach/VJP obligations; they must not be smuggled into node state.
