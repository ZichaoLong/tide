# Periodic local state clocks

`Node.state_clock = StateClock(period, first, count)` declares valid event phases
`[first, first+count)` in every global period. The default `(1,0,1)` is the existing
global clock. Period/count are positive int64, `0 <= first < period` and
`count <= period-first`. The native policy is three integers, with no per-node
phase-array allocation. Non-contiguous phase sets are outside this initial API.
StateClock is a declarative data policy; Python subclasses with callable overrides
are rejected instead of silently exporting different native behavior.

For global event time t, write `(k,p)=divmod(t,period)`. A valid event has local
time `k*count+p-first`. Events outside the interval fail. Local event u maps back
to `floor(u/count)*period+first+u%count`; inverse overflow fails. For any complete
global cut c, the local cut is `floor(c/period)*count +
clamp(c%period-first,0,count)`. The initial state timestamp -1 remains -1.

## State-program boundary

Only Upd's time argument and State.last_time are converted. The wrapper passes
local previous/event timestamps to step, batch, sequence or packed_sequence and
returns global stored timestamps. Tensors, observations, cache rows, source tags,
content and contribution slots retain their meanings and ownership. Reset runs
through the same conversion and preserves clock metadata. Persistent off-clock
timestamps are rejected even when the state is idle.

Full/Emit phase policies, Read/Next requests, region history, input ledgers,
seals and physical edge send/arrival times use global time. No autonomous candidate
or idle update is introduced. In particular, for body `(L+1,0,L)` and readout
`(L+1,L,1)`, the reserved readout phase does not add a body decay tick.

The wrapper delegates the underlying exact/joint capabilities and work counters.
Python programs without packed_sequence keep the existing per-segment sequence
fallback; a wrapper is not evidence of joint batching. Native custom immutable
StateKernel programs may also be wrapped. Custom programs still see physical
source tags, so time/tag-sensitive cross-graph equivalence requires its own proof.
Python-only custom kernels still have no implicit native implementation.

## Decoding, ownership and continuation

Physical Add and fiber-bias Python decoders derive the clock from the wrapped
program. Native decoders infer it from a configured native kernel or accept an
explicit optional StateClock. Raw NodeWeights made through bindings have no native
kernel; supply the clock explicitly when decoding their non-global state.
The decoder projects both stored last_time and complete cut before lazy decay.

An entire shared program must match its graph clock. Individual tensor parameters
may share across different clock wrappers. Graph identity records all three
integers; checkpoint payload has no additional mutable fields. Clock changes
reject saved graph identity before weights change. Continuations store global
timestamps so window validation, input positions and pending messages keep their
existing protocol. See `semantics.md` for current versions.

Tests use independently scheduled local-time graphs as anchors for EMA, SSM,
Linear/Delta, event attention, LH Add and learned same-fiber attention. A cyclic
phase-edge fixture combines SourceDomain aliases with one/two-tick delays and
compares every global cut, including the reserved phase and pending messages.
Qualification status is in STATUS. These primitives alone do not establish the
whole-model single-PDG projection or phase-occurrence ledger reconstruction.
