# Tick-decayed Add: explicit repeat profile

`memory="lh-add-repeat-v1"` stores the post-candidate hidden vector, last adopted
tick (`State.last_time`) and observation count. It has no additional state slots.
Its learned `extra['add_retention']` is a finite scalar in payload dtype, default
`1.0-0.01`. Import LH's **computed** `1.0-decay_rate` into payload dtype; do not
subtract an already-rounded payload-dtype rate. Finite negative or >1 retention
is allowed by the formula, without a stability guarantee.

For a candidate at theta, start with old.value and multiply by retention once
for each tick `old.last_time+1,...,theta`, in that order. Add the complete
Aggregate summary afterward; increment the observation count by one. Initial
last_time=-1 therefore includes decay at tick zero. This is a literal repeat
profile, never a power, scan or shortcut based on zero values. Such regrouping
changes rounding and may change hard routes. Next/clear operate as usual:
selected clear zeros the persistent value but preserves the pre-clear comparison
used by Full. Counters count observations, not elapsed ticks.

Empty fibers cause no state allocation, transition or clock change. Python
`lazy_add.decode(weights,state,cut)` and native `decode_add_repeat` return the
physical hidden after `[0,cut)` by repeating decay through `cut-1`. They do not
modify the encoded state, count, queue or cut. Only an explicit decode visits
idle ticks; sparse cursor advance does not decode every node. Initial nonzero
hidden and wholly absent samples can be decoded in the same way.

Old-mode Read sees the **stored representation**, not a secretly advanced hidden.
LH compatibility uses proposal-mode Read and observe-all/adopt Next. Selected-only
adoption and control-blend remain distinct Tide programs: blending their stored
values is not an eager physical-state interpolation. SettleGraph uses its actual
`stride*position+rank` logical clock in both direct and encoded implementations.
It does not reinterpret one token as one Add decay tick.

## Training, continuation and execution

Autograd differentiates the repeated multiplies, summary addition and explicit
decode. Initial hidden, retention and source/input gradients follow ordinary
Torch rules, independently of LH's in-place/custom backward. Even multiplying
zero retains a connected-zero VJP; decoding zero elapsed ticks leaves retention
disconnected. Observe-all analytic example: rho=1/2, initial=2, inputs 1 at tick 2
and 2 at tick 5 gives states 1.25 and 2.15625; physical cut 8 is .5390625. Its
derivatives to `(initial,x1,x2,rho)` are `(.00390625,.03125,.25,2.4375)`.

Cut composition and the eager LH interpretation require fixed parameters across
the composed forward windows. A checkpoint preserves the encoded value and clock,
not an implicit materialized idle suffix. If an optimizer changes retention while
carrying that state, subsequent deferred decay uses the new retention: this is
the declared lazy recurrence, not an eager run that previously applied the old
retention. An eager interpretation across parameter epochs would require an
explicit rebase/retention-history design; it is not implemented. Detach truncates
the encoded value's gradient, not the elapsed logical clock.

Native streaming buckets independent candidates by equal elapsed ticks and
multiplies stacked payloads. It uses the existing node-parallel worker pool and
public-root semantic replay in training. The sequence contract is exact but uses
a visible causal state loop (`joint_sequence=False`, scalar sequence step counts).
Frontier can still batch Aggregate, Read and Full. A time scan or fast long-gap
power profile is not silently substituted. Costs grow with deferred tick gaps;
no large-gap speed claim follows from sparse allocation.

The profile name already participates in graph identity v11; checkpoint v4 stores
the required value/time/count. No schema change is needed.

## Original Add comparison

The optional runner `scripts/check_lh_selector.py --component all` also builds
`lh-add-check`, linking unchanged snapshotted AccumulateLocal, Hidden, BatchHidden,
Confluence and ModuleUtils sources. Original single hidden, no-grad batch cache,
and batch cache plus individual hidden run under no-grad. Sum Confluence is the
current comparison profile. It checks pre-clear snapshots, norm descriptors and
decoded post-clear hidden at each tick against Tide serial/packed/frontier, plus
whole-window execution. Inputs cover sparse source IDs, full ragged fibers,
present zeros, nonzero initial state, missing samples, idle prefixes/suffixes,
retention 0/1/.99 and selected clear.

This is an Add component gate, not original CHAL/IOCortexNet output equivalence.
Tide independently tests VJPs, sharing, cuts/detach, checkpoints, positive-delay
cycles, TimedDAG and SettleGraph specializations. Current evidence is in STATUS.
