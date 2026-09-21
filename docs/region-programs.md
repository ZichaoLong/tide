# Region programs, history and controls

`region.py` and `tide/region.h` implement SelStep independently of scheduling.
Python `Model(region_programs={region: module})` registers parameters; native
clients install a functional `RegionKernel` in `Model.regions[r].kernel` and its
tensors in `RegionWeights.extra`. The Python adapter exports supported built-ins
and rejects Python overrides. `Region.selector` is a versioned graph profile;
native graph identity is v11. A custom implementation must use its own profile
and maintain its code/parameter interpretation across restoration.

## Request and result

`RegionInput(history, time, candidates, layout)` contains the old complete
region history, logical time, canonical `(node, descriptor)` pairs for **all**
nonempty-fiber candidates and borrowed graph-owned membership/budget/policy.
Membership uses sorted global node IDs with graph-owned local slots; parameters
can be shared between equally shaped regions without capturing physical IDs.
Native membership is a flat row index compiled once.

The program returns `Selection(active, controls, history)`. Active may be empty;
it must be a subset of the candidate set with size at most the region budget.
Controls must cover exactly all candidates, including passive ones. The current
generic control type is one finite floating tensor per candidate, in the payload
dtype/device; it need not be a scalar or a probability. Built-in projection Emit
and control-blend Next explicitly require scalar controls. Named heterogeneous
control records are a future interface extension if an actual profile needs them.
Descriptors are finite scalars under an explicit payload/FP64 precision policy.
Region requests carry payload dtype/device separately; see `lh-selector.md` for
FP64 norm accumulation and built-in control/history conversion rules.

Empty candidate regions call no program, allocate no initial history and preserve
stored history. A nonempty region with empty selection still runs Upd, Read and
Next on its candidates, updates history and executes no Full/delivery. The
independent fixed-topology loops follow the same rule. Selection/history scanning
is causal across frames; region history does not block exact state prefill when
State and comparison-identity Next permit it. No joint region batch is claimed;
`region_steps` counts actual nonempty selector calls in native and block schedules.

## History ownership and persistence

`History(last_time, scalars, node_maps, tensors)` contains named signed int64
scalars, named maps from regional node IDs to signed int64 values, and named
finite floating tensor slots. Field names are nonempty. Profiles validate their
layout and domains; default selected counts are nonnegative. `last_time` lies
between -1 and the current event (strictly before a continuation cut). Programs
are functional: no mutation of old history, parameters or borrowed request data.
Custom programs may preserve/reset clocks according to their declared policy.

Default selected-count increments and built-in state observation increments use
checked nonnegative int64 addition in scalar, batch and sequence paths. A value
at maximum int64 is valid at a cut; an attempted increment fails. This does not
impose an increment rule on custom state/history programs.

History belongs to `(sample, region)`, even when parameters are shared. Forks
copy metadata and retain tensor graphs. Detach cuts all history tensors together
with state and pending messages. Checkpoint **v4** encodes plain records and
detached values; old versions are rejected. Identity, parameter aliases and
record validity are checked before changing weights. The native cursor clones
history tensors at import and snapshot, and detaches them explicitly. Advancing
touches active region histories, without scanning or cloning all stored regions.
The current functional updates copy a touched history's metadata and validate its
result; optimizing very large individual histories remains performance work.

Traces expose post-selection histories for every candidate; comparisons include
all fields. The `history` objective roots tensor slots, and `all` includes them.
SettleGraph embedding shares body region programs, gives boundary regions separate
default programs and maps initial histories without changing body node IDs.

## Built-in anchors

| Profile | Selection and history |
| --- | --- |
| `count-v1` | Ascending selected count (optional), descending descriptor, ascending ID; increment selected counts; softmax over all candidate descriptors |
| `positive-v1` | Same rule, but only strictly positive scores are eligible; controls still softmax over every candidate |
| `lh-count-affect-v1` | Prior selected count ascending, prior affected count descending, descriptor descending, ID ascending; increment affects for every candidate and selections for active nodes |
| `tensor-history-v1` | Scores `d_v + memory*bias[local_slot(v)]`; history `memory' = alpha*memory + sum(d)`; count/score/ID ranking and softmax over scores |

Tensor-history alpha and bias are registered learned parameters. Its history is
`node_maps['selected']` plus scalar `tensors['memory']`; its default initial memory
is zero. The old memory determines controls, independently of the newly returned
memory. Singleton softmax keeps connected-zero score VJPs.

Independent two-event example: initial memory 2, alpha 1/2, bias `(1/4,-1/4)`,
descriptors `(1,3)` then `(0,2)`. Memories are 5 then 4.5; scores `(1.5,2.5)` then
`(1.25,.75)` select nodes 1 then 0. Final-memory derivatives are 1/4 to the initial
memory, 6 to alpha, 1/2 to each first-event descriptor and 1 to each second-event
descriptor. The bias is disconnected from this root. For the second node-0
control let `s = sigmoid(.5)*(1-sigmoid(.5))`: derivatives are `s/4` to initial
memory, `s` to alpha and `(5s,-5s)` to bias. Tests separately cover these roots,
independent samples, empty regions, sparse activation, cuts, checkpoint/optimizer,
sharing, serial/parallel/packed/frontier and specializations in FP64/FP32.

`cpp/test/region_programs.cpp` supplies a custom native selector with signed
metadata, candidate counters, tensor history, vector controls and alternating
empty selection. Its hand-computed values/VJPs are independent of Python.
Qualification status is in `STATUS.md`; this contract makes no performance claim.
