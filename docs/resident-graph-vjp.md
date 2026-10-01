# Device graph reverse progression

The internal `append_graph_vjp` composes Full and state-chain adjoints with
message dependencies from the actual resident forward journals. Its current
profile is single-device HARD/HST/SOFTP, built-in Aggregate, broadcast emission (including
static phases), identity/EMA/Add-repeat/event/fiber-attention state and identity/tanh/LH/SwiGLU Full. Unsupported
modules refuse when requesting `ContentFlow::reverse_tape()`. The separate
[public training owner](resident-training.md) provides retained-window lifecycle;
qualification is recorded per module in ROADMAP.

FP32 and FP16 forward tapes use FP32 cotangents and adjoint accumulation;
half roots fail explicitly. Half forward values, parameter banks and per-operation
rounding remain intact. The integration gate covers sum Aggregate, identity/EMA/
Add-repeat state and identity/tanh Full, with all three Emit modes. The separate
extended retained gate adds normalized Aggregate, LH and SwiGLU. Half attention
modules have local component checks and require their own whole-graph integration
qualification. Public FP16 training and master/checkpoint lifecycle remain guarded.

Static topology/parameter layout preparation is allowed. A device hash table
indexes actual `(sample,node,int64 time)` events, associates physical messages
with producer/consumer events, and makes stable contributor lists and stage
boundaries. Parallel edges remain separate rows. A produced message must point
to an active earlier stage. Messages produced before the complete cut are incoming
boundary leaves, preserving their gradient for an earlier retained window; their
value is not differentiated with respect to this window's current parameters.

The device loop processes actual forward stages in reverse order. For each stage,
it gathers Full cotangents from output/pending roots and already differentiated
consumers, runs the packed Full VJP, seeds the state-chain VJP, carries earlier
state adjoints, and differentiates Aggregate into physical message gradients.
[Normalized profiles](resident-aggregate-vjp.md) additionally preserve each
event's logical coefficient domain and reduce its parameter partials.
Those messages supply earlier stages on subsequent iterations. No event count,
message association, stage choice or gradient connection bit is returned to the
host to advance this loop. The CPU reference independently runs the graph; none
of its numerical results or routes are inputs to the candidate.

Cotangent connection bits preserve None versus connected zero independently of
numerical values. Absent root payloads may contain NaNs and are never evaluated.
Contribution lists provide unique feature-tile writers without conflicting
scatter. Parameter contributions retain deterministic row order. Node parameters
are per-node partials; physical input/Aggregate/delivery scales are reduced on
device. A public parameter registry must still accumulate aliases before optimizer
use. Read parameters have no gradient in HARD. The [control/Read component](resident-control-vjp.md)
adds complete-frame softmax and linear/FP32-norm Read adjoints for HST/SOFTP,
merging direct content/old/proposal roots before state-chain reverse.
[Event attention](resident-event-vjp.md) consumes proposal cotangents and
independently reverses each actual KV owner chain inside the graph stage loop.
Its proposal depends on prior KV, not old visible state. Cache roots/bridges,
evicted rows, clear and shared projection owners retain separate connections.

Outputs include dense initial-state adjoints and all physical message adjoints;
`producer == -1` identifies boundary leaves. The caller binds dense initial rows
only to the state tensors that existed at the incoming cut. Automatically created
zero state buffers are constants, not newly declared trainable inputs. Retaining
an earlier window requires saving its actual tape and connecting these boundary
adjoints; a physical chunk never implies detach. This component alone does not
provide that retained-window lifecycle or optimizer.

The tape borrows the forward owner's tensors/topology, with the same lifetime
rule as state/Full tapes: use before the next advance or owner destruction.
A captured program can replay the same tape with new seeds; it clears previous
adjoints and counters. Construct a new reverse program for another window, or
implement an explicit saved-tape owner before allowing overwrites. Capacity is
per window; no gradient/event truncation occurs implicitly.

Tensor admission reserves space separately for reverse links, Full, state VJP
and graph-owned buffers before allocation. Caller input/tape storage and the
`CannProgram` serial operator workspace are separate budgets. Preflight rejects
malformed stage/message topology before numerical reverse work; runtime/device
errors remain sticky and invalidate the result. Empty windows preserve final-to-
initial connectivity and have no fabricated parameter gradients.

The gate compares independent CPU Streaming autograd at FP32 and FP64 for
feedback, self-loop, DAG and edgeless cases; streaming/greedy; separate output,
state and pending roots, combined roots, absent and connected-zero roots; warm
continuation above 2^55; replay; widths 1/3/257; explicit malformed/budget/dtype
refusals. It checks every boundary input, initial state, Full/state/scale gradient
and connection, with a separate device placement profile. The half oracle uses
independent CPU Streaming plus test-only kernels that retain each declared half
rounding boundary with FP32/FP64 autograd leaves. Sum accumulates FP32 source
products before rounding its result; physical delivery rounds before a later
Aggregate consumes it. Half roots are scaled by256, with VJP rtol2e-3/atol2e-5;
FP32 thresholds remain unchanged. Passing evidence must
be tied to an immutable source revision; this document is the contract.

Internal [owner updates](resident-optimizer.md) and [retained-window bridges](resident-retained.md)
now compose with this component under their own qualification scopes. Remaining
training work includes the remaining half graph modules, public FP16 training and peer
reverse progression. This is neither complete matrix qualification nor full-size
training throughput evidence.

[Physical-slot HARD adjoints](resident-emission-vjp.md) extend this path with
slot-affine projection gradients, unscaled delivery values and parameter publication.
Controlled slot-affine and compact projection placement remain separate work.
