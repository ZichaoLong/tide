# Device graph reverse progression

The internal `append_graph_vjp` composes Full and state-chain adjoints with
message dependencies from the actual resident forward journals. Its current
profile is single-device FP32 HARD, sum Aggregate, broadcast emission (including
static phases), identity/EMA/Add-repeat state and identity/tanh Full. Unsupported
modules refuse when requesting `ContentFlow::reverse_tape()`. Public resident
sessions remain inference-only until the training API and lifecycle are qualified.

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
state adjoints, and differentiates sum Aggregate into physical message gradients.
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
use. Read parameters have no gradient for this HARD profile.

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
and connection, with a separate device placement profile. Passing evidence must
be tied to an immutable source revision; this document is the contract.

Remaining training work includes alias-owner updates, optimizer/finite checks,
retained multi-window graphs and public autograd/explicit training interfaces;
normalized Aggregate, LH/SwiGLU, attention and HST/SOFTP adjoints; FP16 and peer
reverse progression. This is neither complete matrix qualification nor full-size
training throughput evidence.
