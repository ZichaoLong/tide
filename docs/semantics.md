# Adopted models

Fixed core-5 and local/instance revision:
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`.
See the [adopted instance bindings](../math/instances/backbone-comparison.md)
and their HB references; [semantic map](semantic-map.md) locates the mathematics.

FC, HB-Lattice and Exchange use Settle timing. Multiscale supports Settle/Timed.
K = KV/sum; S = SSD/state; X = D/KV and H/SSD (X equals K without hubs).
All use Read+, all positive-reception updates retained, encoded node descriptions,
regional self-attention and the chosen role quad/Shannon or ordinary pool.
Three sharing profiles preserve private node state. Original logical times govern
KV windows, positions and SSD convolution. k bounds mass, not active node count.

The explicit [memory binding extensions](memory-candidates.md) define X_reverse
(D/SSD, H/KV), total Q/K/V internal-width narrowing, Full FFN width, and
memory-only regional sharing. These are selected instance changes, not equivalent
optimizations of the fixed snapshot; `math/` stays byte-for-byte fixed.

Correctness tests target actual values, support, parameter gradients and stateful
continuation. Full action journals and generic intermediate decoding are omitted.
The [public API and private state](instances.md) belong to the active instance
implementation; [architecture](architecture.md) maps its current modules.

FP32 and explicit BF16 matrix arithmetic with FP32 parameters and scalar/SSD
state have been exercised on CPU and NPU. This arithmetic policy does not change
the adopted mathematics or imply identical rounded support, gradients or resumed
updates.
See [precision](precision.md) and the [fixed numerical evidence](evidence/roadmap-numerics-followup-20261118.md)
for actual scope and retained failures; [STATUS](STATUS.md) owns the current handoff.
CPU FP64 is a bounded accuracy diagnostic. FP16 is outside this cycle.

Zero reception performs no KV/SSD write, including its first-order boundary:
an absent padded write has zero derivative. This is separate from the reusable
Attention primitive's right derivative at an explicitly supplied zero key prior.
Whole-window and continued KV evaluation use the same no-write convention.
KV Full tiles original logical positions and restricts each key bank to the
finite history window; sparse event indices never replace logical time.

Quad compares the budget, received mass and box totals before rounding away
their sum tails. The selected operand owns the total derivative. A mathematical
interior can round to a stored endpoint while retaining its interior VJP.
For ordinary quad allocation, a zero control on the constant lower face with a
positive upper bound has zero derivative with respect to all producer inputs.
Reception carries this fact along the corresponding edge and may omit that
key's hypothetical independent prior derivative. The attention quotient, real
edge membership, and query rows are preserved. Movable bounds, fixed-through
zero upper bounds and positive priors retain their original derivative rules.
Single-region and batched regional scoring use this proof for a zero node prior
when every real incoming edge is certified locally constant. Padding is excluded
from this reduction. One independently variable incoming edge retains the
node prior's right derivative.
Timed pending banks carry the face across the one-tick delay. Detached strengths
are constants for the next update. Independently differentiable strengths
(including a re-rooted pending bank) retain their right derivatives. Older
two-field pending checkpoints remain readable.

In Multiscale Timed, a first-layer sender has zero available mass on ticks
without external injection. Routing derives these constant zero priors from
the current window's injection clock, including a flush, and shifts them with
the one-tick delay. The preceding pending bank supplies its own provenance.
Every consumer of a routed mass receives the zero tangent on a certified
constant. A sender whose real inbound masses are all certified zero also has
an identically zero upper bound. This proof propagates through true edges
with their delays, across attached windows and placements. Clock and derived
proofs belong to the live computation. Checkpoints retain the original private
state format. A genuine independent prior whose tiled attention
adjoint becomes nonfinite reports its source index and dtype.

For total-scaled attention, `pool_support_policy="positive-floor-v1"` retains
mathematically positive senders that round to zero at the output dtype. On
those rows the least positive subnormal represents an otherwise lost sender.
Storage adjusts donors only if the row total misses the budget by more than
half a budget ULP. Subnormal budgets therefore conserve integer storage
units exactly. The VJP uses the original unrounded normalization, including
the right derivative of true zero priors. The default `strict` raises on
support loss. Sampled work includes cumulative training-mode forward row
and repair counts since model construction under `pool_support_training_forwards`.
