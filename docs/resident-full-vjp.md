# Device built-in Full VJPs

`tools/device_online/full_vjp.h` is an internal first-order FP32 component for
resident training. `ContentFlow::full_tape()` borrows the actual device event
journal and frozen built-in Full parameter banks; diagnostics are required.
The owner rejects unsupported Full kinds before exposing this tape. Closed or
failed owners cannot expose it. The next advance overwrites the journal;
retaining a multi-window training graph requires separate storage and lifecycle.

For identity, only the content cotangent is connected. For tanh Full,
`g = content + tanh(comparison @ W + b)`, the component recomputes the activation
from the saved comparison and parameters. It never obtains the activation by
subtracting content from g. Weights/biases have a caller-owned zero sentinel row;
only real connected tanh owners and this sentinel enter batched matrix operators.

Values and connection bits are separate. None remains absent, connected zero
stays connected, inactive Full cannot receive a connected cotangent, and poisoned
absent values are never evaluated then multiplied by zero. One device metadata
writer validates the whole logical tape before numerical work and writes all
connection flags. Malformed count/owner/time/activity refuses with error 2;
unavailable kinds refuse with error 12. The caller must inspect the sticky error
at its boundary. No partial numerical result is exposed after preflight refusal.

Device control selects connected rows, advances bounded chunks and reduces
repeated per-node owners in stable row order. Feature tiles have unique writers;
padding gets distinct scratch destinations, avoiding conflicting index-copy.
The parameter outputs are per-node partials; sharing/alias reduction is still a
separate registry operation. An identity-only profile has no weight/bias outputs.

Tensor admission includes gradient outputs, sanitized banks and bounded matrix
work before allocation. The incoming tape/cotangents and the enclosing
`CannProgram` operator workspace have separate budgets. Empty and shorter tape
replay clears old numerical results, connections and chunk counters. Physical
chunk size changes neither the logical VJP nor the optimizer boundary.

The gate uses independent CPU FP32/FP64 autograd, repeated owners/samples,
nonaligned widths, large residual content, poisoned unused parameters and rows,
None/zero cotangents, chunk sizes 1/5, short/empty replay, explicit refusals and
actual resident forward tapes in streaming/greedy modes. Profiling is separate.
The component does not itself implement graph message backward,HST/SOFTP,
public autograd,optimizer,FP16 or peer reverse progression. Integration and
qualification are separately indexed by STATUS/ROADMAP.


## LH and SwiGLU adjoints

The extended implementation preserves the nine declared LH activation/norm
profiles and SwiGLU's content residual. LH depends only on comparison; it must
not fabricate a content cotangent. ReLU's derivative at zero is zero. RMS and
layer normalization retain epsilon1e-7/1e-5 and unit-affine normalization before
applying the owner's weights. Per-row affine gradients are reduced in stable
row order before shared-parameter alias accumulation. Unused LH affine entries
remain disconnected,including bias in RMS and all affine entries in identity norm.

SwiGLU recomputes gate/up projections from saved comparison and device parameters,
uses packed matrix products and SiLU backward,and reduces gate/up/down partials
by compact physical owner. Its content residual remains connected even when
all matrix derivatives happen to be zero. Repeated or cross-matrix aliases
are summed once by the registry before the optimizer updates that owner.

Both paths pack only actual connected rows on device. Poisoned disconnected
rows/owners never enter activation,norm or matrix work. Distinct safe scratch
rows absorb padding writes; a single metadata writer handles connection bits.
The existing forward banks receive every updated alias on device,including
shared Read aliases. Retained tapes copy the updated banks and static mappings.

Tensor budgets divide the declared Full reserve among present basic/LH/SwiGLU
components before admitting their bounded row work. Graph reverse includes
persistent parameter partials in its own budget. A wide retained SwiGLU example
needs a larger explicit reverse budget than tanh; insufficient capacity refuses
before submitting a partial backward. This is additive module support within
the existing single-device FP32 HARD training contract. [Immutable qualification](evidence/resident-extra-full-20261001.md)
records the numerical conditions below; it is separate from throughput evidence.

## Training numerical checks

The Full training checker states its AdamW epsilon explicitly (`--adam-eps`,
default `1e-5`, as in the foundation training qualification). This does not change
the public optimizer default `1e-8`, normalization epsilon, or tensor tolerances.
Local VJPs and the independent multi-step trajectory remain distinct checks.

The checker also retains a cancellation-sensitive RMSNorm first update at
AdamW epsilon `1e-8`. It compares the actual device VJP to independent CPU
FP32/FP64 autograd, and the actual device optimizer to CPU optimizers supplied
with the same observed device gradient. This is an optimizer oracle only:
no reference result enters the candidate. It records independent end-to-end
parameter discrepancies separately, without calling them passing trajectories.

At the initial development source, one scalar gradient was approximately
`1.35e-8` on NPU, `1.22e-8` on CPU FP32 and `1.29e-8` on CPU FP64. The two CPU
updates themselves differed by `1.4e-5`, exceeding their joint comparison
tolerance. Small gradient error alone therefore cannot certify the optimizer
trajectory. `--case=7 --adam-eps=1e-8` preserves the strict reproducer;
`--numerics-only` runs the separately labelled conditioning checks. These
observations are finite numerical evidence, not a guarantee for arbitrary
normalization inputs, optimizer options or long training horizons.

`--control-check=strict` is the checker's default. The explicitly selected
`conditioned` policy handles softmax sensitivity to already-accepted FP32 score
differences. It keeps every score and all other forward tensors, VJPs and updates
at the original tolerances; routes, frame membership and connection flags remain
exact. For a control mismatch it checks each complete candidate softmax against
a CPU FP64 calculation using that side's own scores. It then requires the
cross-control difference to fit `range(candidate_scores-reference_scores)/4`,
plus the two original local softmax rounding tolerances. The bound follows from
shift invariance and the softmax Jacobian's row L1 norm, at most `1/2`.

This is a named comparison policy, not a claim that the original uniform
control tolerance passed. It changes no candidate computation or continuation.
The width257 SiLU/LayerNorm training fixture exposed this distinction after
one AdamW update: scores that passed the original tolerance produced control
differences around `1.7e-6` to `2.3e-6`. CPU-only comparison tests reject incorrect
softmax values, scores outside tolerance, changed routes and missing candidates.
The component runner records an explicit selection through
`--full-training-control-check conditioned`; profiling uses
`--application-arg=--control-check=conditioned`. Raw strict failures remain evidence.
