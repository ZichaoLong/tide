# Device Emit, control and Read adjoints

This component extends the resident FP32/FP16 broadcast path with HST and SOFTP.
HARD remains the default. [STATUS](STATUS.md) distinguishes implementation,
development checks and immutable qualification; this document defines the
contract and does not certify a build or throughput.

Select `ResidentLimits.mode` and `zeta` in C++, or the existing
`ExecutionOptions(mode=..., zeta=...)` in Python. Identity boundary nodes bypass
Emit. Ordinary identity Full still passes its `g==h` value through Emit and keeps
the declared connected zero control gradient. Other selected Full values use
the same formulas in [semantics](semantics.md):

- HST returns the fresh Full value `g` exactly. Its declared VJP sends `u` to
  `g`, a connected zero to content `h`, and `zeta*dot(u,g-h)` to control `p`.
- SOFTP returns `h+p*(g-h)`. Its VJP sends `p*u` to `g`, `u-p*u` directly to
  `h`, and `dot(u,g-h)` to `p`.

Only actual selected Full rows execute these formulas. Absent output roots do
not evaluate their payloads or create connections. A zero root, zero `zeta`,
or `g==h` still has the declared connected gradient. SOFTP retains the actual
unmixed Full value alongside the emitted value; no division or subtraction
from a rounded mixture attempts to reconstruct `g`.

FP16 stores actual half payloads and Read weights. SOFTP rounds the public
control, `g-h`, its product with that control and the final addition to half.
HST still returns `g` exactly; its declared saved difference also rounds to
half. Internal frame probabilities and Read scores remain FP32. The control
VJP uses the rounded control/difference for Emit and the original FP32 frame
probabilities for the softmax Jacobian. Cast VJPs are first-order identity;
all cotangents, Read partials and owner reductions remain FP32. This does not
claim bitwise equality to a backward that accumulates in half.

The reverse program groups the candidate's own event records by
`(sample,region,int64 time)` on device. It builds bounded hash/linked tables and
uses stable contributor order. For each connected frame, every candidate
participates in the softmax Jacobian:
`bar_score_i = p_i*(bar_p_i-sum_j p_j*bar_p_j)`.
An unselected candidate can therefore receive a gradient. Integer ranking,
selection history, readiness and event association have no differentiable path.

Read then contributes to the declared content, old-state or proposal coordinate.
Linear Read produces both the vector cotangent and a physical Read-parameter
partial. FP32 norm Read uses its own recorded norm and has a connected zero
derivative at the zero vector. Identity-node descriptors are constants. No
disconnected Read weight or unused Full value is evaluated.

Vector payload work is tiled across device cores. Device tasks own frame
metadata, connection flags and stable per-node reduction. The resulting Read
partials enter the same alias registry as Full, state and scale partials before
one optimizer update. Updated Read aliases are published to the forward banks.
Retained windows freeze their actual Read banks and unmixed values, preserving
parameter-generation ownership. No reference route, score, value or gradient
feeds the candidate.

The component reserves its own bounded tensor workspace in the graph reverse
budget. It rejects a frame/tape that exceeds admitted capacity; it does not drop
candidates, split the logical softmax denominator or truncate gradients. Replay
clears scratch, numerical adjoints and connection flags, including empty windows.

Training checkpoints add explicit `mode` and `zeta` fields to the existing
resident v1 record. Restore requires the same values. Legacy records that omit
them mean HARD with `zeta=1`; the HARD parameter layout remains unchanged.
Graph identity, eager checkpoint schemas and the independent CPU reference are
unchanged. Non-HARD slot-affine emission is explicitly unavailable in this
increment. Complete FP16 graph/retained-window/public training and peer reverse
progression remain separate from this component and forward inference.

The isolated checker covers complete and ragged frames, all three Read modes,
linear/norm/identity descriptors, zero norms, None/zero roots, zero surrogate
scale, int64 times above2^55, widths1/7/257, poisoned unused storage, short/empty
replay and capacity refusal against independent CPU FP32/FP64 autograd. Public
trajectory checks additionally exercise actual independent forward scheduling,
shared Read/Full/state owners, normalized Aggregate, retained-window backward,
optimizer updates and checkpoint continuation. HST is tested against its declared
VJP, never a finite-difference derivative of its hard forward.

`profile_device_control.py --check control-training` records placement separately
from timing. Its bounded `--storage-limit-mb` defaults to 200; a complete training
gate may require a larger explicit limit to retain the task/metadata association.
Missing operator records are a failed profile, even if the training checks pass.
`--check control-vjp --dtype float16` separately covers selected Emit rounding,
identity/invalid/error bypass and FP32 adjoints. `--check precision-control-flow`
covers actual independent CPU/NPU inference with both Emit modes, state and
attention profiles, both schedules and continuation. These checks do not
certify complete FP16 training or throughput.
