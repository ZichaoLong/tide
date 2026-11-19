# Positive-chain arithmetic after TL-REQ-005 review

This follow-up addresses current FC/HB formulas, not the retired Bern/direct
bindings. It changes evaluation and first-order VJP organization without changing
fixed mathematics, epsilon, support policy, parameter ownership or state.
Qualification is source/device scoped and recorded separately in STATUS.

## Normalization and reachability

For `z=RMSNorm(x)` the mathematical bound is `||z||₂ <= sqrt(width)`.
The following score still contains `z·readout + offset`; both parameters are
trainable without magnitude constraints. Regional Q/K projections also have
trainable weights. Content normalization neither gives received masses a
positive lower bound nor bounds a weighted-prior derivative.

Normal FC ingress and full interlayer connections imply memory weights of one.
For the complete 9B `5-dd4/group/all/Settle` binding, Hub controls are at least
1/4 and subsequent Hub reception at least 1/2. Sparse hooks do not give every
ordinary D node a positive reception lower bound. Each D region still has total
received weight at least one. A whole-region weak-pool witness is therefore not
a reachable-state proof for that particular 9B quad configuration. Individual
weak priors and regional Attention remain separate questions.

Old real training failures establish practical relevance, not a probability
estimate for new bindings or a current 9B failure. Current local witnesses are:

- pool: zero scores, `mu=[1e-40,1e-40]`, cotangent `[1,0]`; the correct mass
  derivative is `[1,0]`, not infinities.
- pool: scores `[90,0]`, `mu=[exp(-90),1]`, cotangent `[1e-30,0]`; total mass
  rounds to one but the first finite mass derivative is about `3.05e8`.
- ordinary Attention: zero scores, priors `1e-40`, values `1` and `1+2^-20`,
  unit cotangent; finite prior derivatives are about `±2.3842e33`.
- Shannon: 63 scores of 2 and one of -98, unit received weights/caps, budget32;
  the final weak sender is about `1.88956e-44` although its probability alone
  is below FP32 range.
- Shannon: equal weak interior slopes and cotangent `[.25,0]`; the coupled
  received derivative can be `.125` even if slope*cotangent first rounds to zero.
  The final score derivative may legitimately round to zero separately.

These are bounded mechanism checks, not recreated downstream learning runs.

## Allocation

For pool, `M=sum(mu)`, `a=min(1,M)`, `p=softmax(score+log(mu))`, `c=a*p`.
Here `logZ=log(sum(mu*exp(score)))` denotes the mathematical normalizer,
before removing the common scales used by the implementation.
Given unscaled cotangent g and `gbar=sum(p*g)`, the composite derivative is

```
dscore_i = a*p_i*(g_i-gbar)
dmu_i = a*exp(score_i-logZ)*(g_i-gbar) + 1[0<M<1]*gbar
```

Remove common prior/score scales before normalization. Retain binary parts
through products, centering, the mass-branch sum and broadcast reduction until
the public root is formed. Real zero priors keep their right derivatives; an
all-zero row has zero VJP. At M=1 pool takes the constant first-min branch,
different from KV Full's clamp derivative at M<=1. Preserve source versions.

Constrained allocation retains its active-face solution derivative. Its coupling
average is `sum(slope*g)/sum(slope)`: slope1 for quad interiors, c for Shannon.
The mean and centered product retain binary scale instead of forming tiny
slope*g or an oversized numerator first. Shannon reconstruction combines the
residual total with exponentials before converting the sender to storage dtype.

## Ordinary Attention

Auto sends visible positive priors below `sqrt(accumulation.tiny)` through the
bounded scaled-prior path in both inference and training. Ordinary Attention
uses amplitude one and no total-mass derivative. math/tiled use the same path
in that range. The threshold selects execution; it does not change weights.
Explicit ordinary native calls reject subnormal priors; mass-scaled native calls
retain their broader weak-prior rejection. Other native calls keep their kernels.
Explicit softmax/tiled prior VJPs combine the exponential and cotangent before
materialization. Dispatch observation distinguishes the ordinary representation.

## Remaining boundaries and reproduction

The first frozen implementation `b4a8d937` exposed an existing TorchNPU CPU
fallback in frexp/copysign. The old stable-mass training and cost logs contain
the same fallback; their numbers remain, with corrected execution attribution.
`3efcc6b1` decodes IEEE binary32 parts on NPU, scaling subnormals by an exact
2^24 before decoding, and uses signbit/where for signed Inf/zero. CPU FP64
continues to use torch.frexp. Device checks disable pytest capture and include
scalar, strided, broadcast and subnormal boundary inputs; kernel placement is
also checked by a bounded trace. This is separate from numerical equality.
The NPU format cast and contiguous conversion may allocate device copies;
removing host fallback does not remove layout or workspace costs.

Final positive Shannon/pool senders outside storage range still raise. Finite
learned offsets `[-110,1]` following normalized features demonstrate this in a
current small FC model. That is a valid parameter configuration, not evidence
of an optimizer trajectory or frequency. Continuing beyond this limit requires
an explicit representation/storage policy. No floor, clipped derivative or
silent whole-model FP64 substitution is introduced.

Arbitrary logits, huge centered dot products, masked extreme anchors, cancellation
and later shared parameter accumulation remain bounded-range limitations.
Attention probabilities can still underflow before a later binary-scaled
product, and its forward numerator can overflow before denominator division.
The scaled product cannot recover information already lost at those points.
Shannon still uses solved/stored c for its slope; these checks do not establish
a general unquantized log-slope contract at every subnormal face. RMS square
overflow and SSD extreme parameter/scan contractions remain separate candidates.

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests/test_allocation_range.py tests/test_ordinary_attention_range.py tests/test_numerical_reachability.py
```

`scripts.qualify_positive_chain` reuses the existing training/continuation and
actual new-process restored-update checker, including two-candidate pool/Shannon
regions, FP32 and a separately reported BF16 case. Submit frozen inputs through
`scripts/launch.py`. Device tests accept `TIDE_STABLE_DEVICE=npu:0` inside admitted
jobs; FP64 formula references remain on CPU.
