# Stable mass reductions in the current instance

This implements the fixed Aggregate and KV Full formulas with stable local
arithmetic. It does not introduce positive-support floors, change allocation,
change epsilon, or add parameters/state. The fixed `math/` snapshot remains
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`. Qualification belongs to the recorded
source and device, not to the existence of these formulas.

## Reception

For nonnegative masses `m[..., E]` and transformed contents `v[..., E, D]`, let
`M=sum(m)` and `u=sum(m*v)/M` for `M>0`. `weighted_mean.py` scales masses by their
largest value before normalization. Binary mantissa/exponent contractions retain
the scale until the final product/reduction, so an intermediate `m*v`, `1/M`,
or `p*v` need not be representable separately. The result stays in FP32 for
FP32/BF16 payloads; CPU FP64 remains an optional diagnostic.

For incoming cotangent `g` the first-order composite VJP is

```
dm[i] = dot(v[i] - u, g) / M
dv[i] = (m[i] / M) * g
```

Centering and scaled contractions precede final conversion to each root dtype.
One positive mass has exactly zero self-mass derivative; other real zero-mass
edges retain their right derivatives. At `M=0`, the existing denominator-one
branch remains: `u=0`, `dv=0`, `dm[i]=dot(v[i],g)`. This is the declared branch
derivative at the origin. Structural padding is masked before this helper and
has zero derivative. Its static mask also reaches the helper, which removes
nonexistent padding mass roots before exponent restoration and range checking.
The existing clamp of negative raw mass remains outside.
Broadcast/shared uses sum their contributions; original source tensors retain
autograd version guards. Nonfinite public results or adjoints raise explicitly.
Those finite checks introduce host synchronization on accelerators; cost must
be measured and is not inferred from reduced arithmetic.

`scaled_arithmetic.py` normalizes the binary fraction and splits exponent
restoration into two bounded powers. The first product remains a normal number;
only the last rounds into subnormals. This avoids a device `ldexp` forming
`2^128` as a separate FP32 infinity when the actual product is representable.
True public overflow still reaches the explicit range guard.

## KV Full

`weighted_attention(..., mass_scaled=True)` composes, for each visible row,

```
M = sum(mask * w)
a = min(M, 1)
p = softmax(logit + log(w); mask and positive support)
y = a * sum(p * v)
```

This option preserves the original KV `clamp(max=1)` derivative at `M=1`.
It is distinct from pool allocation's first-branch `min` convention. Real zero
priors in a nonempty row retain their right derivative; entirely empty rows
use the Attention primitive's zero VJP. Structural masks remain structural.

The explicit math/tiled backward uses the unscaled output cotangent. Weak priors
are first rescaled by their row maximum; direct weighted exponentials avoid an
unnecessary log/exp round trip through tiny weights. The prior VJP combines
`a*exp(logit-logZ)` and `dot(g, v-mean)` with ordinary products where representable
and signed log-space otherwise, then adds the total-mass derivative. Weak Q/K/V
products retain binary scale until multiplication. This avoids the known
huge standalone prior derivative and prematurely underflowed `a*g` cases.
Tiling bounds temporary query/key/feature products; it does not promise that
arbitrary logits, centered dot products or final sums fit the requested dtype.

Ordinary native Attention kernels remain available. With `auto`, any visible
positive prior below `sqrt(finfo(accumulation_dtype).tiny)` selects the explicit
stable tiled path in inference and training. This is an execution threshold,
not a floor: no prior is replaced or dropped. Explicit native requests reject
that range. The bounded admission rule does not protect all possible extreme
logits/cotangents on native paths. BF16 payloads retain FP32 mass arithmetic and
mass-scaled output. FC's proven unit-prior binding retains its existing path.

## Historical origin and remaining boundary

- `169bf675` introduced a centered Aggregate VJP in the retired base.
- `2c11695c` introduced scaled means and fused mass Attention together with an
  explicit Shannon storage policy. Only stable-evaluation ideas are adopted
  here; `positive-floor-v1` is not adopted.
- `d21658c6` captured real downstream pool quotient-backward overflow;
  `d089dd05` repaired that independent implementation's multiplication order.
  Current regression checks reconstruct bounded arithmetic witnesses; retired
  full training checkpoints are not available and are not claimed replayed.

FP32's minimum positive subnormal is `2^-149`, not `finfo.tiny`. Truly
unrepresentable positive senders still fail under the current strict
Shannon/pool contract. Public values/gradients with genuinely excessive range
cannot be repaired by reordering. This work does not qualify new storage
policies, all extreme domains, higher-order derivatives, or task quality.

## Reproduction

In the matching private environment, directed checks are

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests/test_stable_reception.py tests/test_mass_scaled_attention.py
```

The same bounded primitive tests accept `TIDE_STABLE_DEVICE=npu:0` inside an
admitted frozen device job, keeping FP64 oracle calculations on CPU. The
`scripts.qualify_stable_mass` driver reuses the existing training-path checks
for inference, grad-enabled forward, repeated VJP, three continuing AdamW
updates and an actual fresh-process restored update. It records the selected
small diagnostic FFN/KV widths, unchanged thresholds and finite deviations;
these are mechanism cases, not the 9B performance workload. Submit through
`scripts/launch.py` with a fixed revision, new output path and explicit budgets.
