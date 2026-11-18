# Reception self-Attention execution

The mathematical choices remain explicit `ModelConfig.aggregation` values.
`source_residual` transforms each incoming atom before the final weighted mean;
`incoming_light_64`, `incoming_light_128`, and `incoming_wide` additionally mix
incoming atoms with weighted self-Attention. Selecting a different aggregation
changes the model and its parameter/work inventory. Execution candidates below
preserve the selected aggregation rather than choosing one automatically.

For one receiver, sample and logical position, let incoming content be
\(g_j\), nonnegative mass \(c_j\), and source class \(s_j\). The incoming
profiles compute, independently for each head of width 32:

\[
x_j=\operatorname{Norm}(g_j)+e^{SA}_{s_j},\qquad
a_j=\frac{\sum_k c_k\exp(\langle Qx_j,Kx_k\rangle/\sqrt{32})Vx_k}
           {\sum_k c_k\exp(\langle Qx_j,Kx_k\rangle/\sqrt{32})},
\]

\[
y_j=g_j+O\operatorname{ConcatHeads}(a_j),\quad
v_j=y_j+B\operatorname{SiLU}(A\operatorname{Norm}(y_j)+e_{s_j}),\quad
\operatorname{Agg}=\frac{\sum_j c_jv_j}{\sum_j c_j}.
\]

Zero denominators use the declared zero output convention. Negative raw masses
are clipped at zero. The existing weighted-Attention derivative convention at
zero remains in force; this does not assert a general normalized-content
directional derivative at an entirely empty fiber. `source_residual` uses the
same source transform/final mean with \(y_j=g_j\). Light profiles have Attention
width 64 or 128 and MLP width `d//8`; wide uses width `d` for both.

## Explicit execution choices

| Option | Mechanism and boundary |
|---|---|
| `reception_qkv_backend="separate"` | Existing separate projections; default |
| `reception_qkv_backend="fused"` | One compatible base contraction for concatenated Q/K/V; independent adapters and original Parameter names/owners retained |
| `reception_mask_padding=True` | A cached graph-membership mask excludes only nonexistent edges; default false |
| `checkpoint_reception=True` | Recompute Reception during backward; whole-region recomputation takes precedence |

`model.receive.effective_qkv_backend` reports `separate`, `fused`, `none` for
source-residual, or `separate_owner_wrapper` for the explicit placed fallback.
Projection fusion currently concatenates weights on each call; it is a candidate
with a dispatch/copy tradeoff, not a promise of faster decode.

The structural mask is built from routing metadata, independent of current
sender strength. All real edges, including currently zero-mass ones, remain in
the derivative domain. Only nonexistent padded slots are removed. In training,
this can admit native weighted Attention when all real priors are positive;
real visible zero priors still require the zero-prior-compatible path. The
mask does not physically compact query/projection rows, so their padded work
remains counted. Compatible node/region batching remains independently useful.

## Reproduction boundaries

The default training entry exposes the options without editing a preset:

```sh
python examples/train.py --config configs/continuous/hb-settle-small.json \
  --aggregation incoming_light_64 --reception-qkv-backend fused \
  --reception-mask-padding --checkpoint-reception --precision fp32 \
  --steps 3 --checkpoint NEW.pt
```

Compare full Reception after normal input preparation with:

```sh
python -m benchmarks.reception --device cpu \
  --profiles incoming_light_64 incoming_light_128 incoming_wide \
  --cases padded zero --precisions fp32 --modes inference training \
  --candidates separate fused masked fused_masked --backends auto math tiled \
  --nodes 8 --batch 8 --length 128 --edges 8 --d 128 \
  --warmup 2 --steps 5 --threads 4 --output NEW-reception.json
```

For an admitted NPU use its explicit device. Run a second new output with
`--checkpoint-reception` to measure the memory/throughput tradeoff at identical
dimensions and initial parameter hashes. `--modes grad_forward forward_backward`
separates graph construction from reverse computation; only `training` includes
AdamW. Shared and independent parameter ownership are selected with `--sharing`.
The denominator is receiver-node/sample/position count, not graph input tokens.
Whole-model first-output/decode and A–D comparisons remain separate benchmarks.

Every candidate starts from the same FP32 parameter/input fixture. A route mask
already removes padded mass roots on all candidates, so numerical comparisons
retain only real incoming-edge derivatives. The optional structural Attention
mask then changes execution without redefining that domain. Normal samples run
without audit hooks. A separate call records actual Attention/projection paths,
operators/dtypes, full gradients and one further update in training mode. JSON
retains raw timings, memory, fixture hashes and original numerical failures;
nonzero exit status includes finite threshold deviations, not only crashes.

For integrated fixed-state inference, repeated VJPs, multiple updates and an
actual new-process restored update, set the desired incoming profile in a
`ModelConfig` JSON and use the existing path checker:

```sh
python -m scripts.check_training_paths --config INCOMING-CONFIG.json \
  --device cpu --precision fp32 --batch 1 --length 4 --chunk 2 --steps 3 \
  --candidates baseline qkv_fused padding_mask both \
  --candidate-overrides '{"qkv_fused":{"reception_qkv_backend":"fused"},"padding_mask":{"reception_mask_padding":true},"both":{"reception_qkv_backend":"fused","reception_mask_padding":true}}' \
  --output-dir NEW-paths
```

The CPU tests include independent literal formulas and every input/parameter
VJP, genuine zero-prior directional derivatives, FP64 multi-update equivalence,
FP32/BF16 real restore/update checks, and inventory/placed-fallback checks.
These scopes are distinct. The [FP32 Reception path evidence](evidence/reception-numerics-20261118.md)
covers all three incoming profiles on a compact-9 Settle NPU fixture: inference,
grad-enabled forward, repeated VJPs, three continuing updates and nine actual
new-process restored updates. Eight of nine candidates retain `finite_deviation`
under the original thresholds; no recorded support, integer-time or gradient
connectivity difference was observed. All nine candidate-local restored updates
are within threshold, which does not erase cross-candidate training deviations.
The internal NPU witness hit its capture bound, so an earliest causal divergence
was not established. CPU formula and bounded AdamW/thread witnesses support their
own local conclusions only. Performance recommendations retain their separate
source and workload scope. Long experiments follow [execution](execution-flows.md).
