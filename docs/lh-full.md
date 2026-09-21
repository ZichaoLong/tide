# LH post-selection Full and edge signaling

Six explicit `Node.full` profiles combine `lh-{relu|silu}-{identity|rms|layer}-v1`.
They compute activation of **comparison.value**, followed by normalization along
the last, width-sized dimension. They add no FFN or content residual. RMSNorm
has learned `extra['lh_norm_weight']` and epsilon 1e-7; LayerNorm has learned
`lh_norm_weight`, `lh_norm_bias` and epsilon 1e-5. These are original LH defaults.
Non-default epsilon and heterogeneous payload widths are not covered by these
profiles. Profile names already participate in graph identity.

Use existing `emission='slot_affine'` for per-edge signaling. For hard Emit,
each output slot receives `normalized_activation @ emit_w_slot + emit_b_slot`.
LH's concatenated linear output `[out_degree*width,width]` is mapped by taking
one row block per original CSR output position and transposing it. Map physical
edges to those local slots explicitly. A bias-free imported linear uses a zero
bias; freeze/remove training ownership as appropriate for its declared model.
Identity signaling on a compatible one-output node can use broadcast.

The same normalized activation is available as auxiliary Full value. Next and
selected clear already ran; Full reads the retained comparison snapshot and
never changes hidden state. Positive-delay edge delivery still preserves source
identity and adds the edge delay. Missing and zero payloads remain distinct.

HARD matches this original inference component. SOFTP/HST remain Tide's declared
training profiles, using the existing content-based held value, projected held
value per slot, and control VJPs. Under HARD, a Full-only root has no gradient
to held content or unused tanh/FFN parameters; HST supplies a connected-zero
held-content cotangent. LH supplies no training authority.

Python and native implementations support selected batch/time rows; packed
public-root training retains semantic replay. Shared normalization parameters
can span nodes with different graph-owned output domains while their signaling
projections remain separate. Checkpoints preserve that alias topology.

`tests/test_lh_full_formulas.py` checks values and independently derived derivatives
for activation and RMS/Layer normalization, HST/SOFTP, zero variance and malformed
parameters. Schedule tests cover Add/clear, source projections, competing nodes,
ragged samples, output/pending/state roots, serial/parallel, frontier, SettleGraph
encoding/specialization, cuts/detach and shared-parameter AdamW/checkpoints.

The optional original-source runner accepts `--component full` or `all`.
`lh-full-check` calls the actual snapshotted ModuleUtils factories, with both
signaling bias choices, widths 1/3/5 and output degrees 1/3, comparing scalar
and packed Full values and every output slot. Clean qualification is in
[evidence/lh-full.md](evidence/lh-full.md).
This establishes the post-selection component only; original IOCortexNet,
same-fiber attention and token-window Pronounce remain separate gates.
