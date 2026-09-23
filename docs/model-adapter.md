# Small explicit model adapter

`TinyDecoderAdapter` in `python/tidegraph/model_adapter.py` is a finite PyTorch
module-level composition using Tide's event Attention/GQA and SwiGLU weight
layout. Named profile: `tiny-rms-rope-gqa-swiglu-v1`. It composes pre-RMSNorm,
interleaved-pair RoPE on Q/K, causal GQA, residual projection, pre-RMSNorm,
SwiGLU and residual. Default D8/Hq2/Hkv1 has deterministic generated weights.
No pretrained import or arbitrary model compatibility is promised. This example
is not a replacement for either graph Attention profile or a new C++ backend.

Inputs are BTD or explicitly selected TBD. Positions, occurrences and boolean
validity are always separate [B,T] tensors. Valid positions are nonnegative and
strictly increasing, may have gaps, and never come from logical time. Valid
occurrences must continue each sample's complete ledger; padded rows create no
KV/occurrence entries. Causality and optional window use accepted event order,
while RoPE uses the supplied positions. Invalid/padded output rows are connected
zero. The real nontrivial sequence matmul reports calls and max_sequence.

Cache owns cloned K/V/position suffixes and the complete occurrence ledger.
Continuation is functional; detach clones and severs previous-window gradients.
This example's cache is restricted to the same live parameter objects/versions.
An optimizer update or different adapter rejects the stale cache; reset and
prefill explicitly. It is not graph continuation v5, TIDENCK1, or an application
checkpoint and promises no serialized resume. Graph checkpoint scope is intact.

`tests/test_model_adapter.py` uses independent scalar token/head formulas for
norm, RoPE, softmax, GQA, SwiGLU, all cache values and output/cache VJP roots,
None/connected-zero, ragged/empty masks, irregular explicit positions, chunk
cuts, detached continuation, owner/update rejection and metadata errors.

Physical same-fiber projection layout is separately selected at `Model`
construction with `projection_layout="input"|"linear"`. Linear uses a transposed
contiguous backing for QKV/output matrices while preserving logical shapes and
parameter ownership after construction. It is not an executor algorithm or
ProjectionEmit option. Nondefault layout without same-fiber nodes rejects.
`test_projection_layout.py` compares streaming/frontier/chain/encoded Settle,
shared owners, full traces/VJPs, three AdamW updates, stable pointers/strides and
graph checkpoint values. Model construction precedes binding and optimizer setup.
