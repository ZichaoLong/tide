# Device post-attention pooling — 2026-09-30

Clean `691cb31` passes a full standalone build, four CPU CTests, all 26 device
component cells and a separate CANN placement profile. The
[manifest](device-fiber-pool-20260930.json) records exact source/core/component
hashes, raw result hashes and the retained fixture failure. This qualifies the
five existing `lh-fiber-attention-*-repeat-v1` profiles for single-device FP32
HARD inference, extending the earlier [sum attention evidence](device-fiber-20260930.md).

Sum, mean, linear, active-softmax and all-softmax share unchanged QKV/cache
semantics. Coefficients act on completed query outputs before projection and
its once-per-event bias. Device loops pack actual softmax events and determine
source coefficient placement. Mean and active-softmax use present logical
sources; all-softmax also includes absent logical slots. Physical aliases do
not enlarge the domain. A zero source or zero coefficient retains its Q/K/V.
Pooling reservations participate in the bounded query-chunk budget.

The new gate passes 80 independent scalar/alias/cache anchors, 18 wide-domain,
extreme-logit and empty-domain cases, 160 complete windows and 5 refusals. It
covers widths 1/7/33/257, heads 1/3, chunk sizes 1/4, negative/zero weights,
missing versus present-zero messages, 257-slot domains, missing dominant logits,
old KV, mixed pooling profiles, all three Read modes, feedback/DAG, selected-only
adoption, clear, large int64 clocks/counters, InputOrigin, restore and lean
continuation. Complete tensors retain rtol1e-5/atol1e-6; discrete observables
remain exact. An earlier fixture used illegal clock phases; its original failure
is retained and the valid periodic cases in the existing fiber gate still run.

The trace records 76,953 AIV and 1,586 AI Core tasks, including the pooling
kernel, batched projections, gather, transpose and softmax. No AiCPU task or
host fallback diagnostic was observed. Setup, export and CPU assertions are
included: these counts establish placement, not an end-to-end speed comparison.

Attention still admits one complete region-time frame per device stage, with
independent owners/message rows packed. Full node-time attention batching,
key-axis tiling, event-GQA/window, FP16 flow, public presets/matrix, multi-card
progression and resident backward/optimizer remain separate obligations.
