# Next implementation: state and operator programs (M5)

The initial state-program seam and SSM/SwiGLU are implemented and qualified in
`evidence/m5a-state-programs.md`. Matrix-memory profiles are under qualification.
This document retains the remaining interface/module work; it is not a blanket
support claim for arbitrary models.

## First change: remove formula assumptions from schedulers

State formulas now live in Python memory modules and native state kernels behind
`cpp/include/tide/kernel.h`. Scheduling calls these interfaces without Python
callbacks. Full formulas remain in `ops.py`/`ops.cpp`; region/Agg/Full program
generalization remains work. Keep Python scheduling independent.

- Typed content must preserve source-tagged atoms as well as an optional summary;
  a scalar weighted sum alone cannot represent LH's same-fiber attention.
- Extend State with named tensor slots and explicit slot schema. Keep logical
  time and observation count separate from memory; Full consumes the comparison
  snapshot, Next decides every persistent slot before Full.
- Node programs supply initial state, Agg/Upd/Read/Next and Full/Emit. Region
  programs eventually generalize count history and the current top-k selector.
- State kernels expose step, packed independent-sample step, and optional exact
  sequence-block contracts. A sequential fallback must be visible in statistics.
- The packed representation needs values, sequence/sample/node IDs, prefix
  offsets, source tags and event coordinates. Padding cannot create candidates.
- Preserve old EMA/identity behavior as compatibility profiles. Re-run existing
  tests before adding advanced profiles; update checkpoint schema deliberately.

## Concrete module sequence and gates

| Module | State and explicit contract | Independent checks |
| --- | --- | --- |
| Attention/GQA/window | Ragged KV slots; query/KV heads; cache/window policy; clock for positions; causal between events; separate same-fiber policy | explicit masked attention vs packed grouped implementation; forward/VJP/cache; chunk and eviction |
| Linear attention | Positive-feature key/value memory and normalizer; declared epsilon and heads | literal recurrence vs additive scan; initial-memory VJP |
| DeltaRule/Gated Delta | Matrix memory; key/query normalization, decay and beta gates; fixed exact update order | step recurrence vs affine-transform scan; reset and final-memory roots |
| Diagonal selective SSM | Recurrent state, discretization and input-dependent gates declared exactly | literal recurrence vs affine prefix scan; elapsed-time policy |
| SwiGLU FFN | Gate/up/down projections, residual and normalization placement | direct formula and gradients; selected-only execution |
| Agg/Emit | Source-aware weighted sum remains anchor; add declared normalized/attention variants; retain HARD/HST/SOFTP | analytic VJP, shared sources, absent vs zero messages |

These are representative profiles, not claims to reproduce every open-weight
architecture. Model imports need separate parameter/layout, RoPE, convolution,
normalization and mask adapters with model-specific equivalence tests.

For CPU packed attention, start with ragged storage and batch equal/bucketed
length segments using ordinary ATen, with a simple per-segment oracle. Do not
materialize one cross-sample quadratic attention matrix just to call it packed.
Measure allocation/work/shape counters before claiming speed. SSM/DeltaRule
scan algebra does not by itself prove a performance improvement.

## Training and continuation gates

Compare every state slot, not only the read summary, in trace/checkpoint/VJP.
Full blocks may batch across time only when state/selector controls permit.
Clear/selected-only state adoption use a declared joint contract or causal
fallback. Test shared parameters, independently differentiable initial memory,
pending roots and inactive samples. Parameter gradient presence must still agree.

Position clocks require an explicit choice. A general positive-delay fiber can
mix originating token positions, so logical time cannot silently become a token
position. SettleGraph supplies its `stride*position+rank` mapping; LH has both
tick and token-readout clocks.
