# Next implementation: state and operator programs (M5)

The initial state-program seam and SSM/SwiGLU are implemented and qualified in
`evidence/m5a-state-programs.md`. Matrix-memory profiles are qualified in
`evidence/m5b-matrix-memory.md`.
This document retains the remaining interface/module work; it is not a blanket
support claim for arbitrary models.

M5C adds aggregated-event GQA/window attention and checked packed sequence
interfaces; see `attention.md` and the current qualification in `STATUS.md`.
RoPE/position and LH same-fiber attention remain separate profiles to implement.
Joint EMA/SSM batch/sequence scans are qualified in `evidence/m5d-memory-packing.md`.

## Next vertical gates: general local programs

Stable local ports are [qualified](evidence/local-ports.md). Full/Emit now returns
per-slot payloads or absence, supports native/Python custom programs, and retains
the old broadcast profile; its slot-affine/phase profile is
[qualified](evidence/full-programs.md). Remaining gates:

1. Aggregate interfaces and weighted mean, active-source softmax and all-source
   softmax are implemented with full tags, local slots and optional contributions;
   [qualification](evidence/aggregate-programs.md) covers the built-ins. Graph-owned
   source-origin views now restore tag-sensitive custom Aggregate under SettleGraph
   embedding and are [qualified](evidence/source-origins.md). Built-in memory
   profiles still consume the summary. Complete typed content now reaches
   Upd/Read/Full and packed metadata and is [qualified](evidence/content-programs.md).
   Next receives complete content in the following gate.
2. Full Next inputs (old, comparison, time, content, active, control) and an
   explicit comparison-identity prefill gate are implemented (`next-programs.md`),
   pending qualification. Control-sensitive Next uses causal state preparation
   while retaining Full batching. Independent Read now
   supports content, old-state and proposed-state modes (`read-programs.md`),
   now [qualified](evidence/read-programs.md).
3. Region programs need explicit integer/tensor history and controls beyond
   selection counts. Preserve checkpoint/detach/VJP for tensor history; add LH's
   selection-count/affect-count/FP64-norm/stable-ID ordering as one profile.
4. After these seams, implement LH Add and same-fiber attention with lazy idle
   decay, per-edge signaling and token-window Pronounce. Compare an immutable
   snapshot of actual LH C++ sources, never its older Python interpreter.

These gates instantiate the existing upstream functions; do not change Tide's
Next-before-Full or no-autonomous-empty-event semantics to fit an implementation.
Keep modules and schedulers independently testable, with explicit capability
fallbacks. Mixed-profile training/loss statistics and performance follow the
same regression/evidence process, not a one-time blanket certification.

## First change: remove formula assumptions from schedulers

State formulas now live in Python memory modules and native state kernels behind
`cpp/include/tide/kernel.h`. Scheduling calls these interfaces without Python
callbacks. Full programs live in `full.py`/`full_kernel.cpp`, with FFN/Emit
primitives in `ops.py`/`ops.cpp`. Aggregate programs are separate from scheduling;
Read is independent (`readout.py`/`read.h`); Next is separate (`next.py`/`next.h`); Region generalization remains work.
Keep Python scheduling independent.

- Typed content must preserve source-tagged atoms as well as an optional summary;
  a scalar weighted sum alone cannot represent LH's same-fiber attention.
- State has named tensor slots and per-kernel validation. Keep logical
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
