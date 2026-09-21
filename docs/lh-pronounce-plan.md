# Next LH gate: token-window Pronounce

Source audit: immutable LH snapshot from STATUS. Relevant definitions are
`include/CortexNet.h:Pronounce`, `src/CortexNet.cpp:IOCortexNet::think`,
`src/Adjacency.cpp:gather_signals_info`, and `include/AccumulateLocal.h:BaseAL::forward`.
Pooling has passed `evidence/fiber-pooling.md`; this document describes the next
implementation, not completed Pronounce or whole-model equivalence.

## Audited behavior

`think` runs exactly L=`n_layer` body ticks per input token, retaining `oacts[0]`
from every tick, including null phase entries. Pronounce gathers these L source
positions, transposes phase/sample CSR metadata, updates its own Add or attention
state once, applies normalization without the body's activation, then a linear
vocabulary head. BaseAL decays every sample once per call, even a sample with no
rows. Its attention appends all current token-window rows before queries attend.

A globally empty window fails the original `wholex.defined()` assertion. A sample
with no rows is omitted from the returned dense logits rows; the original Tensor
alone does not retain sample IDs. An adapter must return explicit sample/token
coordinates and preserve phase IDs through missing entries. No synthetic zeros.
The original ALConfig does not expose a norm option; the actual default norm is
RMS with eps=1e-7. Use the original constructor for the first oracle.

## Bounded implementation

1. Add norm-only Full profiles under distinct names; keep existing relu/silu
   names unchanged. Verify analytic values/VJPs and native scalar/batch execution.
2. Add independent Python/native token-window input conversion. Given a complete
   body-output interval [a*L,b*L), a readout continuation at token cut a and sealed
   body cut b*L, map tick t to phase slot t%L and readout event time t//L.
   External positions are **contiguous occurrence counters per sample/phase**
   derived from the continuation ledger, not token indices when phases are absent.
   Validate overflow, duplicates, bounds and the original globally nonempty-token
   domain. Preserve tensor graph/identity and leave the input continuation intact.
3. Execute the readout as a singleton generic graph with L input ports on its
   token clock. Apply the vocabulary head at the application boundary so vocab
   dimension need not equal the graph's hidden width. Qualify any packed head
   path's isolated-root connectivity separately; do not assume stacking is neutral.
4. Build an oracle calling the original `Pronounce::forward(batch, phase_rows,
   hidden)` under no-grad, not merely reimplementing its equations. Compare labeled
   logits, normalized output, Add physical hidden or full KV/bias state, and whole
   versus token-cut runs. Reuse explicit assertion variants for active-softmax.
5. Verify Python/native scalar/packed/frontier execution, a direct fixed-readout
   anchor, backward/None-vs-zero, missing/zero rows, multiple samples, continuation,
   detach and checkpoint. Then clean commit -> qualification -> evidence commit.

## Whole-model and single-graph obligations

First compose a body graph running on LH ticks with a readout graph running on
tokens. In autoregressive use, finish/read out token k before sealing token k+1's
embedding input. This composition must not be described as an already proved
single-graph encoding. `think_single_step` uses previous iacts and oacts on all
four adjacency blocks; signaling can move to prior Full only at fixed weights.

Naively adding delays L-phase makes readout arrive at the next token's input tick,
which cannot be sealed before obtaining that readout during generation. A possible
single-PDG embedding instead reserves a readout phase (period L+1), routes the last
body phase across a two-tick gap, and gives body/readout programs explicit local
clock interpretations. Additional phase edges must map back to original incoming
source domains; treating them as new all-softmax denominator slots changes LH.
Keep pending-message, slot, history and state-clock projections explicit. Prove
this construction separately before making a single-graph containment claim.

The current graph schema need not change for the first component gate. Do not
silently rescale decay or add empty state-update events. Full IOCortexNet still
needs four-block wiring, port ordering, selection, token embedding/readout and
complete continuation projection, including nonuniform signaling weights.
