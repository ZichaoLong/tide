# Token-window readout boundary

Python `tidegraph.token_window.token_inputs` and native `tide::token_inputs` are
independent adapters from sealed body outputs to a readout graph's token clock.
They do not invoke either executor. The first use is original LH Pronounce;
qualification status is in STATUS and immutable evidence, not implied by this API.

## Two clocks and complete input

Given L positive body ticks per token, a readout continuation at cut a, stop b,
and sealed body cut b*L, the caller provides **every** selected-port body output
in [a*L,b*L). Tick t maps to token time floor(t/L) and local phase/input port t%L.
The readout graph has one node and L input ports pointing to it. Its existing
Add/attention state modules now advance on tokens, so prior memory decays once
per token, including a sample with no rows via explicit physical cut decoding.

This is an application boundary between two graphs. During generation, run the
body for token k, complete its readout, choose the next token, then seal its input.
No unobserved next-token input is declared absent merely to produce current logits.
A single-PDG clock/phase embedding remains a separate obligation in
`lh-pronounce-plan.md`. The body and readout continuations must both be retained
by a future whole-model adapter; the conversion function owns no hidden buffer.
Calls cover aligned complete token windows, not arbitrary partial body cuts.

External position is a contiguous occurrence counter **per sample and phase**,
seeded from the readout input ledger. It is not token time when a phase is absent.
The adapter sorts by token/sample/phase, rejects duplicate selected-port coordinates,
invalid ranges, overflow, unsealed intervals and malformed accessed ledger entries,
and never mutates its continuation or input tensors. Other output ports are ignored.
The executor still validates graph identity and tensor metadata; the caller must
supply the readout graph whose input domain matches L. A seal is a caller promise
of completeness, not a way to infer omitted outputs from numerical values.

Missing phase/sample rows remain absent. Numerical zero rows remain present.
Original Pronounce rejects a token window empty across all samples; the adapter
rejects that domain explicitly. An empty interval [a,a) is a valid no-op.
Samples absent for an otherwise nonempty token produce no logits row. Public
outputs retain (token,sample) labels; a dense returned Tensor alone loses that
mapping. Token/phase/input-position clocks are checked separately.

## Readout program and training

Use `lh-identity-identity-v1` Full to forward the accumulated observation without
an activation or normalization. This is original Pronounce's default: ALConfig
has no norm option, and `get_norm_type` falls back to identity. Cortex's separate
RMS default must not be borrowed here. `lh-identity-rms-v1` and
`lh-identity-layer-v1` are explicit optional Tide programs, qualified separately
against the normalization factory; they are not the original Pronounce default. Add uses `lh-add-repeat-v1` and
sum Aggregate. Attention uses the separately named same-fiber pooling family:
all phase rows within one token see all new phase K/V, and pool after attention.
No tick-row triangular mask or aggregate-as-token substitution is performed.

The vocabulary linear head is outside the uniform hidden-width graph, allowing
vocabulary size different from hidden width. Oracle/examples retain its weights
in node.extra (`token_head`, `token_head_bias`) so ordinary parameter/checkpoint
records include them. The validation anchor projects each emitted row separately,
preserving isolated-root connectivity. No optimized packed vocabulary head is
claimed. Node attention, Full and graph execution retain their existing batch/
sequence paths and first-order replay contract; LH supplies no training authority.

The direct Python token/phase anchor in `tests/pronounce_cases.py` implements
recurrences and normalization without scheduler or state-kernel calls. Tests
compare generic reference/frontier and native serial/packed/frontier outputs,
complete state, isolated first-row and final-cache VJPs, cuts, detach and saved
head/state restoration. Add/all-softmax also exercise momentum-SGD updates across
a saved optimizer/readout boundary. Adapter tests independently exercise both languages.

## Original-source oracle

`lh-pronounce-check` calls actual `Pronounce::forward(batch, phase_rows, hidden)`
from the immutable LH snapshot, including its CSR phase/sample gather, state,
identity norm and linear vocabulary head. Cases cover Add and five attention pool modes,
L=1/3, four samples with absent samples/phases, real zeros, six tokens, both bias
settings, original single/multi/CROSSBATCH (Attention), and three Tide schedules.
Whole-window and token-cut runs compare labeled logits, normalized output and
complete Add or KV/log-bias state. Head shape is [7,4]; its first four rows expose
pre-head output and the rest exercise a rectangular nontrivial projection.

The existing original FP64 active-softmax multi-batch diagnostic defect also
applies here. Assertions-on explicitly excludes 24 such cases; a separate
assertions-off oracle covers all 204 configurations per precision, with plain
C/C++ assertions still enabled. These are expected domains/counts until a run
passes; actual results belong to qualification evidence. Nothing here proves
whole IOCortexNet wiring, single-graph containment or measured performance.
