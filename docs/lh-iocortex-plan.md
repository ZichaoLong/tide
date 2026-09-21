# IOCortexNet mapping and remaining containment work

Component gates through Pronounce are qualified in `evidence/pronounce.md`.
The whole-model adapter in `lh-iocortex.md` is qualified in `evidence/lh-iocortex.md`.
The single-graph construction below is implemented in the oracle layer;
`lh-single-graph.md` defines its bounded projection and STATUS tracks qualification.
Call actual unchanged IOCortexNet::think and think_single_step from the hashed
snapshot; component parity alone cannot certify whole-model execution.

## Explicit mapping

- Combine both cortexes into one body PositiveDelayGraph. All four adjacency
  blocks read previous-tick activations, including the input-to-output bridge.
  Wires have unit delay; at fixed inference weights signaling can move from
  original next-tick gather into current-event Full.
- Incoming local order: intra CSC, bridge CSC, token slot (inet node zero only).
  Declare PortLayout explicitly. On non-token ticks, all-softmax's declared
  domain still includes the absent token slot even when LH omits its placeholder.
- Slice signaling weights/biases by each original CSR row's local output order.
  Four modules have independent weights; preserve physical edge identities.
- Each base hub gets its own count/affect region; other hubs use full-capacity
  regions. Input/output histories remain separate.
- First gate: equal cortex widths, Add/same-fiber attention, supported act/norm
  and linear signaling. Unequal original cortex widths are outside this gate.
- Compose with token-window readout and keep both continuations. Generation
  completes token k readout before sealing token k+1 embedding input.

## Oracle and acceptance

Build original CortexNet.cpp unchanged. Wrap BaseSelector and delegate to
NaiveSelector (whose select is final); clone candidates before selection and
retain selected BatchSignals to observe post-selection activation. Inspect
hidden/counters without patching LH. Use actual think for token runs and actual
think_single_step for ragged/manual ticks.

Compare candidate/proposal/FP64 descriptor, selection, activation, physical hidden
or complete KV/log-bias, selector counters, token outputs and pending messages.
Pending messages must equal the four original signaling transforms of final
iacts/oacts, with exact edge/sample/send/arrival coordinates. Include idle and
cleared samples, token cuts versus whole teacher-forced windows, serial versus
node-parallel/packed CPU FP64/FP32. Keep original assertion variants explicit.
Add independent Python fixtures/schedules for cross-language whole-model coverage.

Whole-model descriptor checks distinguish accumulation from upstream precision:
each Read equals the FP64 norm of its own candidate under the FP64 tolerance;
the two norms differ by at most the actual candidate L2 perturbation plus FP64
roundoff. Candidates themselves still use their payload-dtype tolerance, and
routes must match exactly. FP32 payloads do not become FP64-accurate by promotion.

## Separate obligations

Two-clock composition is not yet a single-PDG proof. A reserved readout phase
(period L+1) may resolve autoregressive sealing, but the last body phase then
needs a two-tick edge gap and explicit local state clocks. Replicated phase
edges must map to original source domains so all-softmax denominators remain
unchanged. Prove clocks, idle decay and pending projection separately.

Define composite checkpoint and cross-graph parameter/optimizer aliases before
claiming whole-model save/resume or training. LH is an inference oracle only.
Persistent cache/history and scale/performance remain M8 obligations.

## Single-graph construction to qualify separately

Let D=L+1. Map original body tick kL+p (0<=p<L) to global kD+p;
reserve global phase L for token readout. A body wire uses delay one except
when its sender phase is L-1, where it uses delay two. Fixed physical edges
with phase-selective Full can implement these alternatives. Body output phase p
uses delay L-p to reach the readout at kD+L. Readout completes before the next
external input at (k+1)D can be sealed, including greedy generation.

For a complete global cut c, the body cut projection is
floor(c/D)*L+min(c%D,L); the readout cut is floor(c/D). At body events convert
stored last_time and event time into body ticks before applying repeated decay;
at readout events convert them into token ticks. Return stored metadata to the
global clock. Resets preserve that metadata. Arbitrary cuts must project pending
body wires plus partial token-window output buffers, not only token boundaries.

Physical phase wires need a graph-owned logical source domain: all alternatives
of an original wire share its source slot and coefficient. `source-domains.md`
implements this layer separately from bijective PortLayout and is qualified in
`evidence/source-domains.md`. Duplicating all-softmax coefficients over
physical aliases changes normalization even if only one alias arrives. Do not
patch this by scaling arbitrary softmax weights. Qualify an explicit domain map,
parameter aliasing, cache order and duplicate-arrival rejection. `state-clocks.md`
implements the clock wrapper and delegated step/block contracts; treating the
reserved phase as an extra idle decay changes LH. Whole-model complete-cut
projection is implemented separately; these primitives alone are not a containment proof.
Keep original valid-input restrictions (globally
nonempty readout windows), fixed inference weights and equal-width scope explicit.

Original LH phase/sample CSR has no External.position. The two-graph token_inputs
adapter adds a contiguous occurrence ledger per sample/phase. If some phases are
absent, send_time / period gives a token index, not the occurrence position.
Projection to the complete two-graph Tide continuation therefore needs explicit
phase occurrence counters with typed state/checkpoint rules. Alternatively a
narrower projection to original LH may omit that adapter-only ledger, but must
not claim complete two-graph continuation equality. Choose and document the
projection boundary before implementing the single-graph oracle.
