# IOCortexNet mapping and remaining containment work

Component gates through Pronounce are qualified in `evidence/pronounce.md`.
The whole-model adapter in `lh-iocortex.md` is qualified in `evidence/lh-iocortex.md`.
The single-graph oracle construction is qualified in `evidence/lh-single-graph.md`;
`lh-single-graph.md` is the canonical construction/projection contract.
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

## Remaining obligations

The bounded single-PDG projection is qualified; do not treat its primitives
alone as proof for other configurations. Keep original valid-input restrictions
(globally nonempty original Pronounce windows), fixed inference weights and
equal-width homogeneous profiles explicit. Phase replication has O(L*E) cost.
Unequal widths, arbitrary mixed profiles and full pretrained/configuration import
need separate mappings and oracles. Scale/performance remain M8 obligations.

Tide two-clock/single-PDG training and single-graph value resume are qualified in
`evidence/single-graph-training.md`; this uses Tide's declared VJPs and truncated
updates. No training equivalence to old LH is claimed. The composite two-clock application now has named cross-graph ownership and one
value checkpoint containing both continuations, the application cut and unfinished
readout buffer; see `token-application-checkpoint.md` and STATUS for qualification.
RNG/data cursors and an arbitrary training controller remain caller-owned.
Single-PDG v5 resume does not serialize that separate application.

Original LH phase/sample CSR has no External.position. The two-graph token_inputs
adapter adds a contiguous occurrence ledger per sample/phase. With missing phases,
last send time divided by period gives a token index, not an occurrence position.
The qualified readout view therefore omits this adapter-only ledger explicitly.
If full two-clock continuation reconstruction is needed, introduce validated
occurrence counters and persistence rather than fabricating them from time.
Standalone C++ optimizer ownership/serialization remains a separate deliverable.
