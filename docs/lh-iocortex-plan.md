# Whole IOCortexNet inference gate

Component gates through Pronounce are qualified in `evidence/pronounce.md`.
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

## Separate obligations

Two-clock composition is not yet a single-PDG proof. A reserved readout phase
(period L+1) may resolve autoregressive sealing, but the last body phase then
needs a two-tick edge gap and explicit local state clocks. Replicated phase
edges must map to original source domains so all-softmax denominators remain
unchanged. Prove clocks, idle decay and pending projection separately.

Define composite checkpoint and cross-graph parameter/optimizer aliases before
claiming whole-model save/resume or training. LH is an inference oracle only.
Persistent cache/history and scale/performance remain M8 obligations.
