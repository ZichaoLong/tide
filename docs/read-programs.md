# Region Read modes and independent readout programs

[Qualified](evidence/read-programs.md). This instantiates upstream
Read modes without changing candidate preparation or state adoption.

`Region.read_mode` is `content`, `old` or `proposal` (default). Every candidate
still runs Upd. Read receives complete content/time and exactly one allowed
state: none in content mode, the old persistent state in old mode, or the proposed
state in proposal mode. State includes named slots, last logical time and accepted
observation count. Empty nodes never run Read. Comparison/Next and Full remain
later events; selecting old-state Read does not select old-state adoption.

`Node.readout` names a program profile. `linear-v1` returns the dot product of
`w.read` and the content summary or mode-selected state value. Identity boundaries
always return zero. The current selector consumes a finite scalar of the payload
dtype; tuple descriptors and independent FP64 score precision are later region
program work, including LH compatibility.

Python custom `ReadProgram` is a registered module supplied by
`Model(read_programs={node: program})`. Native clients supply `ReadKernel` through
`NodeWeights.read_kernel`. Programs receive `ReadInput`; they do not receive both
old and proposed state. Native requests and ContentViews are synchronous borrowed
views. Custom programs must declare a distinct versioned profile when changing
semantics, and remain functional and deterministic under replay. The Python
adapter rejects overrides without a native implementation.

## Packing and differentiation

Read is separate from StateKernel/StateProgram. State prefill first produces and
binds candidate states, retaining each event's correct previous state in both
grad mode and inference. Read then batches over these complete requests. Scalar
fallback is valid and counted; `read_calls`, `read_scalar_batch_steps` and
`semantic_read_replays` expose the distinction. Numeric batch results are bound
to independent event Read graphs in grad mode. Inference never replays Read.

These changes preserve the existing first-order public-root VJP boundary, including
None versus connected-zero. A scalar score with the wrong dtype/device/shape,
a changed batch length or nonfinite values fails explicitly. Read parameters
participate in sharing, optimizer state and checkpoints through module ownership.

Mode/profile are part of graph identity. Native graph identity format is v9;
checkpoint payload remains v3. Old identities require explicit reconstruction,
not implicit migration. See `tests/test_read_modes.py`, `test_read_contract.py`
and the standalone `cpp/test/read_programs.cpp` analytic checks.

Full Next requests and control-sensitive state-prefill gates are implemented
separately in `next-programs.md`, pending qualification.
