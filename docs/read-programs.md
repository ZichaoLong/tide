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
always return zero. Selectors consume finite scalar descriptors under an explicit
payload/FP64 precision policy. `norm-fp64-v1` accumulates the L2 norm in FP64;
tuple descriptors remain outside the current interface.

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

Mode/profile are part of graph identity. Current graph/checkpoint versions are
owned by [semantics.md](semantics.md). Old identities require explicit reconstruction,
not implicit migration. See `tests/test_read_modes.py`, `test_read_contract.py`
and the standalone `cpp/test/read_programs.cpp` analytic checks.

Full Next requests and control-sensitive state-prefill gates are implemented
and [qualified](evidence/next-programs.md) separately in `next-programs.md`.

Explicit Read precision is `payload | float32 | float64`; `norm-fp64-v1` accumulates
the L2 norm in FP64. Payload dtype remains independent of descriptor precision;
see `lh-selector.md` for controls, histories and the original-selector oracle.

`norm-fp32-v1` explicitly converts the visible value to FP32 before L2 reduction
and returns an FP32 descriptor, including for FP16 or FP64 payloads. Its VJP is
the ordinary conversion/norm VJP, including connected-zero input gradients at a
zero norm. The unused linear Read parameter remains disconnected. Region controls
are converted back to payload dtype by the existing selector contract. The original
`norm-fp64-v1` formula and rejection of FP64 computation on NPU remain unchanged. Profile choice is recorded
in graph identity; rounded ties may select different nodes across precision profiles.

The optional [placement adapter](execution-placement.md) declares descriptor
device separately from payload device. Default Read returns a scalar on the
payload device; validation checks the kernel's declared device rather than
accepting arbitrary returned placement. CPU FP64 Read/control/ranking with NPU
payloads is an explicit mixed configuration. Copies preserve the VJP. Placement
and explicit linear-scoring precision are recorded execution choices, outside
graph/checkpoint identity; named norm precision cannot be silently overridden.
Each execution is compared against its independently scheduled matching profile.
