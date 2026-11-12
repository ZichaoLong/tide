# Resident KV sequence block

This component implements the pinned core-5 KV reception/update over explicit
`[B,L,N]` logical coordinates. It extends [resident block execution](resident-block-execution.md);
mathematics and raw checkpoint identities remain those of [semantics](semantics.md).

This dense-snapshot component remains an explicit comparison/isolated API.
The default graph local path now uses [entry banks and snapshot views](resident-kv-bank.md)
through `append_block_kv_bank_sequence`, including bank-based Read/Full and reverse.

`append_block_kv_sequence` projects each actual positive reception once, then a
device prefix scan preserves raw initial, old, proposal and final banks. Window
expiry uses the original int64 write/query times. A missing or zero-mass write
keeps raw state and its clocks; capacity overflow is an explicit device failure.
Read old/proposal and bound initial are separate inputs to subsequent modules.

`append_block_kv_sequence_vjp` transposes independent old/proposal/final field
roots into actual initial-bank and reception roots. Structural connection flags
retain unused versus connected-zero semantics even when later expiry removes
all numerical contributions. Saved tapes retain operands and byte ownership.
The declared derivative is first order, with FP32 payload/adjoints and int64
metadata. It is independent of host autograd scheduling.

Implementation: `tools/device_online/block_kv_sequence.{h,cpp}` and
`block_kv_sequence_vjp.cpp`; device scan/transpose: `ascendc/block_kv_sequence_math.h`
and two launch kernels. `tests/test_resident_block_sequence.py` checks metadata
with a separately compiled CPU semantic harness. `block_kv_sequence_check.cpp`
compares actual device projections, raw fields and independent VJPs with literal
local formulas at B2/L3/N2, nonempty initial state and retained reverse execution.

Build the `resident-block-kv-sequence` check with `build_device_control.py`, then
run it with `verify_device_control.py` inside an authorized device allocation.
Building or passing this component does not certify a graph owner, routing,
optimizer, checkpoint, multi-device execution or FC16 throughput. Full graph
composition and its qualifications are separately recorded in ROADMAP/evidence.
The current tape stores per-frame raw snapshots; it does not claim a shared
prefix-bank representation or memory use independent of L.
