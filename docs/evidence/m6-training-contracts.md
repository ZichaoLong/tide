# Training identity and execution records, 2026-09-21

Clean tested source: `a2d833eade4c30034706db38f7d805397e35d5f0`.
Command: `python scripts/qualify.py --output-dir artifacts/contracts-20260921-0840`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release; two build workers, one ATen/BLAS thread, up to three node workers.
Build/qualification exit 0: **323 passed**, pytest 26.98 seconds.
Retain `artifacts/contracts-20260921-0840/{status.json,task.log,verification/}`.
The verification manifest includes the C++ source fingerprint, module/executable
hashes and exact build source identity, and validates these before testing.

New coverage: shared node/edge parameter VJPs; checkpoint v2 rejects a different
alias topology before mutating weights; restored sharing is preserved; explicit
cut detach removes both state and in-flight-message input gradients while
preserving forward values; inference-mode state reaches native workers; six
seeded sparse cyclic/ragged-batch cases per dtype; input-position holes rejected.
All previous streaming, frontier, SettleGraph, specialization and CLI checks pass.

Durable cancellation was separately exercised on an owned 30-second sleep unit
at this clean commit. `systemctl --user stop` ended the control group; persistent
status recorded `cancelled`, exit 143, and finish time. Retain
`artifacts/cancel-check-20260921-0840/status.json`. The unit's failed marker was
reset after verifying the expected cancellation; no worker remains.

Static portability audit: 0 errors, 9 reviewed warnings. Seven are inactive
CUDA branches in the copied CPU CLI adapter; the other two are the declared
Torch dependency range and explicit device vocabulary. No accelerator support
is claimed. Artifact cleanup dry run found no eligible old unreferenced jobs.

Limits: this is operator/executor training equivalence for the declared profile,
not training quality. Arbitrary loss statistics, general node/region programs,
advanced memory slots and LH parity remain future work. Value checkpointing does
not serialize live autograd, a training controller, data cursor or RNG state.
