# Stable local ports, 2026-09-21

Clean source: `df2a851d759da9d6e5ccd7cef9ba931e416fdf07`.
Command: `python scripts/qualify.py --output-dir artifacts/ports-20260921-1047`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **1296 passed**, pytest 103.84 seconds.
Retain `artifacts/ports-20260921-1047/{status.json,task.log,verification/}`.
The verification manifest records native source and binary fingerprints.

The 29 added cases check explicit local-slot permutations, physical parallel-edge
identity, independently expected flat inverse rows, default Python/native mapping
agreement, invalid layouts, structural fingerprints and checkpoint/continuation
rejection. A rejected checkpoint leaves model weights unchanged. SettleGraph
cases preserve body-local slots and shared node parameters while converting
boundary ports to edges, comparing FP64/FP32 forward traces and VJPs.

The mixed-input case also covers a node receiving both external and internal
messages at the same event: projection restores canonical fiber order after
restoring external tags. All prior isolated-autograd and executor tests pass.

This is indexing/identity/embedding qualification. Slot-aware Full/Emit and
source-aware Aggregate are subsequent program increments. No throughput,
memory-at-scale or LH numerical claim follows from these tests. Native structural
format is v5; checkpoints with old/different graph fingerprints are rejected.
