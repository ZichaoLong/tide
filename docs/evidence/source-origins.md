# Program-visible source origins, 2026-09-21

Clean source: `dd351c9524f8530929ee66e5d151f3176b85f890`.
Command: `python scripts/qualify.py --output-dir artifacts/origin-20260921-1222`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **1740 passed**, pytest 156.81 seconds.
Retain `artifacts/origin-20260921-1222/{status.json,task.log,verification/}`.
The verification manifest includes C++ source and all native binary hashes.
The unit is inactive and no worker remains.

The prior Aggregate qualification did not cover tag-sensitive custom programs
under SettleGraph embedding. A reproducer that uses kind and origin position
produced direct content `[0.8, 1.6]` versus encoded `[1.6, 4.0]`. Graph-owned
InputOrigin views now restore source kind, port and position while preserving
physical routing, scale lookup and local slots. Sorting after projection restores
the canonical order of mixed boundary/internal fibers.

The Python regression uses a registered custom gain and an order-sensitive fold;
direct and encoded SettleGraph agree on full traces, values, state and isolated
output/contribution VJPs. Malformed/duplicate mappings, graph identity, checkpoint
rejection before weight changes and off-lattice send times are checked.

The standalone native check compares direct external inputs against boundary-edge
encoding with a self-loop, exercising mixed fibers and canonical program order.
Both match the hand-computed loss 1004, per-element input gradients 372 and 30,
and gain gradient 862. The origin index is absent when the graph has no view;
those graphs also skip the extra source sort. All prior Aggregate, serial/
parallel/packed, inference, cursor, SettleGraph and isolated-VJP tests pass.

Native graph format is v8; checkpoint payload stays v3 with a new identity.
Origins currently affect Aggregate's program input. Propagating this complete
content to custom state/Read/Next/Full programs is the next interface gate.
This result establishes no original-LH parity or large-workload performance.
