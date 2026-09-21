# Full programs and sparse slot emissions, 2026-09-21

Clean source: `fda9a66858f008e481c4969f83a0d1d0bf609700`.
Command: `python scripts/qualify.py --output-dir artifacts/full-20260921-1125`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **1489 passed**, pytest 149.68 seconds.
Retain `artifacts/full-20260921-1125/{status.json,task.log,verification/}`.
The verification manifest records source and all four native binary hashes.
The detached unit completed; no worker remains.

The increment qualifies native/Python Full program interfaces, sparse per-slot
emission with independent slot-affine parameters, phase-dependent presence and
absent-versus-zero delivery. It preserves the previous broadcast profile and
HARD/HST/SOFTP VJPs. Streaming, cursor, frontier, independent specializations and
SettleGraph embeddings compare outputs, traces, pending messages, state, VJPs,
shared modules and optimizer behavior in FP64/FP32, including isolated roots and
disconnected never-used slot parameters. Incompatible native adapter profiles,
subclasses, slot domains and phase policies fail explicitly.

The standalone `tidegraph-full-check` combines custom native State/Full programs:
Full reads pre-clear state slots, clocks and counts; emits only selected slots;
and returns no auxiliary value. Its hand-computed loss is 40, each element of
sample 0's input gradient is 24, and sample 1 remains disconnected. Four scalar
fallback steps are reported. Native execution has no Python callback.

Native graph format is v6; checkpoint payload remains v3 with a changed graph
fingerprint. Packed first-order public-root VJP scope and semantic replay costs
remain those in `../packed-autograd.md`. Custom programs must be deterministic,
functional and batch/scalar equivalent. This report makes no higher-order AD,
LH parity or performance claim. Source-aware Aggregate and full Next/Read and
region programs remain subsequent increments.
