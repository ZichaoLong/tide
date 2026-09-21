# Linear Attention and gated DeltaRule, 2026-09-21

Clean source: `ca0a12764ab0f766ba86f46c84a6d60f2e35e303`.
Command: `python scripts/qualify.py --output-dir artifacts/m5b-20260921-0920`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **619 passed**, pytest 45.86 seconds.
Retain `artifacts/m5b-20260921-0920/{status.json,task.log,verification/}`.
The manifest records source fingerprint and hashes of all three native binaries.

New FP64/FP32 coverage: Linear Attention and gated DeltaRule step/sequence
equivalence, packed/unpacked and serial/parallel cyclic streaming, all Emit
modes, clear and selected-only adoption, input/parameter/initial-slot VJPs,
checkpoint and analytic formulas. Mixed SSM/Linear/Delta with SwiGLU also checks
direct SettleGraph against its TimedDAG embedding. Existing 481 tests pass.

These are single-head representative equations specified in `../matrix-memory.md`.
The dense affine Delta scan is an algebraic correctness anchor and may cost
more than stepping; no performance claim follows. GQA/window attention,
optimized Delta chunk kernels, original-LH inference comparison, and scale
measurements remain pending. This report does not certify arbitrary model imports.
