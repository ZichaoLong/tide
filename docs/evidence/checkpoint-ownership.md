# Named optimizer ownership qualification, 2026-09-22

Clean tested source: `c84abbed7c9cfce6d4c6a44f6b7ae5f9aab465d3`.
Implementation commits: `f900e15` (ownership), `a558b87` (single-PDG oracle),
`c84abbe` (explicit gate scopes). The CPU verification stage passed **3946 tests**
in **326.58 seconds**, exit 0, with an empty dirty-state record. Graph v13 /
checkpoint v5. Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu,
C++17/C++11 ABI, two build workers and one ATen/BLAS thread.

Evidence: `artifacts/single-qualified-20260921-2234/verification/{result.json,tests.log}`.
The isolated source is `/var/tmp/zlong-graph-execution-foundation/qualification-20260921-2234`;
its tracked files are read-only and its build is independent. Verification ran
`python scripts/verify.py --device cpu --dtype both --output-dir .../verification`
as a stage of `scripts/qualify.py --lh-snapshot ...`. It finished at
2026-09-21T22:43:32Z. The enclosing original-LH qualification is still running
when this report is written; this report certifies the completed CPU stage only.

The v4 reproducer saved distinct SGD momentum 1 and 7 for two scalar owners.
Reconstructing the optimizer in reversed parameter order silently restored 7
and 1. The repair stores canonical parameter names per group plus optimizer class,
rejects ownership/order/alias mismatch before mutation, and validates shared
weight agreement and ordinary optimizer tensor state. It preflights standard
optimizer loading on a temporary copy. Old checkpoint schemas are rejected.

72 new FP64/FP32 cases cover SGD/Adam/AdamW, same-shaped reordering, different
groups/classes, foreign or repeated aliases, corrupt state IDs/shapes/dtypes/NaNs/
slots, conflicting shared values and shared-subset next-update equivalence.
Existing graph/module checkpoint, training, gradient-connectivity and continuation
tests also pass. 126 new single-PDG and three gate-scope tests are in the same
CPU run, but original-LH clean numerical evidence is a separate pending result.

Limits: one graph/model value checkpoint; no full application controller, RNG,
data cursor, live autograd, cross-graph optimizer or independent standalone C++
checkpoint format. Arbitrary custom hooks with external side effects have no
transactional guarantee. Temporary preflight copying has a checkpoint-time memory
cost. See `../checkpoint-ownership.md` for the contract.
