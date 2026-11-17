# Current handoff — FP32 instance delivery complete, 2026-11-18

Branch: graph-execution-foundation. Fixed math:
4f5b9f20a4030d68ee66ae2d26e34659a5e37234 (108 snapshot files verified).
Model implementation: c17c613f178dd076d963da98017abdea8c09db16.
Later diagnostic/example commits: d2090654 (FP64 diagnosis), a2c1e08b (packing
counts), cc2b5767 (actual resumed optimizer update). Model source is unchanged.
Current results and exact source/commands: docs/evidence/instances-fp32-20261118.md
and its JSON. Start navigation at docs/README.md; API at docs/instances.md.

Delivered fresh FC, Lattice, Exchange, Multiscale Settle/Timed; K/S/X, four
reception profiles, quad/Shannon/pool, three sharing profiles, continuation,
flush and optimizer/checkpoint. FP32 only; optional CPU FP64 accuracy checks.
No retired implementation/tests, old teaching/LH-DAG/PDG, historical backend
matrix or P1–P6 obligations. All source work is complete for this finite delivery.

Frozen CPU: 308 tests passed 32.39s; separate installed consumer performed 3 AdamW
updates and exact next-window replay. NPU Attention 10/10 passed. Six whole-model
checks passed structural/gradient-connectivity/finite-update/exact-checkpoint
requirements:2 within all numerical thresholds,4 finite deviations retained.
One bounded FP64 replay supports rounding sensitivity in the worst case; both
FP32 variants' gradients pass against FP64, with identical support. No threshold
was weakened. The report does not call all six strict numerical checks passed.
An installed FC-K example additionally executed a resumed AdamW update on NPU.
Saved data/forward remain exact; CPU/NPU FP32 updates retain finite common-bias
parameter deviations (max1.20e-5/1.79e-5), explicitly recorded rather than called
bitwise replay. CPU FP64 unit coverage includes exact subsequent updates.

Finite performance:45 CPU/NPU observation pairs have matching inputs/initial
parameter hashes. Selected post-update work and the larger Timed inference
support can differ slightly; retain that distinction. All workflows are terminal:
- cpu-final-c17c613f: passed 372.65s, affinity 0–31/32 ATen threads.
- npu-final-c17c613f: passed 208.51s, affinity 64–71/8 host threads, shared NPU 5.
- npu-precision-d2090654: passed 18.03s, bounded same-function FP64 diagnosis.
- packing-a2c1e08b: passed 7.04s, FC/Lattice selected-versus-padded row profiles.
- consumer-npu-cc2b5767: completed17.70s; finite resumed-update deviations reported.
Records are artifacts/instances/<name>/{status.json,launch.json,run.log,...json};
source-* directories are frozen. No active jobs or queued work remain. The
reviewed evidence is tracked; these artifact paths are not reproduction inputs.
Earlier admission/null-PSE failures and the corrected sorted-zero regression
remain recorded. scripts/status.py reports actual historical states.

Next action: use the public API or select a concrete follow-up from ROADMAP;
do not rerun completed matrices to reconstruct context. Remaining limits include
strict finite numerical tolerances, larger scales and unqualified hardware paths.
Current implementation is PyTorch CPU/single NPU, not a historical native/resident
matrix. Shared timings have actual contention recorded. Conversation-specific
shared leases are not standing permission for a new session.

Preserved user untracked files: docs/tide-learning-lab-TL-REQ-004-handoff.md,
TL-REQ-005-handoff.md, TL-REQ-006-handoff.md and
docs/upstream-lh-semantics-discussion-20261113.md. Public toolchains and reference
repositories remain unchanged. No storage blocker or uncommitted source work.
