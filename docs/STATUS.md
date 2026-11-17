# Current handoff — new instances, 2026-11-18

Authorized branch: graph-execution-foundation. Fixed mathematics:
4f5b9f20a4030d68ee66ae2d26e34659a5e37234 (108-file math/ snapshot verified).
Current HEAD d7437c21; pushed through evidence 02a57ac5. Preserve the four
user-owned untracked docs/*handoff*/upstream-lh discussion files.

Scope: fresh FC, HB-Lattice, HB-Exchange, HB-Multiscale Settle/Timed main
bindings; K/S/X, four reception profiles, quad/Shannon/pool and three parameter
sharing profiles. FP32 implementation/performance; optional CPU FP64 formula
checks. No retired implementation/tests, teaching bindings, LH-DAG/PDG,
historical backend matrices or old P1–P6 obligations.

All prior durable jobs are terminal; inspect python scripts/status.py.
- e73 CPU:227 checks; 8fb CPU:257 checks, finite FC/HB comparisons.
- e73 NPU admission failed before computation; 8fb NPU had six null-PSE runtime
  failures (four Attention passes), then was stopped. Keep both failures.
- d743 NPU:10 Attention forward/VJP checks and four model/device comparisons,
  three AdamW steps and exact next-window checkpoint replay completed.
- d743 cpu-diagnose and npu-hb-medium finished. Their measurements use the old
  device-local random-input policy: exploratory observations, not paired speedups.
Records are artifacts/instances/<job>/{status.json,launch.json,run.log,...json}.
Source snapshots source-e73bc03e/source-8fb66221/source-d7437c21 remain frozen.

Uncommitted increment: static owner metadata/shared-base GEMM; sorted quad with
safeguarded numerical fallback; deterministic input-role order; benchmark CPU
input generation and input/parameter hashes; stronger device root/state checks.
All agent changes are frozen/integrated. First full run found4 failures/296
passes: sorted prefix cancellation invented a2.22e-16 sender at an exact corner.
Stable selected-face reconstruction fixed it without epsilon clipping;8 directed
regressions were added. Final working-source CPU check:308 passed31.81s;
upstream snapshot, harness syntax and diff checks passed. The strengthened
checker partial CPU diagnostic completed only one model before its55s bound:
exact restore passed; continuation keys had finite9e-6 max absolute deviation.
This partial run is not whole-model device qualification.

Next: review tests, commit coherent implementation, freeze it, then run final
CPU/NPU FP32 checks and paired finite comparisons. Compare old/new Lattice with
the same new harness/inputs. Bound diagnostics to these two optimizations;
record remaining costs rather than starting another broad rewrite. Commit
reviewed evidence separately and push the authorized branch without force.

Long work: background.slice, fixed source, explicit runtime/affinity and sampled
64GiB process-tree RSS bound (not a hard cgroup limit). Shared NPU5 policy in
artifacts/instances/admission-borrow-5.json binds actual occupant+lock identity
and expires1794941149.933; recheck before use. Conversation authorization covers
shared use; it is not standing authorization for another conversation. All
public software/reference repositories remain read-only.

Storage rechecked:863GB disk free, quota504/700GB hard with4days soft grace,
273GB shm free. No storage blocker. Current docs/navigation: docs/README.md.
