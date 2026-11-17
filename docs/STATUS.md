# Current handoff — new instances, 2026-11-17

Authorized branch: graph-execution-foundation. Fixed math:
4f5b9f20a4030d68ee66ae2d26e34659a5e37234; complete snapshot math/ checked.
Scope: current FC/HB main bindings, correctness first then performance.
FP32 execution/performance only; CPU FP64 optional for accuracy. Old teaching,
LH-DAG/PDG, historical APIs/backend matrices and old P1–P6 are excluded.
No retired implementation/tests are consulted or reused. Four user untracked
handoff files remain untouched.

Pushed implementation e73bc03e and evidence02a57ac5. New package src/tide:
FC/Lattice/Exchange/Multiscale Settle/Timed; K/S/X, four reception profiles,
quad/Shannon/pool, three sharing profiles, optimizer/checkpoint/flush.
Frozen CPU qualification:227 passed25.26s. First FC FP32 CPU comparison is
recorded in docs/evidence/instances-first-cpu-20261117.{md,json}.
The NPU attempt failed admission (exit3) before device code: every admissible
shared device held an existing exclusive cooperative lock. Preserve this fact.
Both original jobs are terminal; no running measurement at this handoff.

Next implementation increment: zero-reception KV no-write VJP and bounded-key
long-prefill tiles;47 relevant checks passed21.55s including new independent
long-window gradients. Attention remains FP32/CPU FP64; no low-precision work.
Agent owns shared-lease adapter/tests (explicit observed lock-holder identities,
separate borrower mutex, no changes to foreign locks, expiry/owner monitor).
Another agent owns optional stage/profiler diagnostics. Main integrates and
will commit/freeze before new CPU/HB/NPU runs. Do not edit frozen run inputs.

Previous jobs/source/logs: artifacts/instances/source-e73bc03e and
artifacts/instances/{cpu-first,npu-first}/{run.log,status.json,launch.json}.
Inspect python scripts/status.py. New runs use bounded background.slice user
services, all useful CPU/memory and authorized shared NPU, with contention
recorded. Shared admission authorization applies to this conversation only.

Disk has865GB free; user quota502GB/700GB hard (soft300GB, grace4days), shm273GB
at last check. Not blocked. Continue full current delivery per ROADMAP, not
just first FC. User authorized ordinary commits/pushes; no force push.
