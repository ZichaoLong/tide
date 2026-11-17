# Current handoff — instance rewrite, 2026-11-17

Authorized branch: graph-execution-foundation. Fixed math:
4f5b9f20a4030d68ee66ae2d26e34659a5e37234, snapshot math/ (108 files checked).
Scaffold/governance commit c0d5e07b; first complete implementation follows it.
Retired implementation/tests are not consulted or reused. User four untracked
handoff documents remain untouched. Old P1–P6 is not the active task.

New source: src/tide. FC, Lattice, Exchange, Multiscale Settle/Timed; K/S/X,
four reception profiles, quad/Shannon/pool, three parameter sharing profiles.
Independent FC formulas, allocation/Attention/SSD formulas, topology, gradient,
optimizer/checkpoint/continuation/Timed flush checks: 227 CPU tests passed
in 24.43s before this implementation commit. This is development evidence,
not frozen-source or NPU qualification. Execution/performance are FP32 only;
CPU FP64 is used for independent accuracy checks.

Current work: long-window independent KV checks, real-device FP32 checks and
finite comparisons. Independent HB oracle and 21 parameter inventory cases pass. Fresh Transformer
and SSD baselines in benchmarks/baselines.py; benchmark entry benchmarks/run.py.
No active durable jobs. No new device or performance result yet.

Next: finish current owned edits, run CPU tests, commit implementation, freeze
that commit for bounded background.slice CPU/NPU checks and measurements.
Main owns model/config/benchmark integration; agents own independent HB oracle,
Attention/device primitive checks, and SSD/parameter audit respectively.
Shared NPU use is explicitly authorized in this continuing conversation;
scripts/npu_shared_queue.py retains site queue/locks and requires an expiring
observed-PID policy. No shared default for later conversations. Long jobs use
frozen source and separate logs; never edit their inputs.

Disk: 865GB free, user quota 502GB of 700GB hard (soft300GB, grace4days),
shm273GB available at last check. Not blocked; small new source/results only.
Continue all current main candidates and comparisons, not merely first FC.
