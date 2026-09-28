# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch: graph-execution-foundation.

## Authorized work

Make Tide a reusable dependency for training/inference experiments in separate
repositories: versioned configuration, public runtime/session, external data,
configuration equivalence gates, complex-topology coverage, packaging and an
independent consumer example. L1-L4 in ROADMAP owns this active plan.
Commit each coherent tested increment and push immediately. No subagents.
Reference repositories and ObsidianVault remain read-only.

Implementation commits pushed: L1 `31d7b96`, L2 `ff7c486`, L3 `c08cc90`,
L4 `67884f3`. Post-review compatibility fix: include new cmake/ templates in
legacy CPU comparison exports; relocated PDG CMake configuration passed.
Latest static audit: zero errors,45 warning review leads (new ones are explicit
backend guards/negative tests). No new hardware support is claimed.

Active durable gate: tide-library-consumption-20260928-a.service.
Frozen clean source c08cc90 at /var/tmp/zlong-graph-execution-foundation/library-l3-source;
build /var/tmp/zlong-graph-execution-foundation/library-l3-build. Build109/109 is
complete; installed Python/native gates and CMake consumer are still running.
Job/status/log: artifacts/library-consumption-20260928-a/{status.json,task.log};
consumer/{result.json,commands.log}. background.slice/Nice10/KillMode=control-group,
build2 and ATen/OMP/BLAS1. No terminal pass claim yet.
Inspect/stop with systemctl --user show/stop tide-library-consumption-20260928-a.service.

Directed L4:26 API/config tests passed; three mixed active native gates passed:
64-node/768-edge TimedDAG FP32 and Settle FP64,32-node feedback FP64 (pending79).
Earlier installed -001/-002 failures remain; normalized diagnostic losses pass
original strict tolerances in -003. Wheel runtime is independent of source paths.
Next: commit/push export fix, freeze the new head, wait for the consumption unit,
then run scripts/qualify_library.py --reuse-build --build-dir library-l3-build
through a new durable job. It includes full CPU regression,22 complex cells
(D4/B1/T2/three steps; exact P02 has8192 nodes/four active),installed consumers.
CPU FP32/FP64, aarch64 Torch/LibTorch2.10.0+cpu; ATen/OMP/BLAS1, build2.
CPU operator probe passed. Static audit: zero errors, 40 existing review leads.
About 13 GiB disk free at re-entry; recheck before large writes.

## Retained completed evidence

- Six-stage CPU foundation: [final gate](evidence/foundation-final.md).
- E1-E5: [qualification](evidence/cross-family-qualification.md), 8577 tests and
  60 relocated smoke; [performance](evidence/cross-family-performance.md).
- Extended narrow Settle: 46,912 nodes, 8,496,773,056 parameters; 1034.865380s
  total under 1800s. Original 900s failure remains historical evidence.
- [Python NPU](evidence/npu-python-20260925.md): clean7811418, 10 finite FP32
  smoke cases. C++ NPU remains unsupported pending a standalone SDK.
- Latest CPU gate: clean7e1477a, 8580 tests in1383.38s, exit0;
  artifacts/cpu-npu-qualification-20260925-b/result.json.
- Full/Aggregate VJP evidence remains separately scoped; State/Read replay
  remains. Large performance completion does not certify full-scale equivalence.

## Re-entry and resources

Run git status --short --branch, git log -6 --oneline, python scripts/status.py.
Implementation and evidence commits are separate. Long jobs use frozen worktrees
in background.slice/Nice10, unique logs and terminal records. Use at most half
available CPU/memory; never edit a live job checkout. Consumers install into
their own environment and write outputs outside the Tide source tree.
