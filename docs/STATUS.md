# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch: graph-execution-foundation.

## Authorized work

Make Tide a reusable dependency for training/inference experiments in separate
repositories: versioned configuration, public runtime/session, external data,
configuration equivalence gates, complex-topology coverage, packaging and an
independent consumer example. L1-L4 in ROADMAP owns this active plan.
Commit each coherent tested increment and push immediately. No subagents.
Reference repositories and ObsidianVault remain read-only.

L1 `31d7b96` and L2 `ff7c486` committed/pushed. L3 implementation ready.
L3 directed: CMake core-only configuration passed; external installed generic
wheel passed all three Python applications and installed per-config gates in
artifacts/library-installed-directed-003. API/gate FP32 follow-up: 22 passed
in 49.62s. Earlier -001/-002 failures remain: FP32 sum-loss cancellation at
strict tolerances. Mean-scaled diagnostic losses now pass original thresholds;
explicit tolerances and nonfinite rejection are public, recorded options.
L3 committed/pushed `c08cc90`. Active durable consumption gate:
- unit: tide-library-consumption-20260928-a.service, running in background.slice,
  Nice10, KillMode=control-group; build2/ATen+OMP+BLAS1.
- frozen read-only source: /var/tmp/zlong-graph-execution-foundation/library-l3-source
- clean source: c08cc90; build: /var/tmp/zlong-graph-execution-foundation/library-l3-build
- command: python scripts/build.py --jobs 2 --build-dir BUILD &&
  python scripts/library_consumer.py --build-dir BUILD --output-dir JOB/consumer
- JOB: artifacts/library-consumption-20260928-a; status.json and task.log there.
- inspect: systemctl --user show tide-library-consumption-20260928-a.service;
  tail JOB/task.log; cat JOB/status.json and JOB/consumer/result.json.
- stop: systemctl --user stop tide-library-consumption-20260928-a.service.
No pass result yet. Main checkout may develop L4 while frozen job runs.
L4 implemented; preparing commit. Directed gates: 26 API/config tests passed
in 11.73s. active64 TimedDAG FP32, active64 Settle FP64 and feedback32 FP64
native qualification passed (full observables/VJPs, chunks, three AdamW steps,
fresh-process resume); active node counts 64/64/32, feedback pending79 retained.
Next: commit/push L4, freeze it, wait for the L3 consumption unit to terminate,
then run scripts/qualify_library.py --reuse-build using library-l3-build.
L4 suite is 22 cases at D4/B1/T2; exact P02 includes8192 nodes/four active.
Status and the authorized L4 implementation are currently uncommitted.
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
