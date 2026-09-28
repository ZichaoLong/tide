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

L3 durable consumption passed: clean c08cc90, unit
`tide-library-consumption-20260928-a.service` terminal exit0, 02:13:34 UTC.
artifacts/library-consumption-20260928-a/{status.json,consumer/result.json}:
109/109 fresh build,6 external Python/native apps,3 installed config gates,
and installed standalone C++ FP32/FP64 forward/chunk/backward.

Full library gate passed at clean ff708a1, unit
tide-library-release-20260928-a.service, exit0,02:54:59 UTC:
- 8621 CPU tests passed in1531.34s;
- 22 complex cells passed (both CPU dtypes, Python/native active mixed graphs,
  exact P01/P02/T02/S01/A02 topologies; P02 retains8192 nodes/four observed);
- 6 installed Python/native applications,3 installed configuration gates and
  installed C++ FP32/FP64 consumer passed.
Records: artifacts/library-release-20260928-a/qualification/{result.json,cpu,
complex,consumer}; wrapper status/log in its parent. No live job from this gate.

Review follow-up: validator now checks trace-disabled VJPs, not only values.
A deliberately wrong backward with identical forward is rejected; native
compact/deferred release/packed transport passes in both dtypes. Directed:
16 qualification tests passed in83.75s;3 trace-disabled tests passed in14.89s
(18 distinct cases including the new two-dtype optimized-path test).
Only production diff since ff708a1 is qualification.py; executors/models unchanged.
Next: commit/push this validator hardening, freeze new head, durable rerun of all
18 qualifier tests and installed consumer gate. Retain ff708a1 complete evidence;
no need to repeat unaffected executor regression. Then evidence commit/push.
The validator fix, test/docs and this handoff are uncommitted. No live job yet.
CPU FP32/FP64, aarch64 Torch/LibTorch2.10.0+cpu; ATen/OMP/BLAS1, build2.
CPU operator probe passed. Static audit: zero errors,45 reviewed warning leads.
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
