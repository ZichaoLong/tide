# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch: graph-execution-foundation.

## Authorized work

Make Tide a reusable dependency for training/inference experiments in separate
repositories: versioned configuration, public runtime/session, external data,
configuration equivalence gates, complex-topology coverage, packaging and an
independent consumer example. L1-L4 in ROADMAP owns this active plan.
Commit each coherent tested increment and push immediately. No subagents.
Reference repositories and ObsidianVault remain read-only.

Start source: clean `62c83ea`. L1 implementation complete; preparing its commit.
Directed gate: 95 tests passed in 10.70s (test_library + checkpoint_ownership),
CPU FP32/FP64. Next: L2 package-owned per-config qualification, then L3/L4.
No durable job is running. New code/docs are the authorized L1 increment.
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
