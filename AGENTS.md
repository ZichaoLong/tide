# Working on graph-execution-foundation

## Re-entry (also after context compression)

1. Run `git status --short --branch` and `python scripts/status.py`.
2. Read `docs/STATUS.md`; follow its next action and active-job record.
3. Read only the relevant section of `docs/ROADMAP.md`, then use
   `docs/architecture.md` and `rg` to locate the implementation/test.
4. Consult `docs/semantics.md` before changing an observable behavior.
   The immutable upstream revision is in `docs/upstream.json`.
5. Preserve user changes and work only on the authorized branch. Reference
   repositories (`lh`, `fractal-latcarf`, ObsidianVault) are read-only.

## Execution and checkpoints

- Continue implementation autonomously in reviewable, tested increments.
- `STATUS.md` is the single current handoff; `ROADMAP.md` is the single backlog.
  Replace stale status, do not accumulate session diaries or duplicate ledgers.
- `scripts/status.py` shows all live and recent terminal job records. Use
  `--all-jobs` when auditing retained historical failures; never relabel an old
  failure as passed merely because a newer run succeeded.
- Update the handoff before long jobs, at commit boundaries, and before ending
  a session. Include exact next commands, blockers, source identity, job/unit,
  logs, terminal status and any uncommitted work. Never call a live job passed.
- Commit a coherent implementation after relevant tests pass. For durable
  qualification, commit the implementation first, test that immutable commit,
  then commit the evidence separately. Do not push without task authorization.
- Long builds/tests use a detached user service in `background.slice`, bounded
  build/CPU threads, a frozen checkout and durable status/logs. Never edit source
  that an active job reads. Short checks may run interactively.
- Do not delegate to sub-agents unless the user explicitly requests delegation.

## Structure and contracts

- One responsibility per file; target < 350 lines, investigate files > 500.
  Split by semantic layer rather than arbitrary line count. Generated data and
  lock files are exempt. Use symbol searches and bounded reads, not whole trees.
- Keep the C++ core independent of Python. Bindings are a testing/client adapter.
  Python reference scheduling must remain independent of native scheduling.
- Specializations must have independent schedules. Wrapping a generic executor
  is not a specialization or an independent equivalence anchor.
- Keep local kernels, graph representation, scheduling, checkpointing and
  comparison separate. Retain edge identities, including parallel edges.
- Changes to observable semantics need a short contract update and meaningful
  tests. HST is a declared VJP; do not gradcheck its hard forward.
- Avoid in-place autograd mutation. Compare final states, in-flight messages,
  history, routes and gradients, not just outputs. Check disconnected gradients.
- CPU FP64/FP32 are required now. Explicit unavailable devices/dtypes must fail.
  Never silently substitute an algorithm. GPU/NPU support is deferred.

## Evidence and housekeeping

- `docs/evidence/` contains small reviewed reports tied to exact source commits.
  Build products, raw logs, snapshots and checkpoints belong under ignored
  `artifacts/` or `build/`, never alongside source.
- Keep failure reproducers until fixed and tested. Keep artifacts cited by
  current evidence and active jobs. Remove only known obsolete project-owned
  artifacts; inspect a dry run before deleting. Never clean reference repos.
- Git history retains superseded plans/evidence; do not keep renamed copies of
  old handoffs. Fix broken navigation when moving files.
- Report implemented/verified/planned separately. A small formula profile does
  not certify all models, graph families, scales, platforms or performance.
- `docs/semantics.md` owns current graph/checkpoint versions. Module documents
  link there; immutable evidence retains the versions tested at its source commit.
