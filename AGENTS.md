# Working on graph-execution-foundation

## Re-entry (also after context compression)

1. Run `git status --short --branch` and `python scripts/status.py`.
2. Read `docs/STATUS.md`; follow its next action and active-job record.
3. Read `docs/ROADMAP.md` and `docs/execution-flows.md`. Use
   `docs/semantic-map.md`, `docs/architecture.md` and `rg` to locate definitions,
   implementation and independent tests. `docs/development.md` starts from a clean checkout.
4. Consult `docs/semantics.md` before changing an observable behavior.
   The immutable upstream revision and offline textbook snapshot hashes are in
   `docs/upstream.json`; run `python scripts/check_upstream.py`. Do not adopt
   an upstream working-tree draft or a new semantic version without the user's task.
5. Preserve user changes and work only on the authorized branch. Reference
   repositories (`lh`, `fractal-latcarf`, ObsidianVault) are read-only.
6. Old evidence uses original commit IDs. Resolve them with
   `python scripts/resolve_history_commit.py OLD_SHA`; see `docs/history-migration.md`.
   Clone/fetch only `graph-execution-foundation` for the small development base.
   Do not merge or fetch `graph-execution-foundation-old` into that checkout.

## Downstream collaboration

- For experiment-repository requests, read `docs/downstream-contract.md` at the
  adopted protocol revision. Check the request's fixed source, dependency lock,
  actual configuration, evidence and acceptance conditions before changing code.
- Keep the request in its originating repository; link accepted foundation work
  from ROADMAP and current activity from STATUS. Deliver implementation/evidence
  commits and a downstream reproduction command. Upstream delivery and each
  downstream acceptance are separate facts, never inferred from one another.
- Use `templates/downstream/` for portable requests/replies/acceptance and
  `examples/consumer_core5/` for installed public-API consumption. Repository
  access and request text do not grant cross-repository write/push authority;
  existing user authorization still applies without repeated permission checks.
- Freeze dependencies used by active experiments. Keep application-specific
  training in the consumer; contribute general modules/optimizations here with
  independent tests, explicit scope and recorded positive/negative lessons.

## Execution and checkpoints

- Continue implementation autonomously in reviewable, tested increments.
- `STATUS.md` is the single current handoff; `ROADMAP.md` is the single backlog.
  Replace stale status, do not accumulate session diaries or duplicate ledgers.
- `scripts/status.py` shows all live and recent terminal job records. Use
  `--all-jobs` when auditing retained historical failures; never relabel an old
  failure as passed merely because a newer run succeeded.
- Use fsynced staging plus atomic replacement for programmatic handoff writes;
  verify the result. Prefer `scripts/durable_records.py` for these writes.
  Check storage capacity before large writes. A failed write
  must not replace a known-good handoff or be reported as successful.
- Update the handoff before long jobs, at commit boundaries, and before ending
  a session. Include exact next commands, blockers, source identity, job/unit,
  logs, terminal status and any uncommitted work. Never call a live job passed.
- Commit a coherent implementation after relevant tests pass. For durable
  qualification, commit the implementation first, test that immutable commit,
  then commit the evidence separately. Do not push without task authorization.
- Use `docs/verification.md` for development/full gate scopes and isolated
  qualification. Smoke evidence must never stand in for a complete matrix.
- Long builds/tests use a detached user service in `background.slice`, bounded
  build/CPU threads, a frozen checkout and durable status/logs. Builds default to
  all affinity-available CPUs; `--jobs half` or an explicit positive count is
  allowed. No fixed two-worker cap. Record actual constraints when reducing
  workers. Tests may use idle CPU/NPU/memory; heavy timing and builds do not
  overlap within this project. Never edit source
  that an active job reads. Short checks may run interactively.
- Do not delegate to sub-agents unless the user explicitly requests delegation.

## Structure and contracts

- One responsibility per file; target < 350 lines, investigate files > 500.
  Split by semantic layer rather than arbitrary line count. Generated data,
  lock files and checksummed immutable upstream snapshots are exempt. Use symbol
  searches and bounded reads, not whole trees.
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
- CPU FP64/FP32 remain the reference. Eager and resident CPU/CUDA/NPU scopes
  are declared separately in the execution contract and module capability gates.
  Explicit unavailable devices/dtypes must fail; never silently substitute an
  algorithm. CUDA hardware validation remains pending; keep source/build and
  actual-device evidence separate. User alignment overrides generic experiment skills.

## Evidence and housekeeping

- `docs/evidence/` contains small reviewed reports tied to exact source commits.
  Build products, raw logs, snapshots and checkpoints belong under ignored
  `artifacts/` or `build/`, never alongside source.
  The retired-record archive is preserved only on the frozen old branch;
  `docs/history-migration.md` documents explicit retrieval outside this checkout.
  Preserve actual historical evidence identities; use the map to find equivalent code.
- Preserve active-job inputs, unsaved changes and minimal reproducers for new
  unresolved issues. The user authorized retirement of historical experiment
  artifacts on 2026-10-12; old reports need not be re-audited by default. Follow
  `docs/artifact-retention.md`: keep necessary records, inspect the dry run,
  and distinguish historical citations from live dependencies. Never clean
  reference repositories or shared toolchains as experiment artifacts.
- Use `docs/reproduction.md` to connect unresolved issues to tracked generators,
  small fixtures, diagnostics and reviewed evidence. A clean checkout must not
  depend on retired run directories. Keep current limitations only in ROADMAP.
  At authorized retirement, curate necessary lessons/inputs first, then remove
  raw runs and temporary environments/builds; never promote an outer job status
  into a claim that every inner check passed. Retired byte hashes are identities,
  not a guarantee that the original bytes remain available.
- Git history retains superseded plans/evidence; do not keep renamed copies of
  old handoffs. Fix broken navigation when moving files.
- Report implemented/verified/planned separately. A small formula profile does
  not certify all models, graph families, scales, platforms or performance.
- `docs/semantics.md` owns current graph/checkpoint versions. Module documents
  link there; immutable evidence retains the versions tested at its source commit.
