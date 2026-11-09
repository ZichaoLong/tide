# Working on tide-learning-lab

- This is the independent experiment branch of `ZichaoLong/tide`, not the foundation source tree.
- Re-entry: `git status --short --branch`; read `docs/STATUS.md`, `docs/ROADMAP.md`,
  `docs/research-design.md`, `docs/research-map.md` and `dependencies/foundation-source.json`.
- The 2026-11-07 user instruction authorizes the first bounded CPU/NPU experiment batch,
  private environments, small public model/data acquisition, builds, checks, phased commits
  and ordinary pushes. On 2026-11-08 the user accepted the reviewed recommendations
  and authorized the bounded follow-up in `docs/studies/second-batch.md`.
  The subsequent 2026-11-08 approval authorizes the third batch in
  `docs/studies/third-batch.md`, with one shared 12-hour execution deadline.
  The latest 2026-11-08 approval authorizes the multi-stage capability-preservation
  batch in `docs/studies/fourth-batch.md`: one NPU, at most 2 hours allocated compute,
  one shared 12-hour wall deadline, and conditional progression through the listed stages.
  The subsequent research-priority instruction centers natural-text next-token quality
  jointly with scaling, sparsity and locality; current design is
  `docs/studies/language-structure.md`. It supersedes backlog priorities, not historical
  results or active inputs. The latest approval authorizes implementing this package
  and evidence-driven local-module, parameter, loss and recipe adjustments within
  one 12-hour wall deadline, 8 aggregate NPU hours and 2 CPU-only worker hours.
  Keep adjustments finite and named, preserve original failures and thresholds, and
  use public extension points without copying execution core. The design JSON
  remains non-launchable; runtime configs live under `configs/studies/language/`.
  After the v3 terminal review, the user authorized the new bounded follow-up in
  `docs/studies/language-confirmation-v4.md`: A512 plus one extra-exposure Transformer,
  one B rank32 control, and conditional matched FC8; a new shared 12h/8NPUh batch.
  Historical costs and failures remain closed and separate; no automatic seed22.
  The 2026-11-09 acceptance authorizes `docs/studies/language-depth-proposal.md`:
  eight main training runs and two conditional HB40 runs, a new shared 12h and
  24 aggregate compute-worker-hour cap, up to six workers including the old job.
  The user explicitly allows low-HBM/low-AICore shared-card admission and CPU
  multithreading to improve turnaround, overriding the skill's idle-only/FIFO
  rules. CPU training substitutes within that same aggregate compute cap.
  Preserve background.slice, fixed inputs, measured headroom and original failures.
  Authorized retirement covers the audited six obsolete artifacts and redundant
  old Trackio projections with local evidence retained; never delete active inputs.
  Do not expand beyond the selected bounded table. Shared environments and other projects remain read-only.
- Pin mathematics and foundation source. Use installed public APIs and the pinned downstream
  contract; never import foundation benchmark scripts or copy/fork its execution core.
- Foundation, LH, fractal-latcarf and ObsidianVault are read-only from this branch unless the
  user explicitly authorizes a separate change. Do not fetch retired foundation history.
- Keep STATUS as the current handoff and ROADMAP as the backlog. Research design is not a job log.
- Status/review code reads `runs/` and `artifacts/jobs/` without writing there. Use
  `scripts.review_io.write_review_json` for graph/evidence JSON writes; it validates
  the destination before creating directories. A graph schema in a run state file
  is record contamination, never a training state.
- Keep actual results separate from planned experiments and expected result slots. Missing
  evidence never means passed. State each claim's domain, comparisons and transfer limits.
- Model/checkpoint ancestry, execution prerequisites and scientific support are distinct relations.
- Human graph data live in `research/map.json`; `web/index.html` is generated. After edits run
  `python scripts/render_map.py`, `python scripts/render_map.py --check` and relevant checks.
  Keep visualization offline, without credentials, external scripts or implicit network requests.
- Use atomic replacement for handoffs. Freeze source and installed dependencies before long work.
  Local long jobs use background.slice and shared-device allocation rules; no active job inputs
  may be edited. All experiments need finite resources, logs, stops and explicit terminal state.
- No sub-agents unless the user explicitly requests delegation. Preserve user changes;
  commit coherent work on tide-learning-lab, and never force-push.
