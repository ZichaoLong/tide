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
  Do not expand beyond that table without another review. Shared environments and
  other projects remain read-only.
- Pin mathematics and foundation source. Use installed public APIs and the pinned downstream
  contract; never import foundation benchmark scripts or copy/fork its execution core.
- Foundation, LH, fractal-latcarf and ObsidianVault are read-only from this branch unless the
  user explicitly authorizes a separate change. Do not fetch retired foundation history.
- Keep STATUS as the current handoff and ROADMAP as the backlog. Research design is not a job log.
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
