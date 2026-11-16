# Working on tide-learning-lab

- Latest user instruction reduces the current campaign further to FOUR physical
  NPUs: only4/9/12/14, four slots in the existing namespace. See
  `docs/studies/hb-v12-four-card.md` and `configs/studies/language/hb-v12/four-card.json`.
  Keep the already extended2026-11-19 23:08:07+08 deadline and all scientific/disk
  limits. Four original services remain; two additional boundary-saved trains
  plus the fourteen stopped six-card waiters are explicitly requeued as16
  `v12-four-*` descendants. These are administrative continuations, not new
  numerical repairs. Never restart the old12-slot controller or six-card waiters.
  Model source62e32602 remains fixed; only orchestration/resource inputs change.
  Current status: `python scripts/review_hb_v12_six_card.py --policy four-card`.
  The six-card notes below are historical.

- The latest 2026-11-16 user request reduces the current campaign to at most SIX
  physical NPUs and permits more wall time. See `docs/studies/hb-v12-six-card.md`
  and `configs/studies/language/hb-v12/six-card.json`: use only4/6/9/11/12/14,
  deadline2026-11-19 23:08:07+08. This supersedes the old12-slot/original deadline
  for administrative continuation only. Preserve38 chains,128Mi/configs/seeds,
  original thresholds and64GiB target/96GiB stop/>=200GiB hard-quota headroom.
  Six train owners saved complete boundary checkpoints and exited; eight waiters
  and the old controller were stopped. These six raw outer failures are USER
  RESOURCE PAUSES, not new solver failures or second repairs. Frozen model source
  stays62e32602; new orchestration is a separately pinned input. Do not restart the
  old12-slot controller. Inspect `scripts/review_hb_v12_six_card.py` for live state.
- V12 repair qualification is now completed on CPU and NPU with Lens readback;
  eight repair chains have completed, six are retained running, and fourteen
  administrative descendants are submitted and waiting for those six cards.
  Orchestration sourcea9a63245 is a separate frozen input; model source62e32602
  stays unchanged. See `docs/evidence/hb-v12-six-card-handoff.json`.
  The queued-qualification notes
  below are historical. Full comparison quality remains awaiting scoped review.

- The 2026-11-16 user instruction authorizes the bounded v12 repair in
  `docs/studies/hb-v12-repair.md`: repair quad collapsed-interior feasibility
  and live disk-scan disappearance, then explicitly resume the affected original
  configurations under new frozen source. Keep the original deadline,38 training
  chains,128Mi endpoints,12 shared slots and disk limits. No active old inputs
  change, no relaxed thresholds and no silent retry; at most one repair per chain.
- V12 repair preparation is submitted: qualification source555a6863 passed CPU;
  NPU qualification is queued. Durable controller source62e32602 at
  `.local/source-hb-v12-repair-v2` waits for qualification/Lens before submitting
  explicit repair descendants. Inspect `scripts/review_hb_v12_repair.py` and
  `docs/evidence/hb-v12-repair-handoff.json`; do not equate a queued check with
  qualified NPU operation or a running controller with resumed training.
- Latest user approval authorizes the fuller HB mechanism campaign in
  `docs/studies/hb-mechanisms-v12.md`:19 named variants x seeds11/22, each bounded
  at128Mi, shared48h,<=200M parameters, shared12 NPU slots, no aggregate compute
  cap,64GiB disk target/96GiB stop and>=200GiB hard-quota headroom. Implement and
  submit directly after finite checks/calibration. This supersedes v11's
  single-intervention restriction for NEW v12 work only. Preserve active v11
  frozen inputs, original failures and2e-5 feasibility threshold. Explicit
  sharing/readout/top2 variants are different functions, not foundation fixes.
- V12 is now submitted: all38 independent chains use frozen51477014 at
  `.local/source-hb-v12-main`;8 calibration jobs (8e085be3) and fresh262K data
  preparation completed. Re-entry: `python scripts/review_hb_v12.py` and
  `docs/studies/hb-v12-operations.md`. Twenty historical controls are evaluated
  without further training. Actual queue/physical devices are in job records;
  pilot/recovery/128Mi/confirmation/diagnostic progression is already scheduled.
  V11's four HB bias attempts are now all feasibility failures, not successful
  128Mi endpoints. V12's bounded old-source replay is diagnostic only. Do not
  edit active source/configs, relax thresholds or silently restart failures.

- This is the independent experiment branch of `ZichaoLong/tide`, not the foundation source tree.
- Re-entry: `git status --short --branch`; read `docs/RESEARCH_CHARTER.md`, `docs/STATUS.md`, `docs/ROADMAP.md`,
  `docs/research-design.md`, `docs/research-map.md` and `dependencies/foundation-source.json`.
- The durable objective and user preferences live in `docs/RESEARCH_CHARTER.md`:
  natural-text next-token quality jointly with data/capacity/dependency-depth scaling,
  lower activity and bounded locality on the same model. Learning admission and broad
  screening are milestones, not the endpoint. Dense computation plus masks may remain
  the main training implementation when faster; separately record logical activity,
  physical work and achieved throughput. Real sparse execution is not a prerequisite
  for further quality/scaling studies. Cover individual local candidates as a union;
  calibrate Transformer controls at new scales and preserve scoped negative evidence.
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
- The subsequent 2026-11-09 instruction authorizes the bounded TL-REQ-003 acceptance
  in `docs/studies/positive-support-acceptance.md`. It is complete: see its results page.
  Current selected foundation is 74e7bbd in `.venv-positive-floor-v1`; legacy `.venv`
  and frozen jobs retain 85d161d. Explicit `positive-floor-v1` passed scoped CPU/NPU
  admission and CPU application checks, with strict finite failures retained.
  Do not infer full B/Qwen migration or authorization to launch another batch.
- The latest 2026-11-09 user instruction authorizes completing the previously
  underflow-blocked v5 runs using the accepted patch; see
  `docs/studies/language-depth-completion-v5.md`. Preserve the original 24 worker-hour
  cap and 22:49:13 deadline, original candidates/exposure and conditional HB40 prerequisites.
  Shared devices need not be idle or pass FIFO admission; choose by completion time,
  health and memory headroom, retain lifetime locks and background.slice. No new scaling batch.
- The 2026-11-09 user approval now authorizes the concurrent local-candidate coverage
  and substantive language scaling campaign in `docs/studies/language-coverage-scale-v6.md`.
  Cover the union of local candidates, not a full Cartesian product. Broader CPU/NPU
  parallelism and one-to-two-day jobs are authorized; freeze a new 48h campaign with
  separate CPU/NPU and disk caps. Implement, calibrate and submit directly. This
  supersedes the previous small follow-up recommendation, not historical evidence.
- The completion batch is now terminal; see `docs/studies/language-completion-v5-results.md`.
  All five training/continuation chains and evaluations finished by 17:16 on 2026-11-09.
  Scoped learning/quality findings do not waive strict numerical failures. No automatic
  old follow-up is active; the separately authorized v6 campaign owns new work.
- The 2026-11-11 user authorizes the finite HB cost study and read-only learning review
  in `docs/studies/hb-cost-and-learning-20261111.md`, including a non-TIDE direct Torch
  DAG cost anchor. Keep it explicitly different-function and outside the execution
  core; do not adopt the foundation working-tree changes or launch new language training.
  This study is terminal: 25 cost cases and 8 read-only weight VJPs completed, with
  scoped reports in `docs/studies/hb-cost-results-20261111.md` and
  `docs/studies/hb-learning-results-20261111.md`. Recipe/architecture follow-ups are
  recommendations, not submitted jobs; preserve original failures and frozen inputs.
- Pin mathematics and foundation source. Use installed public APIs and the pinned downstream
  contract; never import foundation benchmark scripts or copy/fork its execution core.
- The latest 2026-11-11 user approval supersedes the preceding no-new-training restriction:
  execute `docs/studies/direct-language-v7.md` using the existing ordinary Torch direct graph
  while upstream is unavailable. This different-function model may test recipe/topology,
  never impersonate TIDE execution or routing. Bound 12 screen, 10 main trajectories and
  at most 2 conditional controls within 12h/8 NPUh/32 CPU core-h/32GiB new disk; shared-card
  authorization remains valid. Preserve frozen historical source and all failed evidence.
  This v7 batch is now terminal: 12 screen stages and 10 main 4M trajectories completed;
  residual controls did not trigger. See `docs/studies/direct-language-v7-results.md`.
  Next-batch options are not submitted; do not automatically extend exposure or candidates.
- Foundation, LH, fractal-latcarf and ObsidianVault are read-only from this branch unless the
  user explicitly authorizes a separate change. Do not fetch retired foundation history.
- The latest 2026-11-11 instruction authorizes `docs/studies/direct-coverage-v8.md`:
  redo the 91-recipe FC/HB union with independent direct mathematical modules and
  substantively calibrated Transformer controls. No copying foundation execution core.
  New finite 48h/240 NPUh/768 CPU core-h/128GiB campaign; shared low-load admission
  remains authorized. Defer the separate true-sparsity and scaling main campaign
  until coverage review. Preserve all historical frozen runs and strict failures.
  This batch is terminal: see `docs/studies/direct-v8-results.md`. All 220 training
  attempts and 40 services are terminal; 194 complete and 26 original failures remain.
  The latest user feedback is persisted in `docs/RESEARCH_CHARTER.md`; the replacement
  next-batch recommendation is `docs/studies/language-scale-activity-v9-proposal.md`.
  The latest user response basically accepts the v9 direction and a shared 48h wall
  deadline, and removes aggregate NPU-device-hour and CPU-core-hour caps. Record
  those costs, but do not stop on the superseded 200 NPUh/512 core-h or sub-budgets.
  Low-activity and inlet-tree alignment is now accepted; see
  `docs/studies/hb-low-activity-alignment.md` and the authorized execution table below.
  V9 has a frozen durable controller and actual NPU calibration/training services.
  Report receipt/all-node slots, Full/candidates and Full/all-node slots separately;
  current HB already has partial receipt. Static inventories and small fixed-route
  checks are not trained scale/sparsity or physical compute-saving results.
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

- The latest explicit approval authorizes executing `docs/studies/language-scale-activity-v9.md`, including the inlet-tree/original-inlet comparison. One shared 48h wall deadline, no aggregate NPU/CPU time caps, 64GiB new disk target and >=200GiB hard-quota headroom. Freeze new inputs, calibrate useful batches and Transformer recipes, submit independent durable work directly. This supersedes v9 design-only/no-submission notes, not historical failures.

- The 2026-11-12 user approval authorizes expanded historical checkpoint retirement in
  `docs/studies/checkpoint-retirement-20261112.md`. Preserve all v9 and non-completed
  trajectories, named scientific controls/negative witnesses, records and public models.
  Completed v6-v8 screening weights outside the named retention set and 14 old GEF
  benchmark states may be retired by the fixed manifest. Historical recovery records
  do not promise availability of retired bytes; consult the retirement receipt first.
  This retirement is terminal: 74.7565GiB released, 1,130 files retired and 76 exact
  duplicates hardlinked; selected weights and historical evidence verified intact.
  Do not rerun the executed manifest or assume retired screening weights still exist.

- V9 training is now terminal: 58 main trajectories and 12 inherited extensions completed;
  all services have no live descendants. See `docs/studies/direct-v9-results.md` and its
  read-only auditor `scripts/review_direct_v9_results.py`. The old progress script covers
  main trajectories only. Reviewed validation findings do not complete independent
  confirmation: the fresh 262K split is still unevaluated and test remains sealed.
  Preserve the four original interruptions, strict failures, fixed source and all v9
  checkpoints. No automatic next training batch; retain the scoped negative inlet/depth
  evidence alongside capacity/exposure gains.

- The latest 2026-11-14 approval authorizes `docs/studies/language-capacity-activity-v10.md`: bounded specialized direct implementation optimization followed by capacity/exposure comparisons at about10%/3% activity and matched small dense controls. Maximum200M parameters, one shared48h deadline (first2–4h optimization), no aggregate compute caps;64GiB new disk target/96GiB ceiling and at least200GiB hard-quota headroom. Freeze new source/inputs; preserve v9 and all strict failures. No1G, broader candidate sweep or automatic additional depth.

- The 2026-11-15 approval authorizes `docs/studies/direct-v10-repair.md`: explicit
  new-source ordered quotient VJP, scoped regression, trained-state fresh-process
  recovery and 64-update pilots, then the three failed pool controls to their
  original 128Mi endpoints and predeclared confirmation. Reconstructed pre-failure
  parents carry explicit ancestry; old failures/source remain unchanged. Keep the
  original 48h deadline and global 12-slot policy, 24GiB repair-only disk ceiling;
  no automatic repair extensions or changes to the original controller's decisions.

- The subsequent 2026-11-15 approval authorizes `docs/studies/moe-controls-proposal.md`:
  implement and calibrate ordinary top2 Transformer-MoE (two sizes, E20/E64) and matched
  dense controls, at most30 seed33 8Mi screens /12 selected32Mi continuations /12 fresh
  seed11/22 128Mi main trajectories, then independent262K confirmation. New48h wall
  deadline, at most8 new workers sharing the existing12-slot namespace; no aggregate
  compute caps,64GiB disk target/96GiB stop and>=200GiB hard-quota headroom. No automatic
  256Mi MoE extension. Preserve active v10 source, old failures and independent axes.

- The MoE supplement is terminal as of 2026-11-15 14:26 +08: 30 screens,12 selected
  continuations,12 fresh128Mi trajectories and12 independent confirmations completed.
  See `docs/studies/moe-controls-results.md` and read-only `scripts/review_moe_results.py`.
  Preserve the12 original causal-zero admission failures and finite CPU/NPU mismatches.
  Calibration Adam states were retired by the recorded42-file receipt; formal
  checkpoints and scientific weights remain. No automatic MoE extension/new grid.
  V10 is also terminal as of 2026-11-15 16:54 +08: 20 distinct128Mi configurations
  (17 original successes plus3 explicit repairs),7 successful256Mi extensions and
  all27 independent confirmations. All relevant services/controllers have no live
  descendants. See `docs/studies/direct-v10-results.md` and its read-only auditor.
  Preserve the3 original pool failures and the q10 seed11 extension feasibility
  failure at195.84Mi; no silent restart or relaxation of its2e-5 threshold.
  The q10 256Mi result is seed22 only. Neither this terminal review nor remaining
  wall budget authorizes a new training batch or automatic exposure extension.

- The latest 2026-11-15 approval authorizes `docs/studies/capacity-utilization-v11.md`:
  at most16 endpoint diagnostics,6 large MoE/dense128-to256Mi continuations,2 fresh
  small E64 balance.01 controls to128Mi, and one diagnosis-selected HB intervention
  family (at most2 variants ×2 seeds, short pilots then at most128Mi). New shared48h,
  at most12 NPU workers, no aggregate compute caps,64GiB target/96GiB stop and>=200GiB
  hard-quota headroom. Prepare fresh confirmation excluding all prior splits;
  old confirmation is now seen evidence. Preserve original failures and fixed
  parents; do not turn dense-to-MoE gains into an E20-toE64 scaling claim.

- V11 is executing, not design-only. Six original extension requests made zero new
  updates due to the old128Mi config endpoint; preserve them as unfulfilled requests.
  Explicit endpoint-only r2 migrations (e3b2ab38) passed full-batch trained-state
  recovery and now progress toward256Mi. Sixteen diagnostics completed; the reviewed
  selection is only candidate-aware HB load-bias rates.001/.01 ×seeds11/22, fixed
  source159190c3,8Mi pilot then trained-state recovery and conditional128Mi. See
  `docs/studies/capacity-v11-diagnosis-and-intervention.md` and operations page.
  Do not also launch Read/FFN budget or propagation variants. New confirmation
  is prepared and fixed28 endpoints are controlled in background; test stays sealed.
  Shared deadline2026-11-17 17:43:08+08,12 NPU slots,96GiB stop,>=200GiB headroom.
  No active fixed input may be edited. Re-entry: `python scripts/review_capacity_v11.py`.

- V11 inspection2026-11-15 21:32+08: all8 MoE/dense endpoints and24 fresh
  confirmations completed and scoped-reviewed. See `capacity-v11-progress-20261115.md`.
  HB seed11 variants remain running on6/8; both seed22 variants failed allocation
  feasibility (2.432e-5/2.813e-5 > original2e-5) after21.477/32.288Mi. Do not
  conflate the last progress with saved weights (13.327/32Mi), waive the failure,
  or silently restart. Existing128Mi targets and shared deadline remain. Status
  requests do not launch repair or broaden the selected intervention family.
