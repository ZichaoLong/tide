# Current handoff

Updated 2026-09-29 15:29 UTC. The fixed CPU/NPU baseline, full-size profiling and bounded
scheduler assessment is terminal. No task-owned jobs remain running. No push.
No sub-agents. Reference repositories and ObsidianVault remain read-only.
Re-entry: `git status --short --branch`; `python scripts/status.py`; this file.
Use `git log -4 --oneline` for the local evidence commits.

## Outcome and remaining capability gap

This completes the fixed assessment, not the broader full-size resident scheduler.
The implemented optional backend uses host-built finite static topology/window
expansion, device predicates and native NPUGraph replay. Qualified replay has no
per-event host scalar/index decisions, but may compute inactive rows. A general
preallocated device event queue/activity-sparse executor is not implemented.
Independent full-observable/VJP/optimizer gates qualify only recorded finite
scopes; masking alone is never an equivalence argument. Complete-training replay
passed at small scale. Full-size captured replay remains unqualified.

Remaining engineering for that broader objective: reuse/bound peer notifications
or redesign completion, reduce static-expansion/workspace retention, then repeat
independent semantic/gradient/optimizer gates and full-size capacity/performance.
Eight-device12-token notification probes failed before full-size replay. The
12-device Attention OOM had external contention; uncontended fit is unknown.
Do not silently retry unchanged sweeps or call these capabilities delivered.
Baseline/profiling and backend work overlapped; the earlier recommendation to
finish baselines first was not followed in that order.

## Reviewed evidence

- docs/evidence/accelerator-cpu-npu-comparison-20260929.{md,json}: same-source
  CPU/NPU timings, four full-size profiles, work-count/record/source audits,
  original failures, export recoveries and observed shared load.
- docs/evidence/bounded-scheduler-qualification-20260929.{md,json}:84 immutable
  CPU/NPU qualification cells and24 tracked smokes. FP32 unchanged1e-6/1e-5;
  FP16 tiny.004/.02, actual.02/.02; discrete/None exact. Actual465-node/4418-wire
  graph atD8/B1/V17/T3 onCPU/2 devices; eight-device qualification is tiny.
- docs/evidence/bounded-scheduler-capacity-20260929.{md,json}:15 experiments,
  2 successes/13 failures,2 additional12-device gate cases and10 prerequisite
  skips. Full-size Add bounded-eager inference passed FP32/FP16 on8 chips;
  this is not captured replay or a matching token-warmup timing scope.

ROADMAP D1/D2/D6 now record the terminal finite assessment; D3-D5 retain their
small finite qualification boundaries. README, navigation and the support matrix
link these reports. CUDA device execution and new host/version combinations
remain target-machine work; previous build/CPU checks do not establish GPU use.

## Matched baseline and profiling results

Clean consumer source951031e1f7f8b40365ae6ce7f87e7d162a6d5ab2,
consumer hash7b47afb5a2b8f96893ab3584d1b2dff6d59e1f7cf9712d6f2bc71e4a613f3589,
core hash5e342902e64abbc384099c3903317955e2eac2c9bfa599cd1a9ddf9eb9fbf439.
D2048/B512/V50304/T12/seed7. Add9,468,020,899 parameters (historical8.8B label);
Attention17,269,426,339. CPU Read/control/ranking/events useFP32 in timing cells.
CPU node/head workers56 Add,160 Attention, forward ATen/BLAS1 and backward/
optimizer16; NPU host workers16/head1, inference2/4 chips, training4/9.

24 planned timing cells:22 launched,21 completed,1 CPU Attention training timeout;
its later2 repeats skipped. Three-process medians of measured means,ms/sample-token:
Add infer CPU7.387522/NPU2 17.858098; Attention infer CPU19.531847/NPU4 56.012343;
Add train CPU78.793172/NPU4 47.932888; NPU9 Attention train128.275286.
NPU/CPU inference time ratios2.4173/2.8677; Add CPU/NPU training time ratio1.6438.
CPU Attention completed4018.027822s warmup then timed out at7200s process bound;
no measured update, no valid steady training ratio. Matched measured work counts
and all3 Attention warmup work counts agree; this is not full-size tensor parity.

All4 profile collections passed. Required operator/API exports cover exact
allocations. CPU32 Add token187854 tasks; CPU32 Attention token828056;
all32 Attention token1004756;9-chip CPU32 Attention backward4085290.
Only all32 has AiCPU tasks:197 INT64 Sort,2.0189% of summed task duration,
not wall time. Other3 traces contain no AiCPU task type. Many small vector
operations, copies, scalar checks, launches and barriers remain. Task sums and
operator interval unions are not hardware utilization or a wall decomposition.
All32 still uses host C++ dispatch; it is not the bounded captured backend.

## Terminal records and exact locations

TASK_ROOT=/mi/data2T/zlong/tide-device-scheduler; repository links are
artifacts/device-scheduler-JOB. Sources/builds/launchers/plans and raw failures
are immutable. OLD=/mi/data2T/zlong/tide-npu-performance is sealed.

- matched-baseline03 completed15:08:53UTC. Its outer success means the finite
  inventory ended; CPU training and two export failures retain their own status.
- observe-resources01/02 are terminal.957 overlapping samples cover10:38:19–
  15:09:22UTC, not earlier history or continuous exclusivity. Third NPU Attention
  timing has77 overlapping-observer samples of outside use on its allocation.
- finalize-analysis01 passed: reports/matched-analysis-final01/analysis.json and
  reports/audit-completed-final01/audit.json. Audit:67 experiments (53 completed,
  14 failed),118 passed prerequisite cases/18 groups, all67 Trackio projections
  matched. Original incomplete exports were explicitly skipped by this analyzer.
- export-profile-all32-repair01: export passed, old single-CSV analysis failed;
  original export had hit8GiB disk bound. Separate reanalyze-profile-all32-01
  passed on reporting-only source381d44a and verified4-device coverage.
- export-profile-backward-repair01 passed15:22:46UTC after original8GiB export
  failure. New hashed raw copy,48GiB output/64GiB RSS/1800s export bounds,
  CPUs310..317. Five operator CSV slices and9-device coverage passed. Directory
  about19.985GiB. No NPU workload was rerun for either recovery.
- reports/delivery-evidence03/evidence.json (also linked from artifacts) is the
  reviewed pack; copied byte-for-byte into the new evidence JSON. Its SHA256 is
  6514e30dc61c90c14bec7c4658523aca81fc088a128efd26ae74fe7b39053cc0.
  pack-delivery03.py was run with --base TASK_ROOT --out TASK_ROOT/reports/
  delivery-evidence03 --repair TASK_ROOT/runs/export-profile-all32-repair01
  --repair TASK_ROOT/runs/export-profile-backward-repair01
  --reanalyze TASK_ROOT/runs/reanalyze-profile-all32-01.
  It verifies original/copied raw hashes, exact device coverage and immutable
  observer snapshots. Private path scan found no leaked site paths in the JSON.

Vendor parent-directory/Cluster Tuning warnings remain in the evidence. Accepted
coverage is required operator/API data, not successful execution of every tuner.
Reporting-only381d44a adds streamed consecutive CANN CSV slices, rejecting gaps,
duplicates and mixed exports.10 directed CPU tests passed. Test-import-only follow-upd0c23ad also passed
all10 independently with PYTHONPATH=python; no runtime changes or new full public-
regression claim. Runtime/checker source remainsf8648f7, bounded
implementationff251745; prior evidence commits158dc76/4383bc1 are preserved.

Environment: module libtorch-npu/2.10.0-cann9.0.0, standalone SDK under
/opt/software/libtorch-npu; Python /opt/miniconda/envs/
ascend900-train-full-torch-npu-2.10.0-py311/bin/python; local Trackio0.35.0 under
TASK_ROOT/trackio (viewer /home/zlong/venvs/trackio/bin/python). No dashboard.
320 CPUs,8 NUMA nodes;16 Ascend910_9392/A3 compute chips,64GiB/chip.
Always refresh shared-device availability before any future authorized workload.
