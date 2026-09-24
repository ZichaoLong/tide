# Current handoff

Updated: 2026-09-25 (Asia/Shanghai). Branch: graph-execution-foundation.
**Active: authorized cross-family streaming/prefill policy extension (E1-E5).**
The first clean qualification attempt is terminal failed; the second clean
qualification job passed and its evidence is recorded below.
Push is authorized for this continuation; no subagents/reference writes. STATUS owns handoff,
ROADMAP owns backlog, semantics.md owns the contract. User explicitly approved
execution of the five-stage plan; the previous no-queued-task notice is obsolete.

## Current work and next action

Re-entry checked clean worktree at extension start; no live durable records.
CPU operator probe passed on aarch64/Torch2.10.0+cpu. Static portability audit:
12 review warnings, all in existing runtime/test boundaries; no detected errors.
Implementation committed as957b004: Python isolated Full/Aggregate VJPs;
independent packed streaming and chain/diamond/Settle layered block schedules;
native Settle streaming frontend; block packed=False now controls Full/Agg;
versioned foundation-v2 policy/client/config/phase-recording entry. Contract:
docs/cross-family-policies.md. Python directed gates:102+112 passed,4 native
checks deselected pending rebuild. Raw pytest basetemps under artifacts/
cross-family-python-dev-{a,b}; no formal speed/qualification claim yet.

Development job-a:terminal exit1,898 passed/124 import-typo failures.
Development job-b:terminal exit1,1629 passed/3 FP32 S03 squared-loss failures.
All failures and source archives retained in artifacts/cross-family-dev-20260924-{a,b}/.
The new v2 test accidentally squared the assembled objective via a raw-root VJP
helper. It now differentiates the declared loss directly and separately roots
raw content/output. No tolerance changed; corrected v2 gate130 passed locally.
Diagnostic retained at artifacts/cross-family-loss-reproducer-20260924/;
this finite contract does not certify every rescaling of cancellation-sensitive
FP32 derivatives. Original scalar/batched near-zero AdamW tests pass.

Development job-c is terminal:1632 passed in311.12s, exit0, MainPID0.
Archive/source identities are in artifacts/cross-family-dev-20260924-c/.
The later training-observation artifact/CLI check passed7 tests in60.71s;
raw output: artifacts/cross-family-records-dev-a/. Training audit snapshots are
final-window owner values/gradients/aliases, not resumable checkpoints.
Development checkout /var/tmp/zlong-graph-execution-foundation/development/cross-family-20260924-a
was reused only between terminated attempts; per-job source archives are immutable.

The first clean qualification attempt is terminal failed, not passed. The
second attempt ran as unit `tide-cross-family-qualification-20260925-a`
in `background.slice`, Nice=10, from frozen source `eaa15c6`:
/var/tmp/zlong-graph-execution-foundation/qualification/cross-family-20260925.
It uses driver copy
/var/tmp/zlong-graph-execution-foundation/qualification/cross-family-20260925-driver.py
(SHA256 0abed3b859776457f7dafbf868b82d97378e1a44c62069f9fd8795f73366c546).
The prior failed attempt tested frozen source957b004:
/var/tmp/zlong-graph-execution-foundation/qualification/cross-family-20260924.
Driver: artifacts/cross-family-drivers-20260924/qualification.py.
Command: Python driver OUTPUT RELOCATED; scripts/job.py records exact argv/cwd.
Current output/status/log: artifacts/cross-family-qualification-20260925-a/{status.json,task.log,qualification.json}
and artifacts/cross-family-qualification-20260925-a-launch.log.
Current relocated target: /var/tmp/zlong-graph-execution-foundation/relocated/cross-family-20260925.
The driver ran independent full CPU qualification, relocated export/rebuild and
all60 v2 smoke variants; no unchanged original-LH oracle rerun. The terminal
result is exit 0: **8,577/8,577 CPU tests passed; relocated smoke 60/60 passed
with 0 failures; standalone Settle FP64 and FP32 passed.** Inspect with
`systemctl --user show tide-cross-family-qualification-20260925-a` and the
status/log/qualification artifacts. The unit is now inactive; preserve its
outputs and frozen source.

The prior 109-step clean build and standalone loader audit completed; its full
CPU gate failed with 8,547 passed and 30 failures, all in the native-unpacked
replay-counter contract. Preserve that failure record; it is not relabeled as
a pass.

The contract correction is present in the current worktree and its focused
isolated-root gate passes 252 tests for FP32/FP64. The new clean qualification
now passes; the old failure remains retained as historical evidence.

Next: audit the passing qualification, then run fixed v2 medium and
resource-staged large assessments without overlapping build/test load. Commit
reviewed evidence separately. No formal speed claim exists yet.
The performance driver is prepared at artifacts/cross-family-drivers-20260924/performance.py;
the running unit is `tide-cross-family-performance-20260925-a` in
`background.slice`, Nice=10. It runs from passing frozen source `eaa15c6` at
/var/tmp/zlong-graph-execution-foundation/qualification/cross-family-20260925,
using driver copy
/var/tmp/zlong-graph-execution-foundation/qualification/cross-family-20260925-performance.py
(SHA256 8b8cbedf3e976c4f5745d0d339e34c43faf9d13589181e3076e3bb9155bbb3ee).
It writes unique medium/large output under
artifacts/cross-family-performance-20260925-a and launch output under
artifacts/cross-family-performance-20260925-a-launch.log. Resource discovery
resolved a combined budget of 160 CPUs and about 841 GiB effective memory;
the fixed driver uses medium 4 workers/1 thread/180 s and large dynamic
workers/4 threads/900 s. Inspect the unit and assessment.json; no performance
result is available until the unit reaches a terminal state.
Prepared next driver: artifacts/cross-family-drivers-20260924/performance.py
(OUTPUT QUALIFICATION), then review.py (FROZEN_SOURCE QUALIFICATION ASSESSMENT).
Medium4 workers/1 thread,3 processes,180s bound; large32 workers/4 threads
(with dynamic budget reduction),1 process,900s bound including setup/warmups.
Do not end at submission. No formal speed claim yet.

Read ROADMAP's "Active extension" for finite coverage and completion. Preserve
foundation-v1 and historical results. Native local kernels are largely shared;
do not infer actual sequence execution or throughput from option acceptance.
Native SettleExecutor selects Frontier or Streaming over its native encoding.
Frontier/Specialized reject coordinator phase profiling. Attention policies
configure kernels; projection layout configures Model, not core.Options.
Do not blindly forward kernel settings into native Settle core.Options.

Key files: scripts/foundation_{execute,workloads,measure,worker,control}.py,
python/tidegraph/{blocks,full,aggregate,specialized,frontier,settle,native}.py,
cpp/src/{block,block_prepare,specialized,frontier,settle}.cpp. Keep schedules
independent and serial/replay oracles. Checkpoint identities exclude execution
policies. Preserve None/connected zero and per-event normalization backward.

## Retained completed evidence

- Six-class finite CPU acceptance: evidence/foundation-final.md (7741 tests,
  17 relocated smoke variants, 36 fresh-process checkpoint trajectories).
- Optional native Full VJP: evidence/full-batched-autograd.md.
- Optional native Aggregate VJP: evidence/aggregate-batched-autograd.md;
  552 selected clean-build tests, all25 experiment records terminal/exit0.
  Wide Add D2048/B512/56 CPUs: replay11.61919 -> batched8.55906ms/sample-token
  median with Full batched (3 processes each); forward with grad only.
  Small actual training Add1.44187->1.04816s, Attention5.47745->4.98471s.
  No new LH timing. Full/Aggregate defaults remain replay; State/Read replay remains.
- Raw evidence: artifacts/aggregate-vjp-comparison-20260924-b/ and
  artifacts/aggregate-vjp-qualification-20260924-a/. Frozen source:
  /var/tmp/zlong-graph-execution-foundation/qualification/aggregate-vjp-20260924.
- Previous portable kit: artifacts/aggregate-vjp-portable-20260924/cpu-attention-compare.tar.gz.
  SHA256 fa5099e7e4424cc7fc69ab24e2974e24d8273ff7fffb528832d85ab5f0106334.
- Keep prior failures/reproducers and timeout/unlaunched results. Do not restart
  completed acceptance or infer arbitrary-model/backend certification.

## Re-entry commands and resource contract

```bash
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```

Python /home/zlong/anaconda3/bin/python. TORCH_DEVICE_BACKEND_AUTOLOAD=0;
correctness ATen/OMP/BLAS1, build2. Half effective CPU and half dynamically
available/effective memory across the task; at most8 Ascend cards only for a
separate authorized backend extension. Main filesystem has about27GiB free at
start: check actual build/artifact targets before large writes. Long jobs use
frozen inputs, independent builds, background.slice, Nice10, durable records.
Implementation and reviewed evidence commits are separate. No live source edits.
~/llm/lh, ~/llm/fractal-latcarf and /home/zlong/ObsidianVault remain read-only.
