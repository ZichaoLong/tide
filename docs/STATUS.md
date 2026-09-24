# Current handoff

Updated: 2026-09-25 (Asia/Shanghai). Branch: graph-execution-foundation.
**Active: authorized cross-family streaming/prefill policy extension (E1-E5).**
The first clean qualification attempt is terminal failed; no live project job.
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

The first clean qualification attempt is terminal failed, not passed. Unit:
tide-cross-family-qualification-20260924-a, previously in background.slice, Nice10.
It tested frozen source957b004:
/var/tmp/zlong-graph-execution-foundation/qualification/cross-family-20260924.
Driver: artifacts/cross-family-drivers-20260924/qualification.py.
Command: Python driver OUTPUT RELOCATED; scripts/job.py records exact argv/cwd.
Output/status/log: artifacts/cross-family-qualification-20260924-a/{status.json,task.log,qualification.json}.
Relocated target: /var/tmp/zlong-graph-execution-foundation/relocated/cross-family-20260924.
Driver runs independent full CPU qualification, relocated export/rebuild and
all60 v2 smoke variants; no unchanged original-LH oracle rerun. The109-step clean
build completed; the full CPU gate failed with 8,547 passed and 30 failures,
all in the native-unpacked replay-counter contract. Standalone loader audit
resolved all libraries and found no Python runtime dependency. Preserve this
failure record; it is not relabeled as a pass.

The contract correction is present in the current worktree and its focused
isolated-root gate passes 252 tests for FP32/FP64. It still needs a new clean
frozen qualification; do not use the old frozen source for a passing claim.

Next: establish a new frozen source from the current clean commit and run the
full qualification plus relocated export/rebuild and v2 smoke. Audit actual
paths, training observations, records and terminal jobs. Then run fixed v2
medium and resource-staged large assessments without overlapping build/test
load; commit reviewed evidence separately.
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
