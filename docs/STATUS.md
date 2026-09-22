# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: graph-execution-foundation.
Composite checkpoint implementation is ready to commit after its passed directed gate.
Next: reject invalid Python coordinate types, then qualify a clean combined commit.
No push; no sub-agents. LH/fractal-latcarf/ObsidianVault are read-only.
Run git status and scripts/status.py on re-entry, then follow this file.

## Verified state

Graph v13 / checkpoint v5. Current reports distinguish each tested source:

| Scope | Clean source | Result / evidence |
| --- | --- | --- |
| CPU execution, training and single-PDG resume | d233429cd5807614869214dcea21d9492e309fd1 | 4901 tests / 584.15s; evidence/single-graph-training.md |
| Original LH bounded single-PDG inference | c84abbed7c9cfce6d4c6a44f6b7ae5f9aab465d3 | All seven stages passed; evidence/lh-single-graph.md |
| Atomic value-checkpoint publication | 00bbf78fb99e410d46b4d7e36e37266eb3d8cf9e | 88 directed clean tests / 6.82s; evidence/checkpoint-io.md |
| First native streaming performance pilot | aee0da48e4c6661d7a75fd97ca39ddaba654ee4d | 16 cases / 80 events; evidence/m8-streaming-pilot.md |

Training adds 940 cases covering independent roots and None/zero VJPs,
HARD/SOFTP/HST, six profiles, selected clear, shared physical-phase parameters,
serial/parallel/packed/cursor, three truncated optimizer updates and two value
save/restores inside partial windows. Helpers: single_graph_{training,roots,
optimizer}.py. Contract: single-graph-training.md. It is Tide training; LH is an
inference reference. The complete single graph needs no invented readout ledger.

LH assertions-on FP64: 180 configurations / 4950 cuts, 24 known unavailable
original diagnostic configurations. Other three dtype/assertion variants: 204
configurations / 5610 cuts each. Independent Python: 48 fixtures / 28680 original
events / 660 single-PDG cuts. Projection is bounded equal-width, fixed inference
weights and homogeneous profiles; readout view omits the two-clock adapter's
occurrence ledger explicitly. Contracts/navigation were updated after success.

## Next action and latest directed gate

No active jobs. Commit the token-application checkpoint, then fix the independent
Python integer-coordinate validation defect before one combined clean full CPU
qualification. Reproducer: artifacts/coordinate-types-probe/probe.json. Python
accepts time=0.5 and drops the event while updating the ledger; native rejects it.
Both accept bool coordinates. Reuse history.int64 for strict non-bool int64
validation of external/window metadata and imported continuation owners/counters.
Apply this before reference/frontier execution, native conversion and cursor
advance. Cursor checks must scale with incoming inputs, not all held state.
Test rejection without mutation, retry and checkpoint preflight; keep existing
counter-overflow behavior. No C++ kernel change is needed for Python types.

Composite implementation: token_checkpoint.py, checkpoint_values.py, v5 codec
refactor, application tests/controller helper extraction and docs. Contract:
token-application-checkpoint.md. Common Emit mode/zeta bind application identity.
Whole-bundle preflight validates named cross-graph aliases, optimizer layout,
both clocks/continuations/ledgers and partial buffers before changing live owners.
The bundle persists a complete two-clock value boundary; RNG/data cursors and
standalone C++ persistence remain separate backlog obligations.

Directed gate tide-foundation-token-bundle-dev-20260922-0038 completed:
1398 passed / 342.38s, finished 2026-09-22T00:52:00Z. Service inactive/dead,
MainPID 0, Result=success, exit 0. Source d3ab673 plus the archived uncommitted
implementation. artifacts/token-bundle-dev-20260922-0038/ holds status.json,
development.json, task.log, source.tar.gz and post-run-audit.json. All 335 archive
files matched the unchanged worktree; tree SHA256
38226eef1ea1fc4c342853d879cde06840a05133177a197a8d6b49e2f703bbc4.
This is development evidence; clean qualification remains pending.

Exact directed command from /home/zlong/llm/graph-execution-foundation:

```sh
/home/zlong/anaconda3/bin/python scripts/job.py --output-dir artifacts/token-bundle-dev-20260922-0038 -- /home/zlong/anaconda3/bin/python scripts/develop.py --output-dir artifacts/token-bundle-dev-20260922-0038 --jobs 2 tests/test_token_checkpoint.py tests/test_checkpoint_io.py tests/test_checkpoint_ownership.py tests/test_checkpoint.py tests/test_single_graph_training.py tests/test_single_graph_optimizer.py tests/test_single_graph_resume.py tests/test_single_graph.py
```

After the coordinate fix and directed tests, commit, create a read-only detached
worktree and run scripts/job.py --output-dir ABS_OUTPUT -- python
scripts/qualify.py --output-dir ABS_OUTPUT --jobs 2 through the usual durable
service. No C++/LH oracle changes require rerunning the original LH full matrix.
ROADMAP remains the only backlog.

## Terminal records and source retention

All units below are inactive/dead, MainPID 0, Result=success, exit 0:

- tide-foundation-single-qualified-20260921-2234, output
  artifacts/single-qualified-20260921-2234/, finished 2026-09-22T00:10:46Z.
  Check status.json, verification/result.json, oracle/result.json,
  oracle-release/result.json, pronounce-release/result.json,
  iocortex-release/result.json, iocortex-python/result.json.
  qualification-audit.json rechecked 513 hashes and exact fixture inventories.
- tide-foundation-training-qualified-20260921-2345, output
  artifacts/training-qualified-20260921-2345/, finished 00:09:41Z.
  status.json and verification/{result.json,tests.log}; all build hashes checked.
- tide-foundation-checkpoint-io-qualified-20260922-0000, output
  artifacts/checkpoint-io-qualified-20260922-0000/, finished 00:00:12Z.
  status.json, development.json, source archive, dispatch.json and task.log.

Directed development gates also passed and retain exact source archives:
single-training-dev-20260921-2335 (826 tests / 178.62s) and
single-resume-dev-20260921-2340 (500 tests / 148.65s). Every archived source file
was compared to the unchanged working tree after exit before committing.
Two finished qualification worktrees/builds were removed after dry run, clean
source/ancestor checks and terminal audits; their commits and all cited logs,
fixtures, manifests and LH snapshot remain retained. Cleanup manifest:
/var/tmp/zlong-graph-execution-foundation/qualification-worktree-cleanup.json.
Seven obsolete draft files were removed; no other old artifacts were eligible.

## Numerical boundaries and failure reproducers

AdamW training/resume uses epsilon 1e-5 explicitly. Default 1e-8 FP32 packed
attention amplified tiny gradients beyond the strict parameter tolerance;
artifacts/single-training-adamw-fp32-repro/ retains the failure and diagnosis.
No tolerance was widened. artifacts/single-training-initial-buffer-repro/
retains an invalid test assumption about a missing second-window output.
artifacts/checkpoint-partial-write-repro/ retains the old ENOSPC failure that
published an invalid 18-byte final file; the atomic writer repairs this.
Keep earlier cited single-python-dev-20260921-2050, single-dev-20260921-2059
and single-dev-20260921-2054 failures; never relabel historical failures.
Packed first-order VJPs use semantic replay; higher-order AD is unclaimed.

## Runtime and storage

CPU aarch64, /home/zlong/anaconda3/bin/python, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. TORCH_DEVICE_BACKEND_AUTOLOAD=0;
OMP/OpenBLAS=1, two build jobs, Nice=10/background.slice. FP64 atol/rtol
1e-10/1e-8; FP32 1e-6/1e-5; routes/identities exact. Keep builds isolated.
LH snapshot artifacts/lh-source-20260921-1428 has 69 files, identity
ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f,
original HEAD 5fd237d40c9880ccb6e511e4bf20799c7022fd1e plus actual dirty hashes.
Original FP64 active-softmax diagnostics use an FP32 denominator; assertions-off
fills numerical coverage while retaining ordinary C++ asserts. LH is unchanged.

Shared storage filled twice. /home/zlong/llm/graph-execution-foundation now
symlinks to /var/tmp/zlong-graph-execution-foundation/repository (including .git);
build/artifacts symlink into that local parent. Repository migration checked
2027 file hashes and both source identities before removing the duplicate.
Correctness qualification was SIGSTOP/SIGCONT paused at 23:14:08.216657Z to
23:14:11.782289Z on Sep 21 for that switch, then resumed and fully passed.
Relocation inventories are in the local parent; main CMake rebuilt successfully
at the new physical path. Git connectivity check passed after worktree cleanup.
Check capacity before large writes. Use fsynced atomic handoff writes and verify
them. No tracked source/document exceeds 500 lines in the final size audit.
