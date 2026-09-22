# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: graph-execution-foundation.
The standalone C++ named-owner/optimizer implementation is committed at
5c440e1 and its frozen clean qualification passed. The reviewed report is
evidence/cpp-optimizer-ownership.md. The prior token-bundle, integer-coordinate
and durable-record qualification remains complete. Native value persistence is
the next bounded increment.
No push; no sub-agents. LH/fractal-latcarf/ObsidianVault are read-only.
Run git status and scripts/status.py on re-entry, then follow this file.

## Verified state

Graph v13 / single-graph checkpoint v5 / tide-token-application-v1.
Source and scope are separate for each report:

| Scope | Clean source | Result / evidence |
| --- | --- | --- |
| Complete CPU regression, two-clock checkpoint and strict coordinates | 69ca37900e9c10d3fca95570ea1ebca8f9079f46 | 6233 tests / 665.45s; evidence/token-checkpoint-coordinates.md |
| Durable status publication and damaged-record re-entry | 3604ec002722e701c74bd13e6f681b88ada14199 | 16 tests / 0.44s; evidence/durable-records.md |
| Bounded two-clock/single-PDG training and single-graph resume | d233429cd5807614869214dcea21d9492e309fd1 | 4901 tests / 584.15s; evidence/single-graph-training.md |
| Original LH bounded single-PDG inference | c84abbed7c9cfce6d4c6a44f6b7ae5f9aab465d3 | All seven stages passed; evidence/lh-single-graph.md |
| Atomic value-checkpoint publication | 00bbf78fb99e410d46b4d7e36e37266eb3d8cf9e | 88 directed clean tests; evidence/checkpoint-io.md |
| First native streaming performance pilot | aee0da48e4c6661d7a75fd97ca39ddaba654ee4d | 16 cases / 80 events; evidence/m8-streaming-pilot.md |
| Standalone C++ named ownership and SGD/AdamW parity | 5c440e1648070c74eb25955893e10bffd3091d5c | clean frozen build; 338 directed tests plus FP64/FP32 executable checks; evidence/cpp-optimizer-ownership.md |

The bundle preserves two complete continuations, real occurrence ledgers,
unfinished token buffers, named cross-graph aliases and optimizer state. 192
three-update/two-restore comparisons cover Python/native serial/parallel/packed,
Add/all-softmax, clear, HARD/SOFTP/HST, SGD/AdamW and both dtypes. Corruption
preflight leaves both live owners unchanged. It is not a training controller or
standalone C++ file format. Native execution has no Python callbacks; optimizer
and persistence ownership in these gates is Python.

Strict Python int64 checks reject bool/float/overflow before scheduling, native
conversion or checkpoint restoration. Cursor import checks complete metadata
once; advance checks new inputs only. Source 69ca379 did not include the later
durable-record tooling, whose independent clean 16-case gate is listed above.
Both results have source/terminal audits. No C++/original-LH oracle changed.

## Next action

The standalone C++ owner registry and independent LibTorch SGD/AdamW
updates are implemented and cleanly qualified at 5c440e1; see
evidence/cpp-optimizer-ownership.md. The next increment is native value
serialization and resume, with an explicit schema/identity and transactional
preflight. The standalone C++ SettleGraph construction frontend is still a
separate interface obligation; do not rerun the original LH full oracle unless
its relevant code/mapping changes.

Later M8 Attention/SSM/prefill/training measurements require new declared fixed
workloads; the current EMA pilot does not certify those cases. Wider module
imports and Delta chunk optimization remain in ROADMAP, the only backlog.

## Latest terminal jobs and retained source

All listed units are inactive/dead, MainPID 0, Result=success, exit 0:

- tide-foundation-named-optimizer-qualified-20260922-1010, finished
  2026-09-22T02:17:44Z. The clean frozen worktree at commit 5c440e1 passed
  both standalone C++ FP64/FP32 checks and 338 directed tests in 64.18s.
  Output is artifacts/named-optimizer-qualified-20260922-1010/; source audit
  reports no dirty files. Build and worktree are retained under
  /var/tmp/zlong-graph-execution-foundation/qualification/.
- tide-foundation-named-optimizer-dev-20260922-0940, finished
  2026-09-22T02:06:34Z. Its dirty-source build and directed gate passed:
  both standalone C++ FP64/FP32 checks and 338 native/PyTorch, Python
  ownership and single-graph optimizer tests. Output is
  artifacts/named-optimizer-dev-20260922-0940/; the independent build is
  /var/tmp/zlong-graph-execution-foundation/build-named-optimizer. A previous
  path-correctness interruption is retained in
  artifacts/named-optimizer-dev-20260922-0930/ and is not a pass result.
- tide-foundation-token-coordinates-qualified-20260922-0106, finished
  2026-09-22T01:21:19Z. Output artifacts/token-coordinates-qualified-20260922-0106/:
  status.json, task.log, verification/{result.json,tests.log}, dispatch.json,
  qualification-audit.json. All 339 frozen source files and eight binary hashes
  checked. Worktree retained read-only with its own build:
  /var/tmp/zlong-graph-execution-foundation/qualification/token-coordinates-20260922-0106.
  Exact command there (Python=/home/zlong/anaconda3/bin/python):
  scripts/job.py --output-dir ABS_OUTPUT -- python scripts/qualify.py
  --output-dir ABS_OUTPUT --jobs 2. ABS_OUTPUT is the absolute output above.
- tide-foundation-records-qualified-20260922-0115, finished 01:13:20Z.
  artifacts/records-qualified-20260922-0115/{status.json,task.log,post-run-audit.json}.
  Clean main source stayed unchanged during scripts/job.py --output-dir ABS_OUTPUT
  -- python -m pytest tests/test_durable_records.py -q --dtype both.
  All 341 tracked files matched git archive after exit.
- tide-foundation-token-bundle-dev-20260922-0038: 1398 passed / 342.38s,
  finished 00:52:00Z. Its archived 335 dirty-source files were compared to the
  unchanged main tree before c08105e. Artifacts and exact command are retained
  in artifacts/token-bundle-dev-20260922-0038/ and the latest evidence report.

Earlier qualifications, fixtures, snapshots and failure logs remain retained.
The old training/LH qualification worktrees were audited and removed; their
commits and evidence remain. Cleanup manifest is
/var/tmp/zlong-graph-execution-foundation/qualification-worktree-cleanup.json.
The dry-run artifact cleaner found no eligible old records. Do not delete cited
artifacts or the latest frozen worktree without a fresh terminal/source audit.

## Numerical boundaries and retained failures

Training uses AdamW epsilon 1e-5 explicitly. Default 1e-8 FP32 packed attention
amplified tiny gradients beyond the unchanged strict tolerance; keep
artifacts/single-training-adamw-fp32-repro/. Higher-order AD is unclaimed; packed
first-order VJPs use semantic replay and have separately measured cost obligations.
Original LH inference remains bounded to equal width, fixed inference weights
and specified homogeneous profiles. Its single-PDG readout view omits the
adapter-only occurrence ledger; a token index is not an occurrence count.

New repaired failures are retained in artifacts/coordinate-types-probe/,
coordinate-test-key-collision-repro/ and durable-records-postmortem-repro/.
The last contains an initial strict-loader misclassification of an explicitly
recorded old failed launch with observed time but no workload start. The original
artifacts/single-dev-20260921-2054/status.json remains failed and unchanged.
Keep earlier cited single-python-dev-20260921-2050, single-dev-20260921-2059,
single-training-initial-buffer-repro/ and checkpoint-partial-write-repro/ too.
Never relabel historical failures when a later run passes.

## Runtime and storage

CPU aarch64, /home/zlong/anaconda3/bin/python, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. TORCH_DEVICE_BACKEND_AUTOLOAD=0;
OMP/OpenBLAS=1, two build jobs, Nice=10/background.slice. FP64 atol/rtol
1e-10/1e-8; FP32 1e-6/1e-5; routes/identities exact. Keep builds isolated.
LH snapshot artifacts/lh-source-20260921-1428 has 69 files, identity
ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f,
original HEAD 5fd237d40c9880ccb6e511e4bf20799c7022fd1e plus actual dirty hashes.

Shared storage filled twice. The stable repository path
/home/zlong/llm/graph-execution-foundation symlinks to
/var/tmp/zlong-graph-execution-foundation/repository (including .git); build and
artifacts use that local parent too. Migration inventories/cleanup records remain
there. Check capacity before large writes. Use scripts/durable_records.py for
fsynced atomic handoff writes, then read back and verify. A directory-fsync error
may occur after complete publication; never call a failed write successful.
No tracked source/document exceeds 500 lines; relative Markdown links were checked.
