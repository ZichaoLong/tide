# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: graph-execution-foundation.
No push; no sub-agents. LH/fractal-latcarf/ObsidianVault remain read-only.

## Next action and training increment

Tide two-clock versus single-PDG training passed its directed development gate:
artifacts/single-training-dev-20260921-2335/, 826 tests in 178.62s, including
700 new training cases and 126 prior inference/guard cases. Both durable records
passed; unit tide-foundation-single-training-dev-20260921-2335 is inactive/dead,
MainPID 0, exit 0. Frozen source archive was compared byte-for-byte with the
unchanged tree after exit. The relocated main CMake build reconfigured and
rebuilt successfully; no special cache repair was needed.

Contract: single-graph-training.md; helpers single_graph_{training,roots}.py.
Covers isolated roots, parameter aliases, HARD/SOFTP/HST, six profiles, clear,
serial/parallel/packed/cursor and truncated SGD/AdamW updates. AdamW explicitly
uses epsilon 1e-5. Default-epsilon FP32 trajectory failure remains retained at
artifacts/single-training-adamw-fp32-repro/; strict tolerances were not changed.
Earlier invalid buffer-presence test assumption is archived at
artifacts/single-training-initial-buffer-repro/.

Checkpoint extension passed 500 optimizer/resume tests in 148.65s:
artifacts/single-resume-dev-20260921-2340/, both records passed, unit inactive/dead,
MainPID 0, exit 0. Its source archive matched every working source file after
termination. Two complete single-PDG v5 saves/restores occur inside nonempty
partial readout windows, preserving aliases, detached state/pending payloads,
weights and optimizer updates. No composite two-clock/controller serialization
or standalone C++ optimizer qualification follows from this test.

## Active clean training qualification

Source d233429cd5807614869214dcea21d9492e309fd1 is frozen, clean and read-only at
/var/tmp/zlong-graph-execution-foundation/training-qualification-20260921-2345,
with its own build. Unit tide-foundation-training-qualified-20260921-2345 is
confirmed active/running, MainPID 618952, background.slice, Transient=yes,
control group outside focus.service. No full result yet. Command:

```sh
/home/zlong/anaconda3/bin/python scripts/job.py --output-dir /var/tmp/zlong-graph-execution-foundation/artifacts/training-qualified-20260921-2345 -- /home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir /var/tmp/zlong-graph-execution-foundation/artifacts/training-qualified-20260921-2345 --jobs 2
```

Inspect that unit's ActiveState/SubState/MainPID/Result/ExecMainStatus and
artifacts/training-qualified-20260921-2345/{status.json,task.log} plus
verification/{result.json,tests.log}. Stop only if necessary with
systemctl --user stop tide-foundation-training-qualified-20260921-2345.
Never edit the frozen worktree/build. Original LH runs independently below.

## Current main work and next actions

Atomic checkpoint repair 00bbf78fb99e410d46b4d7e36e37266eb3d8cf9e is qualified by
88 clean directed tests (6.82s), including ownership, fault injection and four
native resume/update cases. Unit tide-foundation-checkpoint-io-qualified-
20260922-0000 ended inactive/dead, MainPID 0, exit 0 at 00:00:12Z. Both records
in artifacts/checkpoint-io-qualified-20260922-0000/ passed; archive/tree match.
Report: evidence/checkpoint-io.md. Prior ENOSPC failure is retained in
artifacts/checkpoint-partial-write-repro/. v5/load are unchanged.
Reviewed redundant training/IO draft directories were removed (7 files, about
20 KiB); source archives, failures and active jobs are retained. General artifact
cleanup dry run found no other eligible entries.

Next finish the two independent clean qualifications, inspect terminal status,
source/build/fixture inventories and counts, then commit their evidence and
replace pending LH/training claims. Do not edit either frozen worktree/build.
Composite two-clock and standalone C++ optimizer ownership remain in ROADMAP.

## Active immutable qualification

Unit: tide-foundation-single-qualified-20260921-2234.
Source: c84abbed7c9cfce6d4c6a44f6b7ae5f9aab465d3, clean.
Worktree: /var/tmp/zlong-graph-execution-foundation/qualification-20260921-2234.
Tracked files read-only; its build/ is independent. Do not modify either.
Confirmed active/running, MainPID 254881, background.slice, Transient=yes,
control group outside focus.service. Started 2026-09-21T22:33:32Z.
CPU stage passed 3946 tests in 326.58s, exit 0, at 22:43:32Z;
evidence/checkpoint-ownership.md qualifies only that completed stage.
Original assertions-on Selector/Add/Full/Attention/Pronounce passed both dtypes.
IOCortex FP64 passed 180 cases / 4950 cuts (24 known diagnostic limitations).
Assertions-on FP32 also passed; oracle/result.json passed at 23:23:42Z.
Assertions-off component gates are running; no whole-job result yet.

Exact command, from the worktree:

```sh
/home/zlong/anaconda3/bin/python scripts/job.py --output-dir /var/tmp/zlong-graph-execution-foundation/artifacts/single-qualified-20260921-2234 -- /home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir /var/tmp/zlong-graph-execution-foundation/artifacts/single-qualified-20260921-2234 --jobs 2 --lh-snapshot /var/tmp/zlong-graph-execution-foundation/artifacts/lh-source-20260921-1428
```

Inspect `systemctl --user show tide-foundation-single-qualified-20260921-2234
-p ActiveState -p SubState -p MainPID -p Result -p ExecMainStatus` and
artifacts/single-qualified-20260921-2234/{status.json,task.log},
verification/result.json, oracle/result.json, oracle-release/result.json,
pronounce-release/result.json, iocortex-release/result.json,
iocortex-python/result.json. Stop only if necessary with
`systemctl --user stop tide-foundation-single-qualified-20260921-2234`.

## Completed increments

Graph v13 / checkpoint v5. Single-PDG map: a558b87. Named optimizer ownership:
f900e15, qualified in evidence/checkpoint-ownership.md. Explicit IOCortex
smoke/full: c84abbe; full remains default. The 6-case smoke passed all four
assertion/dtype variants (165 cuts each); no fixture export in smoke scope.

Full single-PDG development gate artifacts/single-lh-dev-20260921-2102/ passed
at 22:24:53Z; unit inactive/dead, MainPID 0, exit 0. Dirty source archive SHA256
8db7e411e2fab3cb6a09fa8719816988020e668a351e6e99df107377b84482f6,
compared byte-for-byte with unchanged source after exit. On/FP64: 180 cases /
4950 cuts, 24 unavailable. Other three variants: 204 cases / 5610 cuts each.
Python: 48 fixtures / 28680 original events / 660 cuts. Development evidence
only; clean qualification above is still running. Contract: lh-single-graph.md.
Readout projection explicitly omits the adapter-only occurrence ledger; it does
not claim complete two-clock continuation equivalence or arbitrary import.

M8 native benchmark and fixed pilot: aee0da48e4c6661d7a75fd97ca39ddaba654ee4d.
Clean 15-test gate artifacts/bench-anchor-20260921-2302/ passed (exit 0).
Pilot artifacts/streaming-pilot-20260921-2300/ passed: 16 cases / 80 events,
all source clean; unit inactive/dead, exit 0, finished 22:58:16Z.
Portable record validator: zero warnings. All Trackio projections explicitly
report degraded (package unavailable). Report: evidence/m8-streaming-pilot.md.
Small shared-weight EMA workload only; other scales/profiles/prefill/training
remain unverified. Shared host load and co-running LH gate limit interpretation.

## Runtime, storage and retention

CPU aarch64, /home/zlong/anaconda3/bin/python, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. TORCH_DEVICE_BACKEND_AUTOLOAD=0;
OMP/OpenBLAS=1, two build jobs, Nice=10, background.slice.
FP64 atol/rtol 1e-10/1e-8; FP32 1e-6/1e-5; routes/identities exact.
Packed first-order VJPs use semantic replay; higher-order AD is unclaimed.
LH snapshot artifacts/lh-source-20260921-1428: 69 files, identity
ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f,
original HEAD 5fd237d40c9880ccb6e511e4bf20799c7022fd1e plus actual dirty hashes.
Original FP64 active-softmax diagnostics use an FP32 denominator; assertions-off
fills numerical coverage, retaining ordinary C++ asserts. LH is unchanged.

Shared disk filled twice. Original repository path now symlinks to
/var/tmp/zlong-graph-execution-foundation/repository, including its independent
.git; build/artifacts symlink into the same local parent. Migration verified
2027 file hashes; main and frozen worktree HEAD/status stayed clean. Qualification
was SIGSTOP/SIGCONT paused 2026-09-21T23:14:08.216657Z to 23:14:11.782289Z
for the switch, then confirmed running. This is a correctness gate, not timing.
Verified duplicate shared-volume repository removed after dry run and recheck.
Inventory, source/target and cleanup record: ../repository-relocation.json under
the local parent. Do not repair the live worktree: its gitdir remains valid via
the original path. Check free space before large writes. Main CMake cache may
need relocation handling; never alter the qualification build.

Keep cited artifacts and failed reproducers. In particular single-python-dev-
20260921-2050 (empty windows), single-dev-20260921-2059 (derived norm comparison)
and single-dev-20260921-2054 (disk-full pre-launch failure) remain honest failed
records. Earlier build/artifact relocation manifests are in the local parent.
Use atomic fsynced handoff writes with read-back; no duplicate status diaries.
