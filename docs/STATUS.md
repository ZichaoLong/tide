# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: graph-execution-foundation.

## Verified scope and next action

Clock implementation bd933e0980a2076bace5c6ffbb09da61bc400593 passed 3745 tests
in 300.89s; evidence/state-clocks.md (evidence commit 6aa0963). Graph v13 /
checkpoint v4 at that source; working checkpoint v5. SourceDomain evidence: evidence/source-domains.md.
Latest clean original-LH qualification remains 776fc2e597dac872f4204ac3a09966db39ab74c0,
evidence/lh-iocortex.md. The single-PDG implementation below passed its complete
development gate; clean full qualification and reviewed evidence are next.

## Active clean qualification

Source c84abbed7c9cfce6d4c6a44f6b7ae5f9aab465d3 is frozen at
/var/tmp/zlong-graph-execution-foundation/qualification-20260921-2234.
Tracked files are read-only; git status is clean; build/ belongs only to that
worktree. Unit tide-foundation-single-qualified-20260921-2234 was confirmed
active/running, MainPID 254881, background.slice, Transient=yes and a control
group outside focus.service. Started 2026-09-21T22:33:32Z. CPU stage passed 3946 tests in 326.58s, clean,
exit 0, finished 2026-09-21T22:43:32Z; evidence/checkpoint-ownership.md qualifies
that completed stage. Original LH stages are still running; no whole-job result.

Working directory is the frozen worktree. Exact command:
/home/zlong/anaconda3/bin/python scripts/job.py --output-dir
/var/tmp/zlong-graph-execution-foundation/artifacts/single-qualified-20260921-2234
-- /home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir
/var/tmp/zlong-graph-execution-foundation/artifacts/single-qualified-20260921-2234
--jobs 2 --lh-snapshot
/var/tmp/zlong-graph-execution-foundation/artifacts/lh-source-20260921-1428.

Main checkout artifacts/single-qualified-20260921-2234/ resolves to those logs.
Inspect `systemctl --user show tide-foundation-single-qualified-20260921-2234`
with -p ActiveState -p SubState -p MainPID -p Result -p ExecMainStatus, then
status.json, verification/result.json, oracle/result.json,
oracle-release/result.json, pronounce-release/result.json,
iocortex-release/result.json and iocortex-python/result.json.
Stop only if needed: systemctl --user stop tide-foundation-single-qualified-20260921-2234.
Do not edit the frozen worktree/build or change the immutable LH snapshot.
Main development can proceed independently. Commit evidence only after terminal
success and verified source/clean-state/inventories/counts.

## Current implementation and next work

Single-PDG implementation: a558b87. Named optimizer ownership and checkpoint v5:
f900e15; 110 directed tests passed (8.09s). Explicit original IOCortex smoke/full
scope: c84abbe; full remains the default. Actual development gate
artifacts/iocortex-smoke-dev-20260921-2229/ is passed in all three records;
unit inactive/dead, MainPID 0, exit 0, finished 2026-09-21T22:31:30Z. Each of
four dtype/assertion runs passed 6 cases / 165 single-PDG cuts. Both oracle
manifests explicitly say smoke and export no fixtures. Archive/source matched.
Three integrated scope/inventory/rejection tests passed in 0.14s.

Stale module/Add/attention/Pronounce navigation is refreshed; native/Python
snapshot identity-copy cost is explicit. All 15 modified/new files in the temporary
ownership-dev worktree matched main byte-for-byte; that redundant worktree was
removed after inspection. Its implementation is committed in f900e15/c84abbe.
M8 implementation passed its directed development gate and is ready to commit: cpp/bench/{streaming.h,config.cpp,
workload.cpp,compare.cpp,main.cpp,metrics_jsonl_writer.h}, CMakeLists.txt,
scripts/{build,benchmark_streaming,experiment_record}.py,
tests/test_streaming_benchmark.py, docs/streaming-benchmark.md. It uses separate construction,
advance and snapshot; include complete correctness checks outside timing, work
counters, per-repetition latency and process peak memory. Experiment skill read;
Torch interpreter has no Trackio and no site viewer is configured. Keep portable
local metrics and report any best-effort tracking degradation. Do not claim a
speedup or broad workload support without measurements. M6 composite ownership,
more training objectives and further performance work stay in ROADMAP.
No push; no sub-agents. Reference repositories remain read-only.

## Single-PDG development result

Unit tide-foundation-single-lh-dev-20260921-2102 is terminal inactive/dead,
MainPID 0, Result=success, exit 0; finished 2026-09-21T22:24:53Z. All four records
under artifacts/single-lh-dev-20260921-2102/ are passed: status.json,
oracle/result.json, oracle-release/result.json, python/result.json.
Frozen source was 6aa0963 plus archived dirty changes; source.tar.gz SHA256
8db7e411e2fab3cb6a09fa8719816988020e668a351e6e99df107377b84482f6.
Every archived file was compared with the unchanged working tree after exit.

Original assertions on: FP64 180 cases / 4950 single-PDG cuts, 24 known
unavailable diagnostic configurations; FP32 204 cases / 5610 cuts. Assertions
off: FP64 and FP32 each 204 cases / 5610 cuts, filling the FP64 gap. Both
variants exported 48 hashed fixtures. Independent Python: 48 fixtures,
28680 original candidate events, 660 single-PDG cuts. Source/snapshot/library/
fixture identities passed. This is development evidence, not a clean qualification.

Single-graph contract/navigation: lh-single-graph.md. Existing clock/domain
primitives map equal-width fixed-weight homogeneous-profile IOCortex inference;
no runtime/schema change. Every global cut checks full body state/trace/history/
ledger/pending, partial readout buffers, readout state/history and hidden/logits.
The readout projection omits the two-graph adapter's occurrence ledger explicitly.
Python phase-major physical IDs differ from the native edge-major construction.
Replicated slots alias original parameters. There is no whole-model training,
composite checkpoint, arbitrary importer or large-sparse performance claim.

Directed tests: 120 schedule/profile/clear/dtype cases passed (22.95s), six
corrupted-read/route and missing-phase occurrence guards passed (0.49s).
Retain initial failures and source archives:
- artifacts/single-python-dev-20260921-2050/: synthetic empty windows incorrectly
  used original token_inputs; fixed by sealing empty manual/ragged graph windows.
- artifacts/single-dev-20260921-2059/: 96 passed / 24 failed; FP32-derived FP64
  norms incorrectly required FP64 cross-implementation equality. Comparison now
  uses own-candidate FP64 norm plus actual candidate L2 perturbation. Payload
  tolerances and exact routes unchanged. Original failed record stays failed.

## Runtime, storage and retention

CPU aarch64, /home/zlong/anaconda3/bin/python, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. TORCH_DEVICE_BACKEND_AUTOLOAD=0;
OMP/OpenBLAS=1, two build jobs, Nice=10, background.slice. FP64 tolerances
1e-10/1e-8, FP32 1e-6/1e-5; identities/routes exact.
LH snapshot artifacts/lh-source-20260921-1428: 69 files, identity
ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f,
original HEAD 5fd237d40c9880ccb6e511e4bf20799c7022fd1e plus actual dirty hashes.
Original FP64 active-softmax diagnostics use an FP32 denominator. Assertions-off
fills numerical coverage while retaining ordinary C++ asserts; LH was not patched.

Shared storage filled around 2026-09-21 20:52 UTC. Recovered zero-length docs from
Git and intended changes using fsynced staging/atomic replacement/read-back.
artifacts/single-dev-20260921-2054/ failed before its workload started; preserve
its empty status.tmp and explicit postmortem status, not a test pass/failure.
Project build/artifacts now symlink to /var/tmp/zlong-graph-execution-foundation/.
Migration SHA256-checked 851 artifact and 158 build files before removing only
their duplicate project-owned shared-volume copies. Relocation manifests are in
that local parent. Check capacity before writes. Never clean reference trees.
Keep cited failures/evidence; cleanup dry run found no eligible old artifacts.
Packed first-order VJPs use semantic replay; higher-order AD remains unclaimed.

## M8 development and pilot

Development unit tide-foundation-streaming-bench-dev-20260921-2252 is terminal
inactive/dead, MainPID 0, Result=success, exit 0. Build passed; 15 tests passed in
10.49s, finished 2026-09-21T22:53:59Z. Both status.json and development.json in
artifacts/streaming-bench-dev-20260921-2252/ are passed; frozen dirty source was
86a980b plus archive, tree SHA256
b6af7db5d376aa27f2e595aee40af78ee22a93a92eb1b76292bc8392915aa509.
No performance claim follows from these correctness checks.

Commit the benchmark and fixed pilot driver, then dispatch
unit tide-foundation-streaming-pilot-20260921-2300 from main:
/home/zlong/anaconda3/bin/python scripts/job.py --output-dir
artifacts/streaming-pilot-20260921-2300 -- /home/zlong/anaconda3/bin/python
scripts/pilot_streaming.py --device cpu --dtype float32 --output-dir
artifacts/streaming-pilot-20260921-2300/comparison --jobs 2 --tracking best-effort.
The pilot rebuilds manifests from the clean commit, then runs exactly 16 cases:
nodes 32/4096 x functional/cursor x workers 1/3 x packed 0/1; each has two warmups
and five measured repetitions, four active rings, batch 4, width 16, ticks 32.
Freeze main source/build until this small controlled comparison terminates.
Original-LH qualification remains independent and live; record co-running load.
Inspect outer status, comparison/pilot.json and all 16 run.json/summary.json and
raw metric files; validate complete records and report Trackio degradation.
