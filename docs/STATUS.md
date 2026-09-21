# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: graph-execution-foundation.

## Verified scope and next action

Clock implementation bd933e0980a2076bace5c6ffbb09da61bc400593 passed 3745 tests
in 300.89s; evidence/state-clocks.md (evidence commit 6aa0963). Graph v13 /
checkpoint v4 at that source; working checkpoint v5. SourceDomain evidence: evidence/source-domains.md.
Latest clean original-LH qualification remains 776fc2e597dac872f4204ac3a09966db39ab74c0,
evidence/lh-iocortex.md. The single-PDG implementation below passed its complete
development gate; clean full qualification and reviewed evidence are next.

1. Single-PDG implementation committed as a558b87. Checkpoint ownership v5 is
   now integrated from the isolated development worktree; 110 directed tests
   passed there, including SGD/Adam/AdamW next-update equality and failure before
   mutation. Commit this repair; clean full qualification remains pending.
2. Bring the explicit IOCortex smoke scope from
   /var/tmp/zlong-graph-execution-foundation/ownership-dev. Its scope changes are
   CMakeLists.txt, scripts/{build,develop_lh,check_lh_selector,
   check_lh_iocortex_python}.py, cpp/test/lh_iocortex_{scope.h,oracle.cpp},
   cpp/test/lh_scope_check.cpp, tests/test_lh_gate_scope.py, docs/verification.md.
   Standalone scope inventory test and two Python scope-rejection tests passed;
   actual original-LH smoke gate still needs to run. Full remains the default.
3. Commit these coherent changes after their directed gates. Launch full clean
   qualification from an isolated frozen exact-commit worktree with independent
   build directories and the same immutable LH snapshot. Main source may then
   progress without changing the long job's inputs. Evidence is a separate commit.
4. Refresh stale plan/navigation text, then continue M6 composite ownership and
   broader module training, and M8 sparse performance. ROADMAP owns the backlog.

No active job now. No push; no sub-agents. Reference repositories remain read-only.

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
