# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch: graph-execution-foundation.
Active increment: optional packed source transport between Aggregate and fiber
attention, plus batched Next adoption/reset. User authorized implementation and
bounded performance trials, preserving canonical semantics and optional policies.
Baseline is fc97b46. Development build succeeded; 251 directed checks passed and 12 failed from
invalid new test fixtures. Corrected 12-case retest passed in4.37s against the
unchanged C++ build. Both services are terminal. Implementation is ready to
commit before immutable CPU/performance qualification; no job currently runs.
Preserve `packed-transport-dev-20260923-142352` (failed, source archive retained)
and `packed-transport-retest-20260923-143254` (passed), without relabeling either.

Plan: retain defaults; validate independent Python/native values, source slots,
routes, isolated VJPs, clear/comparison snapshots, clocks and checkpoint policy
switching; then freeze/commit and run complete CPU gates and a bounded fixed
17.27B baseline/source/Next/combined comparison with repetitions. Options are
execution policies, outside graph/checkpoint identity. Custom modules retain an
explicit counted scalar fallback. Training keeps semantic replay where needed.

The overall objective remains Python/LibTorch generic and independent specialized
PDG/TimedDAG/SettleGraph execution, complete training/inference equivalence,
sparse streaming/prefill performance and LH inference inclusion. Its full
acceptance matrix and remaining scope live only in [ROADMAP](ROADMAP.md).
Historical8.8B/8.5B numbers are references, not strict performance targets.
No push; no sub-agents. LH/fractal-latcarf/ObsidianVault remain read-only.
Run git status and scripts/status.py on re-entry, then follow this file.

## Verified state

Graph v13 / single-graph checkpoint v5 / tide-token-application-v1.
Source and scope are separate for each report:

| Scope | Clean source | Result / evidence |
| --- | --- | --- |
| Fixed CPU memory-policy2×2 follow-up using qualified binaries | fa31ad8202b688dc220f814ef6c2ff6ba2c6e465 | checked small smoke,4 wide cases,12 stages and terminal audit; evidence/fiber-numa.md |
| Fiber execution/storage options, complete CPU gate, portable kit and12 wide cases | fa31ad8202b688dc220f814ef6c2ff6ba2c6e465 | 6719 tests/832.99s;32 stages and terminal audit passed; evidence/fiber-efficiency.md |
| Optional common operator diagnostics and fixed17.27B on/off pair | 2619ed3fc79fe299e1a573b5088c552748f87196 | 67 clean directed tests; fresh LH full-logit checks; evidence/operator-profiling.md |
| Complete CPU regression, exact/single fiber policy and refreshed portable kit | 9e950bfbb7ceee6a5105d7978c0419aab6877859 | 6553 tests/745.74s; fresh relocated LH/PDG smokes; fixed 17.27B pair; evidence/attention-packing-policy.md |
| Portable paired LH/PDG CPU source kit | dd024e6f1d57153c22ab7cef2762d059cbd3ac7d | 18 directed tests; 3 fresh relocated native builds/runs and numerical anchors; evidence/cpu-comparison-kit.md |
| Complete CPU regression plus optional canonical streaming optimizations | da5a17bdbda1196fb32e2352fba9aa3b95e6dde3 | 6459 tests / 716.61s; evidence/pdg-streaming-optimization.md |
| Complete CPU regression, two-clock checkpoint and strict coordinates | 69ca37900e9c10d3fca95570ea1ebca8f9079f46 | 6233 tests / 665.45s; evidence/token-checkpoint-coordinates.md |
| Durable status publication and damaged-record re-entry | 3604ec002722e701c74bd13e6f681b88ada14199 | 16 tests / 0.44s; evidence/durable-records.md |
| Bounded two-clock/single-PDG training and single-graph resume | d233429cd5807614869214dcea21d9492e309fd1 | 4901 tests / 584.15s; evidence/single-graph-training.md |
| Original LH bounded single-PDG inference | c84abbed7c9cfce6d4c6a44f6b7ae5f9aab465d3 | All seven stages passed; evidence/lh-single-graph.md |
| Atomic value-checkpoint publication | 00bbf78fb99e410d46b4d7e36e37266eb3d8cf9e | 88 directed clean tests; evidence/checkpoint-io.md |
| First native streaming performance pilot | aee0da48e4c6661d7a75fd97ca39ddaba654ee4d | 16 cases / 80 events; evidence/m8-streaming-pilot.md |
| Standalone C++ named ownership and SGD/AdamW parity | 9aef27aa2e8eeeda6f3d6298ae295bf687b3a66b | clean frozen build; 338 directed tests plus FP64/FP32 executable checks; evidence/cpp-optimizer-ownership.md |
| Standalone C++ `TIDENCK1` value checkpoint | 4325bb14434bbe0e9702aff244f77ed71e75cbee | clean frozen build; 344 directed tests plus FP64/FP32 checkpoint and optimizer checks; evidence/cpp-native-checkpoint.md |
| Portable LH Attention source kit / clean relocated CPU build | eda5357da86ea0022c8e8aab43f75e315f71de79 | 6 report checks; 26 qualification stages / 4 small native modes; evidence/lh-portable-repro.md |
| Original LH local 9B Add performance pilot | a7edf44eb7046a307c9a53085ea2420e76e644b1 / wrapper 9548be6ecb6635c2beed2aee0b62a1eb5cb05282 | 5 harness checks; 8 large cases / 36 forward events; evidence/lh-local-scale-pilot.md |

The application bundle preserves two complete continuations, real occurrence ledgers,
unfinished token buffers, named cross-graph aliases and optimizer state. 192
three-update/two-restore comparisons cover Python/native serial/parallel/packed,
Add/all-softmax, clear, HARD/SOFTP/HST, SGD/AdamW and both dtypes. Corruption
preflight leaves both live owners unchanged. The application bundle is not a
training controller or standalone C++ file format. Native execution has no
Python callbacks; optimizer
and persistence ownership in those application gates is Python. The separate
native `TIDENCK1` gate covers named values and built-in optimizer state only.

Strict Python int64 checks reject bool/float/overflow before scheduling, native
conversion or checkpoint restoration. Cursor import checks complete metadata
once; advance checks new inputs only. Source 69ca379 did not include the later
durable-record tooling, whose independent clean 16-case gate is listed above.
Both results have source/terminal audits. No C++/original-LH oracle changed.

## Outcome and next action

Qualified implementation source: fa31ad8202b688dc220f814ef6c2ff6ba2c6e465.
Options: CSR fiber pooling, immutable per-sample KV reuse, head-major temporary
attention layout, deferred old-state release and scale projection strides.
They complement exact/single packing and preserve graph identity, canonical
complete-fiber Aggregate/Upd/Read/SelStep/Next/Full, clocks, ownership and public
VJPs. Packed backward still uses semantic replay; this is inference performance
work with gradient correctness checks, not an optimized-backward claim.

At17.27B/D2048/B512/V50304/FP32/no_grad/12 tokens/warmup4:

- Two baseline means28.82426/29.11735 average28.97080 ms/sample-token.
- All160 combination28.79132/28.56860 averages28.67996, only1.0039% lower.
- All116 single observation27.92335 is3.6156% below the two-baseline mean;
  it is not a repeated result. The primary LH observation is25.04663.
- Follow-up LH default25.43408/interleave27.16406: interleave is6.8018% slower.
  PDG default29.35316/interleave29.16130: only0.6536% lower. The smaller gap under
  interleave largely comes from slowing LH; do not claim it solves PDG overhead.
- All primary PDG per-token model/work/operator inventories match. Within each
  NUMA engine pair, inventories and output checksums match exactly. Large sums
  supplement complete small values/states/routes/VJPs; they do not replace them.

Next action: commit this implementation, create a detached read-only worktree
at that commit, and launch `artifacts/qualify_packed_transport.py` through job.py
in background.slice with affinity160–319. Driver: complete CPU FP64/FP32, fresh
relocated kit smokes, four small operator-cost probes, LH and six fixed17.27B
PDG cases (baseline/source/Next/combined plus baseline/combined repeat). Then run
`artifacts/audit_packed_transport.py RECEIPT`; retain exact source/run identities
in `artifacts/packed-transport-qualification.json`. Performance is not yet known.
No complete batched persistent KV storage or optimized backward is claimed.

## Completed job and artifact identities

Both services are inactive/dead, MainPID0, Result=success, ExecMainStatus0,
matching status.json, pipeline.json and post-run-audit.json.

| Scope | Unit / output directory suffix | Result |
| --- | --- | --- |
| Full CPU, portable kit and ablations | tide-fiber-efficiency-20260923-113526 / fiber-efficiency-20260923-113526 |6719 tests/832.99s;32 stages;14 run records |
| Process memory policy | tide-fiber-numa-20260923-131329 / fiber-numa-20260923-131329 | checked smoke;4 wide cases;12 stages;5 run records |

Both use frozen source
`/var/tmp/zlong-graph-execution-foundation/qualification/fiber-efficiency-20260923-113526`.
Outputs live under `/var/tmp/zlong-graph-execution-foundation/artifacts/`.
Exact argv/cwd/source/driver identities: artifacts/fiber-efficiency-qualification.json
and artifacts/fiber-numa-qualification.json. Drivers and independent terminal
audits: artifacts/qualify_fiber_efficiency.py, audit_fiber_efficiency.py,
qualify_fiber_numa.py and audit_fiber_numa.py. Inspect summary records directly
or run the corresponding inspect_fiber_efficiency.py / inspect_fiber_numa.py.

Source audit verifies439 tracked files against Git archive,12 native binary
hashes and248 packet files. Both relocated kit engines build/run; optimized
PDG has complete small scalar/packed/parallel state checks, and original LH
small full logits exactly match the old anchor. Packet PDG binary matches the
primary binary. Reference repositories are unchanged. Trackio best-effort is
degraded/unavailable; all local records/validators are complete.
One read-only NUMA scan occurred during the primary pooling timing and may
perturb its small difference; the explicit note is retained in that run's
numa-observations/. No such scans occurred in the controlled NUMA timings.

## Current portable kit

`artifacts/fiber-efficiency-20260923-113526/export/cpu-attention-compare.tar.gz`,
391597 bytes; SHA256
`83c63e8df95a027d68c706d03f23ec43f922ebccac52c02e0c6ca6cca46aa62a`.
It includes exact/single, all five candidate switches and optional operator
profiling. No commit checkout or original LH source is required to use it.
Extract and run sequentially in the target Torch/LibTorch environment:

```bash
python run_lh.py --device cpu --threads 56 --output-dir runs/lh-wide
python run_pdg.py --device cpu --threads 56 --output-dir runs/pdg-default
python run_pdg.py --device cpu --threads 56 --fiber-pooling csr --fiber-cache owned \
  --defer-state-release 1 --projection-layout linear --attention-layout head \
  --output-dir runs/pdg-candidate
```

Use common target-machine affinity, new output directories and optionally
`--smoke` first. The candidate command is available for comparison, not a
promised speedup on the user's Intel server; x86_64 remains unverified here.
Detailed configuration and standalone LibTorch discovery:
[tools/cpu_compare/README.md](../tools/cpu_compare/README.md).
Do not enable a global interleave default based on the NUMA gap ratio.

## Retained evidence and failures

Keep currently cited snapshots, records and packets. Older source kits and
measurements retain their original source/scope in docs/evidence; the archive
above is the delivered current version. No cleanup was needed in this increment.

Current development records: fiber-efficiency-dev-20260923-112506 preserves
375 passed/48 failed from a malformed no-edge test objective; the corrected48
passed/2.96s in fiber-efficiency-retest-20260923-113224 against unchanged C++.
Keep fiber-efficiency-first-failure and the original failed source archive/audit.
The candidate fiber-efficiency-dev-20260923-110604 was cancelled before compiling.
The profiler namespace-error failure operator-profile-dev-20260923-104828 and
its passed retest operator-profile-retest-20260923-105234 remain retained.
All are fixed/covered by the later clean gate without relabeling old failures.
Use scripts/status.py --all-jobs for retained terminal records.

## Numerical boundaries and retained failures

The two development jobs remain failed: attention-dev-20260923-092147 had a
missed private constructor call; attention-dev-20260923-092415 built and had
402 passed/4 failed because the new test reconstructed AdvanceResult as the
wrong result record. The corrected 80-case test file passed/13.27s against the
same C++ binary (artifacts/attention-single-retest.json). The clean full gate
covers both fixes. Preserve both failed logs and archived source snapshots.

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

CPU aarch64, /home/zlong/anaconda3/bin/python, Python3.11.15,
Torch/LibTorch2.10.0+cpu, GCC10.3.1, C++11 ABI. TORCH_DEVICE_BACKEND_AUTOLOAD=0.
Correctness uses OMP/OpenBLAS/MKL1. Long tasks use immutable source, bounded
build jobs, Nice10/background.slice and durable records. These wide cases used
about-half-host affinity160–319 and1024GiB address-space bounds; exact effective
thread counts and limits belong to their run records. Requested OpenBLAS1 alone
does not ensure one actual thread under an OpenMP BLAS build.
FP64 atol/rtol1e-10/1e-8; FP32 1e-6/1e-5; routes/identities exact.

Shared storage filled in earlier increments. The stable repository path
/home/zlong/llm/graph-execution-foundation symlinks to
/var/tmp/zlong-graph-execution-foundation/repository (including .git); build and
artifacts use that local parent too. About27GiB remained after this increment;
check capacity before large writes. Migration/cleanup inventories remain there.
Use scripts/durable_records.py for fsynced atomic handoff writes and read back
the result. Do not delete referenced evidence or reference repositories; inspect
a fresh dry run before removing only known obsolete project-owned artifacts.
