# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch: graph-execution-foundation.
The previous increment is complete: configurable native same-fiber
Attention `exact|single`, default exact, with full CPU regression, a refreshed
portable source kit and a bounded 17.27B same-binary performance comparison.
See [the policy and API](attention-packing-policy.md) and
[reviewed evidence](evidence/attention-packing-policy.md).

The overall objective remains Python/LibTorch generic and independent specialized
PDG/TimedDAG/SettleGraph execution, complete training/inference equivalence,
sparse streaming/prefill performance, and LH inference inclusion. Its full
acceptance matrix and remaining scope live only in [ROADMAP](ROADMAP.md).
Historical 8.8B/8.5B numbers are references, not strict performance targets.
No push; no sub-agents. LH/fractal-latcarf/ObsidianVault are read-only.
Run git status and scripts/status.py on re-entry, then follow this file.

## Verified state

Graph v13 / single-graph checkpoint v5 / tide-token-application-v1.
Source and scope are separate for each report:

| Scope | Clean source | Result / evidence |
| --- | --- | --- |
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

## Next action

User authorizes performance investigations and optimizations within canonical PDG
semantics. Common optional timers are committed and their diagnostic run passed;
see evidence/operator-profiling.md. Exact disabled-timer baseline: LH25.10840,
PDG30.36451 ms/sample-token. Profiling variants were ~5% faster for both engines,
so do not treat their difference as an overhead estimate. Worker sums are not
wall latency. Pooling is small; data layout and event management remain candidates.

Candidate implementation is ready for committing, defaults unchanged:
CSR fiber pooling; immutable per-sample KV reuse; head-major temporary attention
layout; deferred old-state retirement; scale projection weight layout. No graph
identity, Aggregate/Next behavior, scalar oracle or public VJP is intentionally
changed. Contract: fiber-efficiency.md. No candidate speedup is established yet.

Directed development:375 tests passed initially;48 failures came only from a
new isolated no-edge fixture requesting a nonexistent pending-gradient objective.
Corrected48/2.96s passed against the unchanged compiled C++ source; both services
terminated and were audited. Values/states, routing, gradients, strided shared
optimizer owners, snapshots and checkpoint policy switches pass. Records:
artifacts/fiber-efficiency-dev-20260923-112506/ (retained failed archive),
artifacts/fiber-efficiency-first-failure/, and
artifacts/fiber-efficiency-retest-20260923-113224/ (passed retest).
No candidate wide timing or full regression result yet.

Next: commit the candidate implementation and run
artifacts/qualify_fiber_efficiency.py from a new clean frozen worktree through
job.py/background.slice. It builds, runs the complete CPU gate, exports and
smokes a relocated source kit, then fixed17.27B independent/combined ablations.
The driver also tests116 workers and repeats baseline/combined. Pass
--lh-prepared artifacts/operator-profile-20260923-105551/wide-prepared by absolute
path. Use CPU affinity160–319,4 build jobs,1024GiB per native case; allow enough
bounded service time for serial tests/experiments. Do not overlap project builds
with measured wide cases. Continue through terminal analysis and evidence;
compare source identities and work counters before interpreting speed.

The original queued candidate service fiber-efficiency-dev-20260923-110604 was
cancelled before compiling to add head-layout tests; its cancelled status and
snapshot remain. The profiler's first development failed on a namespace error;
original source.tar.gz/log/terminal audit are retained at
artifacts/operator-profile-dev-20260923-104828. The retest178/85.20s passed at
artifacts/operator-profile-retest-20260923-105234. The clean diagnostic's67/70.94s
and all LH/wide stages passed at artifacts/operator-profile-20260923-105551;
terminal-audit.json matches inactive/dead, MainPID0, Result=success, status0.
No push, no sub-agents; reference repositories remain read-only.

The previous runtime implementation remains qualified at9e950bf, with evidence
at8bd17a0. Its source archive and records below remain the current delivered kit
until a later increment is qualified and exported.

The portable archive now includes `--attention-packing exact|single`. To test on
the user's Intel server, copy and extract the new archive, then run sequentially
from its cpu-attention-compare directory in the target Torch environment:

```bash
python run_pdg.py --device cpu --threads 56 --attention-packing exact --output-dir runs/pdg-exact
python run_pdg.py --device cpu --threads 56 --attention-packing single --output-dir runs/pdg-single
python run_lh.py --device cpu --threads 56 --output-dir runs/lh-wide
```

Use new output directories, common affinity and optionally `--smoke` first.
No commit checkout or original LH source is needed. Commands/configuration:
[tools/cpu_compare/README.md](../tools/cpu_compare/README.md).
Intel x86_64 remains unverified locally. Do not repeat the completed wide pair
without a new measurement question. Follow the M8 section of ROADMAP for the
next bounded performance work: pooling, temporary KV movement and allocation;
longer context, repetitions, narrow shape and training remain separate scopes.

## Prior complete CPU qualification

Unit tide-attention-policy-20260923-093207 passed / exit 0; all 12 driver stages
passed. Terminal audit observed inactive/dead, MainPID 0, Result=success,
ExecMainStatus 0, agreeing with persistent status/pipeline. Finished 02:05:21Z.

- Frozen source: `/var/tmp/zlong-graph-execution-foundation/qualification/attention-policy-20260923-093207`.
- Records: `artifacts/attention-policy-20260923-093207/`: status.json,
  pipeline.json, full-cpu/, kit-pdg/, kit-lh/, wide-exact/, wide-single/,
  analysis.json, post-run-audit.json, export/ and relocated kit with spaces/.
- Fixed driver: `artifacts/qualify_attention_policy.py`; audit:
  `artifacts/audit_attention_policy.py`. Exact argv/cwd are in the job/run records.
- Current archive: `artifacts/attention-policy-20260923-093207/export/cpu-attention-compare.tar.gz`,
  384753 bytes; SHA256
  `b45ab7b2fdb18b032f44382c2a5074377adcff854343c83839aacff7efc3b6c5`.
- Inspection: `/home/zlong/anaconda3/bin/python scripts/status.py` and
  `cat artifacts/attention-policy-20260923-093207/post-run-audit.json`.

Fresh CMake build; 6553 CPU FP64/FP32 tests/745.74s. Relocated PDG-single and LH
fresh builds and six-token smokes pass, with complete small PDG state checks
and exact LH full-logit agreement with the prior anchor. All 428 frozen tracked
files match Git archive; 11 build hashes, 244 packet files, prepared source and
binary hashes, archive/manifest and four run records pass the terminal audit.

The new pair uses 17,269,426,339 parameters, D2048/B512/V50304, FP32/no_grad,
12 tokens/warmup 4, seed 7/fixed IDs, workers/head-workers 160 in separate phases,
ATen/OpenMP/BLAS/inter-op 1. Exact 29.35656 versus single 31.53019 ms/sample-token:
single is 7.4042% slower in this one short-window run. Attention calls fall
5769.25→921.625 per batch-token (84.0252% fewer); score padding 1→1.80757;
peak RSS 109.51699→114.45406 GiB. All other model/work/operator inventories match
for all 12 tokens. Output-sum difference≤0.00561374 is only a checksum observation;
complete value/state/route/VJP equivalence is established by the small tests.
The prior exact run's inventories/checksums are unchanged. Keep default exact.

Wide affinity 160–319, 1024 GiB address-space and 1200s bounds per native process;
service 7200s, Nice10/background.slice, 4 build jobs. Trackio best-effort/degraded
(unavailable); all local records complete. This does not establish a general
speedup, optimized backward, long-context result or new LH–PDG timing comparison.

## Retained prior evidence

The original portable kit remains retained under artifacts/cpu-kit-20260923-0345/
and its [evidence](evidence/cpu-comparison-kit.md), including the standalone
Python-without-Torch discovery check. Deliver the new archive above for the
packing option; do not overwrite or relabel the old packet.

The prior [LH–PDG work comparison](evidence/lh-pdg-operator-work.md) remains
scoped to f0c31be: LH 24.69385 versus PDG 28.44610 ms/sample-token, matrix work within
0.00846%, independent weights and one short window. Records and its source
remain in artifacts/operator-work-20260923-0300/ and the corresponding
qualification directory. It is not a contemporaneous baseline for the new pair.
The earlier optional [streaming optimizations](evidence/pdg-streaming-optimization.md)
and complete 6459-test gate remain in artifacts/pdg-opt-20260923-0100/.

Older local LH pilots and terminal/failed cases remain referenced by
[local scale](evidence/lh-local-scale-pilot.md),
[original Attention reproduction](evidence/lh-portable-repro.md), and
[PDG scale evidence](evidence/pdg-scale-attention.md). Their exact source,
commands, units, binaries and artifact locations belong to those reports.
No cleanup was needed here. Keep currently cited snapshots and failure
reproducers; inspect a fresh dry run before removing project-owned artifacts.

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

CPU aarch64, /home/zlong/anaconda3/bin/python, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. TORCH_DEVICE_BACKEND_AUTOLOAD=0;
Default correctness OMP/OpenBLAS=1, two build jobs, Nice=10/background.slice.
The earlier Add cases use ATen/OpenMP56, BLAS56 or1, inter-op1 and
CPU affinity160–215. The a10fdb1 original Attention cases use ATen/OpenMP160,
OPENBLAS_NUM_THREADS=1 requested (effective BLAS count was not logged),
original inter-op defaults and CPU affinity160–319. See the profile diagnosis
for the fresh-process OpenMP BLAS correction. FP64 atol/rtol
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
