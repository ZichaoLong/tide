# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: graph-execution-foundation.
The standalone C++ named-owner/optimizer implementation is committed at
5c440e1, with the native region-name fix at 9aef27a. The native `TIDENCK1`
value-checkpoint implementation is committed at 4325bb1 and has passed clean
frozen qualification; see evidence/cpp-native-checkpoint.md. The prior
token-bundle, integer-coordinate and durable-record qualification remains
complete.
The latest user explicitly authorized local performance experiments and clarified
that historical 8.8B/8.5B sizes/times are references, not strict targets. New
standalone original-LH preparation/build/timing tools and bounded local
experiments are complete: five small harness checks and eight large cases
passed. See [local scale evidence](evidence/lh-local-scale-pilot.md). Large matched numerical checks remain unmeasured. The user now prioritizes
graph-only comparable-scale PDG Attention timing with random weights, without
waiting for a weight-preserving importer.
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

The latest user explicitly requests comparable-scale PDG performance without LH
weight import. Reuse the existing four-block LH graph, freshly initialize
Attention/all-softmax + SiLU/RMSNorm + linear Emit, and match width/batch,
parameter count and region budgets. Record actual candidates/selected rows,
source rows/edges and packing, because random routes can change work. Implement
a standalone native cursor benchmark, validate small serial/parallel/packed and
row-versus-slot Emit, then ramp to width2048/batch512 under at most160 cores and
roughly half host memory. Weight-preserving import is a separate future task,
not a blocker for this lane. Do not claim numerical LH parity for these runs.

The native benchmark, graph-only exporter, row Emit and recorded wrapper are
implemented. Development gate artifacts/pdg-scale-dev-20260922-1120 passed:
57 tests in24.91s, FP32/FP64 including full trace/state/route parity and existing
streaming/cursor checks. Unit tide-pdg-scale-dev-20260922-1120 passed/exit0.
The fixed pilot driver adds a timeout-finalization check, small clean matrix,
width256/batch32 and width2048/batch64 ramps before wide2048/batch512 parallel
(12steps) and serial(4steps), then narrow128/batch512 parallel(8steps).
Wide/narrow inputs are artifacts/pdg-scale-input-20260922/{wide,narrow}.txt
and adjacent JSON provenance; no LH weights were read.

Active clean source f6686c124c8f588a44c1825603ffa5c8843dd422 is frozen at
/var/tmp/zlong-graph-execution-foundation/qualification/pdg-scale-20260922-1020,
with validated build executables/manifest copied into its build/.
The fixed driver scripts/pilot_pdg_scale.py runs through scripts/job.py in unit
`tide-pdg-scale-20260922-1020`, CPUs160-319, background.slice, Nice10,
RuntimeMaxSec7200, BLAS1 and at most160 node workers. Job output:
artifacts/pdg-scale-20260922-1020/ (pilot cases under comparison/).
Submission verified active/running in background.slice, transient=yes.
Timeout-finalization and three smoke stages are accepted; the timeout case
remains recorded as failed, with native exit-15 and no unreaped child. The
width256/batch32 ramp completed6steps. Initial terminal record validation is in
artifacts/pdg-scale-20260922-1020/record-validation.json. The width2048/batch64
case is constructing its model; no large PDG token result is available yet.
The frozen source and build remain fixed.
Inspect status.json, comparison/pilot.json, per-case summaries and unit state;
stop with systemctl --user stop tide-pdg-scale-20260922-1020 if needed.
Do not claim submission or partial/failed cases as passed.

The original Attention pipeline has terminated failed/exit1 (unit
`tide-lh-a10-attention-20260922-0830`, MainPID0, inactive computation).
Records: artifacts/lh-a10-attention-20260922-0830/; source6ade84d.
Wide nograd completed100 steps:17,269,426,339 parameters,
26.6342ms/sample-token, peak193.506GiB. Wide grad-forward timed out after79
steps and has missing terminal native resource fields; retained postmortem
preserves them as null. Narrow nograd timed out before any completed token,
with16,608,289,021 parameters and native exit-15; its summary records
606,587,662,336 peak RSS bytes and1814.4654 process seconds. Narrow grad was
skipped. These failures are retained; no large original-LH rerun is needed now.
The previously documented cleanup/finalization defect remains in ROADMAP.

The portable LH17.27B source kit is complete and qualified:
[commands](../tools/lh_repro/README.md), [evidence](evidence/lh-portable-repro.md).
Archive: artifacts/lh-repro-export-20260922-0930/lh-a10-wide-repro-kit.tar.gz.
Unit tide-lh-repro-check-20260922-0930 passed/exit0. Preserve the archive and
artifacts/lh-repro-check-20260922-0930/. Target Intel execution is the user's step.

## Latest terminal jobs and retained source

Both large pilot jobs passed with exit 0; systemd inspection found inactive/dead,
MainPID 0, Result=success. Persistent status.json and all case exit codes agree:

- `tide-lh-scale-pilot-20260922-0730`, finished 2026-09-22T07:35:40Z.
  Records: `artifacts/lh-scale-pilot-20260922-0730/`.
  Frozen source a7edf44: qualification/lh-local-20260922 under the local parent.
- `tide-lh-window-threads-20260922-0740`, finished 2026-09-22T07:41:15Z.
  Records: `artifacts/lh-window-threads-20260922-0740/`.
  Frozen wrapper 9548be6: qualification/lh-local-threads-20260922.
  Contains analyze.py, comparison.json, record-validation.json and
  post-run-audit.json; both clean trees match all 365 tracked Git-archive files,
  and binary/snapshot/input hashes match. All eight run records validate.

The native binary is retained at
`/var/tmp/zlong-graph-execution-foundation/build-lh-local/tide-lh-bench`.
Prepared inputs: `artifacts/lh-local-{wide,narrow}-input-20260922-0720/prepared/input.json`.
Build/input preparation and all five smoke jobs passed; identities and logs are
in the linked evidence. No full Tide regression rerun was needed for this
separate harness/docs increment; the complete baseline remains scoped above.

Brief observations (FP32/batch512/56 physical cores): wide2048/narrow128
actual parameters 9.468B/9.025B; nograd steps4–11 mean 6.096/16.109 ms/token.
Aligned steps1–3 grad-forward versus nograd time ratios are 1.072x/5.068x.
BLAS1 versus BLAS56 steps4–7 ratios are 1.009x/0.895x. This is a single-seed,
single-repetition Add workload, not attention or full-scale backward timing.
Grad windows are short and retained continuously. Aggregate counts/logit sums
match in the aligned comparisons; they do not prove complete large-model parity.
Trackio was best-effort/degraded because it is not installed; local data are complete.

Earlier qualification and failure artifacts remain retained and are referenced
by their evidence reports. Old training/LH qualification worktree cleanup is
recorded in /var/tmp/zlong-graph-execution-foundation/qualification-worktree-cleanup.json.
No cleanup was needed for this increment. Keep current cited snapshots,
inputs, binaries and records; inspect a fresh dry run before any deletion.

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
Default correctness OMP/OpenBLAS=1, two build jobs, Nice=10/background.slice.
The earlier Add cases use ATen/OpenMP56, BLAS56 or1, inter-op1 and
CPU affinity160–215. The a10fdb1 original Attention cases use ATen/OpenMP160,
BLAS1, original inter-op defaults and CPU affinity160–319. FP64 atol/rtol
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
