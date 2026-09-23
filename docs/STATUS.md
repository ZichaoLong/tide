# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch: graph-execution-foundation.
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

Active user-authorized increment: implement independent same-fiber Attention
packing policy exact|single, default exact. Single flattens queries into one
padded node-local batch while preserving event visibility/decay/clear. Keep
QKV/output batching and existing compact persistent state/pooling semantics.
Expose through the immutable C++ fiber-kernel factory, Native adapter, scale
CLI and portable run_pdg.py; keep graph/checkpoint identity unchanged.
Development complete: attention-dev-20260923-092415 built successfully;
402 passed and4 failed/99.55s, all four failures were the new test's mistaken
AdvanceResult->Result reconstruction. Corrected new-file repeat:80 passed/13.27s
against the identical compiled C++ hash; record artifacts/attention-single-retest.json.
The other326 directed tests passed. Both development units are terminal/failed;
retain their logs and source archives. Earlier attention-dev-20260923-092147
failed compilation from a missed private constructor call; that fix is tested.
Next: commit implementation, freeze clean source, then run fresh build, complete
CPU FP64/FP32 regression, exported/relocated LH and PDG-single smoke, and fixed
wide exact/single pair (D2048/B512/12steps/warmup4, workers160, intra-op1,
CPUs160–319, at most1024GiB address space per native process). No active job.
Fixed pending driver: artifacts/qualify_attention_policy.py. Fresh packet will
replace the old kit only as a new separately identified artifact; preserve both.
Use /home/zlong/anaconda3/bin/python with backend autoload disabled. No reference
repository edits, no push, no sub-agents. Full initial baseline is eebd877.
The existing portable kit remains qualified at dd024e6; see
[its evidence](evidence/cpu-comparison-kit.md). Retain its archive and logs;
the new policy will need a newly exported packet after qualification.

The earlier user-approved LH–PDG operator-work comparison is complete.
Implementation source f0c31bef864af0ccdfa82afc1889c686890fbca6; authority tide-core-3.
Optional inference counters, original-LH preparation instrumentation and small
parity tooling are documented in [operator-work.md](operator-work.md).
[Reviewed evidence](evidence/lh-pdg-operator-work.md) has commands, units and limits.

Development81 tests/47.55s, frozen directed repeat81/48.62s, both FP64/FP32.
Original LH small D16/B4/V257 six-token counted/uncounted complete logits match;
serial/4-thread outputs and all counts pass. The previous full6459-test CPU
regression remains scoped to da5a17b; this increment ran the directed gate.

Fresh wide pair:17.27B/D2048/B512/V50304/FP32/no_grad, 12 tokens/warmup4,
seed7/fixed IDs, same four CSR blocks and independent weights. Means for4–11:
LH24.69385ms/sample-token; PDG28.44610ms (+15.1951%). Matrix FLOPs differ
only0.00846%; all linear projections differ0.06748%. Body selections both32
per sample-token. Boundary-adjusted Emit differs0.01325%. Attention padding:
LH1.80835× vsPDG1.0×, but PDG attention calls6.258× as many. This establishes
comparable matrix work, not exact whole-model function equality or pure dispatch
cost. It does not establish an improvement over the earlier PDG timing.

Unit tide-operator-work-20260923-0300 passed/exit0, inactive/dead, MainPID0;
all6 stages and five portable run records passed. Records:
artifacts/operator-work-20260923-0300/ (status.json, pipeline.json, per-stage logs,
small/wide prepared LH copies, lh-small-parity/, lh-wide/, pdg-wide/, analyze.py,
analysis.json, post-run-audit.json). Fixed driver:
artifacts/operator-work-runner-20260923-0300.py. Immutable source:
/var/tmp/zlong-graph-execution-foundation/qualification/operator-work-20260923-0300.
PDG copied build matches its C++ hash; original development build metadata is
retained (not a fresh clean compile). LH was freshly compiled. Source, binaries,
inputs and flow identities passed audit. All12 PDG model/work/checksums match
prior uncounted optimized output; full-state equality is a small-test claim.
Trackio best-effort/degraded (unavailable); local records are complete.
CPUs160–319, at most160 active workers per phase, 1280GiB address-space bound.
Preserve failed work-dev-20260923-0252 (fixed test brace), and the two repaired
analysis preflight failure records. No reference repo was modified; no push.

Next bounded M8 investigation: measure/update data layout, tensor allocation
and attention bucket/call costs in Aggregate/State/Read and Next/Full/Emit.
Current update7.05349s +Full5.96256s per batch-token dominate timing. Consider
coarser attention packing only with independent complete-state/VJP anchors;
more padding may reduce small operator calls. This is a hypothesis, not a
measured optimization. Wide counter timing overhead, repetitions, longer context,
narrow shape and training performance remain unmeasured. Do not repeat the
completed comparison without a new hypothesis. Weight-preserving import is a
separate exact-inference objective; broader goals stay in ROADMAP.

Earlier optimized source da5a17b and its complete6459-test gate remain retained
in artifacts/pdg-opt-20260923-0100/ and evidence/pdg-streaming-optimization.md.
Older failed narrow/grad pilots remain failed in their evidence and records.

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
