# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch: graph-execution-foundation.
The bounded packed-source/batch-Next increment is complete. No job is active and
no new large run is queued. Implementation source:
`488500b28a2ce1bebcea62ee0e4f6d93b7cc51df`.
This checkpoint contains only reviewed evidence, the scoped support contract and
documentation; no implementation changed after qualification. Run git status
and scripts/status.py on re-entry to confirm the worktree and job records.
No push; no sub-agents. LH/fractal-latcarf/ObsidianVault remain read-only.

The overall objective remains Python/LibTorch generic and independent specialized
PDG/TimedDAG/SettleGraph execution, complete training/inference equivalence,
sparse streaming/prefill performance and LH inference inclusion. The full
acceptance matrix and remaining work live only in [ROADMAP](ROADMAP.md).
Historical8.8B/8.5B timings are references, not strict performance targets.
[Architecture](architecture.md) indexes contracts, implementations and evidence.

## Qualified outcome

Graph v13 / single-graph checkpoint v5 / tide-token-application-v1 are unchanged.
The [two independent options](packed-transport.md) require packed native
Streaming and default off:

- `packed_sources` / `--packed-sources 1`: immutable scaled source rows shared by
  Aggregate and fiber attention; packed content passed directly to state batch.
- `batch_next` / `--batch-next 1`: adopt-v1 and selected fiber reset batches;
  custom/unsupported local programs retain explicitly counted fallbacks.

Source identities, contributions, present zeros, comparison snapshots, clocks,
Next results and Full inputs are preserved. Physical allocation/call counts are
not canonical semantic obligations. Training retains scalar semantic replay,
including isolated first-order Next VJPs; no optimized-backward claim follows.
Persistent KV remains per sample, and Event/State records still exist.

Clean build and full CPU FP64/FP32 regression: **6897 passed / 891.90s**.
Fresh relocated source-kit builds/smokes passed for both engines, including
complete small PDG states and original-LH full logits. Independent terminal audit
passed:446 tracked source hashes against Git archive,12 native binaries,
250 packet files,30 stages and13 completed experiment records.
[Reviewed evidence](evidence/packed-transport.md) owns the detailed scope/limits.

Fixed17.27B/D2048/B512/V50304/FP32/no_grad/12 tokens/warmup4, exact attention:

| Configuration | ms/sample-token | Peak RSS GiB |
| --- | ---: | ---: |
| LH, one process | 24.69775 | 148.963 |
| PDG baseline, two-process mean | 29.45082 | 107.725–110.428 |
| Packed sources only, one process | 28.80887 | 113.234 |
| Batch Next only, one process | 28.96047 | 111.320 |
| Both options, two-process mean | 29.61758 | 110.874–111.161 |

Combined is0.5662% slower than baseline and19.9201% slower than the single LH
observation. Baseline repeats29.97732/28.92433 differ−3.5126%; combined repeats
29.40178/29.83338 differ+1.4680%. Single-option observations are not repeatable-gain
proof. No stable end-to-end improvement is established; defaults stay off.
Small profiled probes confirm less input-pack work and fewer Next calls, with
some work moving into Aggregate. All PDG logits checksum sequences agree
exactly; original logical/matrix work agrees, while source scaling is honestly
reported as reuse. Historical default values/work also agree exactly; historical
timing differences are not a controlled source-revision comparison.

## Completed job and exact inspection

Unit `tide-packed-transport-20260923-143526`: inactive/dead, MainPID0,
Result=success, ExecMainStatus0, matching both persistent terminal records.
Frozen checkout:
`/var/tmp/zlong-graph-execution-foundation/qualification/packed-transport-20260923-143526`.
Output:
`/var/tmp/zlong-graph-execution-foundation/artifacts/packed-transport-20260923-143526`.
Main-tree `artifacts/` links to the same artifact parent.

- `status.json`, `pipeline.json`, `full-cpu/result.json`, `post-run-audit.json`:
  terminal state and source/build/acceptance audit.
- `analysis.json`, each case's `run.json`, `summary.json`, `metrics.jsonl`:
  configurations, all raw tokens, medians/spread, resources and numerical counts.
- `task.log` and stage/case logs: full execution output.
- `artifacts/packed-transport-qualification.json`: exact argv/cwd/source/unit.
- `artifacts/qualify_packed_transport.py`, `audit_packed_transport.py`:
  preserved driver and independent audit.

```bash
python artifacts/inspect_packed_transport.py
python artifacts/audit_packed_transport.py artifacts/packed-transport-qualification.json
```

All cases ran sequentially, affinity160–319, build2, PDG160 node/head workers
with ATen/OpenMP/BLAS1, original LH160-thread schedule,1024GiB address-space and
1200s native timeout bounds. No project build overlapped wide timing. Timings
exclude construction and divide by batch. This shared host is not exclusive.
Trackio best-effort was degraded/unavailable; complete local records passed all
validators. No delivered/live dashboard is claimed.

## Current portable kit

[CPU comparison source archive](../artifacts/packed-transport-20260923-143526/export/cpu-attention-compare.tar.gz),
394690 bytes; SHA256
`cb45681cac044257c9a15fc11257e928714feda209f5f87edcaac3149b8f015e`.
Includes exact/single, five earlier fiber options, both new switches and optional
operator profiling. No commit checkout or original LH source is needed.
Extract and run sequentially in the target Torch/LibTorch environment:

```bash
python run_lh.py --device cpu --threads 56 --output-dir runs/lh-wide
python run_pdg.py --device cpu --threads 56 --output-dir runs/pdg-default
python run_pdg.py --device cpu --threads 56 --packed-sources 1 --batch-next 1 \
  --output-dir runs/pdg-transport
```

Use common target-machine affinity and new directories; add `--smoke` first.
Either switch can be enabled alone. Intel x86_64 remains unverified here.
Details and standalone LibTorch discovery: [packet instructions](../tools/cpu_compare/README.md).
Earlier immutable reports retain their original packets/scope; the link above
is the current delivery. No artifact cleanup was necessary.

## Next action and retained boundaries

This requested trial is finished. On the next implementation increment, consult
ROADMAP and choose one bounded question: stable node/worker locality or sparse
batched persistent state/cache ownership. Establish small independent state,
route and isolated-VJP anchors plus a local cost probe before another wide run.
Do not assume the remaining LH gap is required by PDG semantics; do not repeat
this completed series without a new hypothesis. Long context, narrow scale,
prefill and backward/replay performance remain separate unfinished work.

Keep the current failed development record
`packed-transport-dev-20260923-142352` (251 passed/12 invalid-fixture failures)
and corrected `packed-transport-retest-20260923-143254` (12 passed/4.37s against
unchanged C++), their archives and logs. The later full gate covers the fixes;
never relabel the old failure. Prior failures/packets remain retained and are
navigable through `scripts/status.py --all-jobs`, prior evidence and Git history.

AdamW training uses epsilon1e-5 explicitly; the retained FP32 packed-attention
1e-8 reproducer is `artifacts/single-training-adamw-fp32-repro/`. Higher-order AD
is unclaimed. Exact original-LH inference qualification is bounded to equal
width, fixed inference weights and specified profiles. This graph-only scale
comparison uses independent weights and does not claim complete cross-engine
function equivalence. Application-bundle/native-value checkpoint boundaries and
broader module/controller goals remain documented in semantics and ROADMAP.

## Environment and persistence

CPU aarch64; /home/zlong/anaconda3/bin/python; Python3.11.15,
Torch/LibTorch2.10.0+cpu, GCC10.3.1, C++11 ABI. TORCH_DEVICE_BACKEND_AUTOLOAD=0.
FP64 atol/rtol1e-10/1e-8; FP32 1e-6/1e-5; routes/identities exact.
Requested OPENBLAS_NUM_THREADS=1 alone does not establish effective BLAS threads.

The stable path /home/zlong/llm/graph-execution-foundation symlinks to
/var/tmp/zlong-graph-execution-foundation/repository, including .git; build and
artifacts use that local parent. About28GiB remained during this increment;
check capacity before large writes. Shared storage filled in prior increments.
Use scripts/durable_records.py for fsynced atomic handoff writes and read back
the result. Keep cited evidence and reproducers; inspect a fresh cleanup dry run
before deleting only known obsolete project-owned artifacts. Never clean the
reference repositories. Do not edit source/binaries still used by active jobs.
