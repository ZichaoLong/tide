# Packed source transport and batch Next: CPU qualification

Source: `488500b28a2ce1bebcea62ee0e4f6d93b7cc51df`.
Frozen checkout: `/var/tmp/zlong-graph-execution-foundation/qualification/packed-transport-20260923-143526`.
Records: `artifacts/packed-transport-20260923-143526/`; fixed driver:
`artifacts/qualify_packed_transport.py`; independent terminal audit:
`artifacts/audit_packed_transport.py`; exact launch receipt:
`artifacts/packed-transport-qualification.json`.

Service `tide-packed-transport-20260923-143526` is inactive/dead, MainPID0,
Result=success, ExecMainStatus0. All 30 stages, 13 run records and the independent
`post-run-audit.json` passed. Correctness is qualified; end-to-end speedup is not established.

## Scope and correctness

Two independent, default-off [native Streaming policies](../packed-transport.md)
change physical storage/call granularity while preserving Graph v13/checkpoint v5.
`packed_sources` shares immutable scaled source rows between Aggregate and fiber
attention, and passes a content matrix directly to state batch. `batch_next`
batches adopt-v1 and fiber-state clear. Original source identities, contributions,
comparison snapshots, logical clocks, Next results and Full inputs remain intact.
No Aggregate/Next logical result is skipped. Unsupported custom programs retain
counted fallbacks; unsupported schedules reject the options explicitly.

Clean CPU FP64/FP32 regression: **6897 passed in 891.90s**, after a fresh build.
New cases exercise ragged cycles, reordered source slots, parallel edges, present
zeros, all/selected/clear/old adoption, HARD/SOFTP/HST, exact/single attention,
serial/node-parallel execution, five Aggregate programs and non-fiber consumers.
They compare complete traces/continuations, output/state/pending/content/Next/
contribution VJPs, structural gradient absence, custom fallbacks, source origins,
clock wrappers, no_grad/inference_mode, snapshot ownership, policy changes after
checkpoint, and shared-parameter AdamW with chunk continuation.

Packed training preserves first-order semantic replay, including scalar Next
binding; it is not an optimized-backward claim. Persistent KV storage is still
owned per sample; this increment does not eliminate all Event/State objects.
Batch clear returns row views of a zero matrix and shares immutable empty cache
tensors. Batched source storage may live longer than individual temporary rows;
actual peak RSS is part of the comparison.

Retained development records: `packed-transport-dev-20260923-142352` built and
reported 251 passed/12 failed. New fixtures requested a pending-root objective at
an empty pending cut or shared a whole module across incompatible source domains.
The corrected 12 cases passed in 4.37 s against unchanged C++ in
`packed-transport-retest-20260923-143254`; the clean full gate covers the fixes.
Keep both original source archives/logs and terminal labels.

## Fixed inference comparison

17,269,426,339 parameters; D2048/B512/V50304; FP32/no_grad; four attention heads,
all-softmax pooling, SiLU/RMS and clear; 12 tokens, warmup 4, indices 4–11 measured.
Seed7/fixed external IDs; exact attention, packed row emission, parallel regions
and compact events common. The five earlier fiber options retain their defaults.
LH and PDG use the same four static graph blocks and comparable matrix arithmetic
with independently initialized weights; this is not complete cross-engine
function equivalence.

CPU aarch64, Torch/LibTorch2.10.0+cpu, GCC10.3.1; common affinity 160–319,
sequential processes, 1024 GiB address-space and 1200 s native bounds per wide case.
PDG uses 160 node/head workers in separate phases and ATen/OpenMP/BLAS 1; LH uses
its original 160-thread ATen/OpenMP schedule. Effective PDG pools report 1.
Initialization is excluded; ms/sample-token divides batch elapsed time by 512.
Coarse coordinator timing and work counts are enabled; detailed operator timers
are off for wide timings. No project build or benchmark overlaps a wide timing.

| Case | ms/sample-token | Delta from baseline mean | Token SD | Peak RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| LH, original schedule | 24.69775 | -16.139% | 1.14715 | 148.963 |
| PDG baseline | 29.97732 | +1.788% | 1.25912 | 110.428 |
| Packed sources only | 28.80887 | -2.180% | 1.07373 | 113.234 |
| Batch Next only | 28.96047 | -1.665% | 0.99344 | 111.320 |
| Both options | 29.40178 | -0.167% | 1.06248 | 111.161 |
| Both options, repeat | 29.83338 | +1.299% | 1.25726 | 110.874 |
| PDG baseline, repeat | 28.92433 | -1.788% | 1.04302 | 107.725 |

The two baseline means average **29.45082**, the two combined means
**29.61758** ms/sample-token: combined latency is **0.5662% higher**.
The baseline repeat is 3.5126% faster than its first process; the combined repeat
is 1.4680% slower. The smaller single-option observations (sources −2.1798%,
Next −1.6650% versus the baseline mean) each have only one process and do not
establish repeatable gains. The combined mean remains 19.9201% above the single
LH observation. Its peak RSS is 110.874–111.161 GiB versus 107.725–110.428 GiB for
baseline and 148.963 GiB for LH. Do not infer a general memory reduction from
small differences between separate processes.

Coordinator wall seconds per batch-token, averaged over both repetitions:

| Phase | Baseline | Combined |
| --- | ---: | ---: |
| Update | 7.448782 | 7.499240 |
| Full | 6.035481 | 6.107525 |
| State/message publication | 0.479783 | 0.458884 |
| Cleanup | 0.412374 | 0.424325 |

Neither Update nor Full improves in the combined repeated observation. Together
with the small profiled probes below, this supports retaining the options but
not promoting them to defaults. It does not establish which allocator, cache,
scheduling or memory-lifetime effect offsets the saved work. Fewer Tensor
operations are useful implementation evidence, not a wall-time speedup proof.
The current experiment compares policies within one binary; historical timing
changes are not a controlled before/after measurement of the source revision.

## Local cost probe

A separate D128/B128/V257 probe uses 16 workers, 8 tokens/warmup 2 and detailed
operator timers. Each variant has one process; these profiled small differences
are not stable speedup estimates.

| Probe | ms/sample-token | Token SD | Peak RSS GiB |
| --- | ---: | ---: | ---: |
| baseline | 2.85224 | 0.33170 | 0.941 |
| sources | 2.80058 | 0.20604 | 0.952 |
| next | 2.94620 | 0.44887 | 0.947 |
| combined | 2.78908 | 0.20048 | 0.958 |

Per batch-token mean, seconds summed over workers:

| Metric | Baseline | Sources | Next | Combined |
| --- | ---: | ---: | ---: | ---: |
| Aggregate worker seconds | 0.152226 | 0.174454 | 0.150282 | 0.173201 |
| Input-pack worker seconds | 0.040494 | 0.002348 | 0.038259 | 0.002310 |
| Next calls | 18290.000000 | 18290.000000 | 801.333333 | 801.333333 |
| Next worker seconds | 0.084185 | 0.082512 | 0.053005 | 0.051413 |

Summed worker timers include scheduling pauses and are not coordinator wall time.
They cannot be added to or substituted for end-to-end latency. Source transport
moves some work into Aggregate while reducing subsequent input packing; fewer
Next calls likewise do not by themselves establish a throughput gain.

## Work, numerical checks and terminal audit

The independent audit verifies 446 tracked files against the tested Git
archive, 12 native binary hashes, 250 source-kit files, archive/manifest
hashes, prepared LH source/graph/binary identities and all 13 completed records.
Both relocated kit engines build/run in a path containing spaces. The kit PDG
binary equals the primary binary byte-for-byte; optimized PDG smoke passes its
complete small scalar-slot/scalar-row/packed/parallel state anchors. Original
LH complete small logits equal the earlier anchor exactly.

Across the six wide PDG cases, all original per-token model/logical-work and
remaining operator inventories agree. The declared physical change is source
scaling reuse: average `op/fiber_scale_elements` 173,023,232 becomes 0, with the
same count in `op/fiber_reused_elements`. Source transport introduces an average
921.625 `work/packed_source_batches`; batch Next reports 921.625 `next_batches`
and 896.375 `next_reset_batches`, while 74,857.125 logical `next_steps` remain.
New execution counters do not pretend these are unchanged physical operations.
Major counted matrix FLOPs differ by only 0.00846% between LH and PDG, as before;
source reuse changes neither QKV/projection work nor exact attention grouping.
Every PDG variant's complete 12-token logits checksum sequence equals baseline
exactly (small probes also agree). No unexpected counted workload change occurs.

All 13 run records have complete local histories and passed validators. Trackio
best-effort is degraded at initialization (`ModuleNotFoundError`); intended
project `tide-graph-execution`, local root under this run's `trackio/`, storage
mode auto. No successful dashboard delivery or live dashboard is claimed.

A read-only historical default check compares all 12 token records with the
prior `fiber-efficiency-20260923-113526/baseline` run: all previously reported
model/work/operator counts and logits sums agree exactly. The identities and
empty difference list are in `historical-default-check.json`. This checks values
and workload, not timing equivalence across source revisions or host conditions.

Large logits checksums are supplementary checks, not full-tensor error bounds.
Independent complete small values/states/routes and VJPs provide numerical
anchors. Per-token SD is not an independent-repeat uncertainty estimate: cache
and work change within the token window. Shared-host, short-window observations
do not establish long-context, training, other-width/batch or x86_64 performance.
Trackio is best-effort; unavailable Trackio must not erase complete local records.

## Portable source kit

`artifacts/packed-transport-20260923-143526/export/cpu-attention-compare.tar.gz`
(394690 bytes), SHA256:
`cb45681cac044257c9a15fc11257e928714feda209f5f87edcaac3149b8f015e`.
Both relocated engines build/run; original LH complete small logits exactly match
the retained anchor. See [the packet instructions](../../tools/cpu_compare/README.md).
No commit checkout is required. After extraction, compare sequentially:

```bash
python run_lh.py --device cpu --threads 56 --output-dir runs/lh-wide
python run_pdg.py --device cpu --threads 56 --output-dir runs/pdg-default
python run_pdg.py --device cpu --threads 56 --packed-sources 1 --batch-next 1 \
  --output-dir runs/pdg-transport
```

Use matching target Torch/LibTorch, common target-machine CPU affinity and new
output directories. Add `--smoke` first for target build/state checks. Either
switch can be enabled alone. Defaults remain off; x86_64 must be rebuilt and
verified on the user's server. Reference repositories were not modified.
