# Same-fiber batch/sequence packing and original CROSSBATCH

2026-09-22 (Asia/Shanghai), clean implementation
`eb9dc6c86ef46ab4db9e06029b0029412a890395`.
**2764 tests passed in 241.39 seconds**. Outer job, verification and original
oracle all passed with exit 0 and empty dirty status at that commit.
Unit `tide-foundation-fiber-pack-20260921-1638` is inactive, MainPID 0.
Records: `artifacts/fiber-pack-20260921-1638/{status.json,task.log,verification/,oracle/}`.

Command: `python scripts/qualify.py --output-dir artifacts/fiber-pack-20260921-1638
--jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428`, via scripts/job.py in
background.slice. CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu,
C++11 ABI; two build jobs, Torch/BLAS single-thread, backend autoload disabled.
Schemas at this source: native graph v11, checkpoint v4.

## Actual packed work

Python/native frontier projects flattened real source rows and evaluates separate
sample score matrices, grouped by initial cache and total source-row lengths.
Native streaming batches independent samples through the same kernel. Scalar
step remains available for reference, serial scheduling and semantic VJP replay.
Contracts and source navigation: `../fiber-packing.md`, `../fiber-attention.md`.

New work-counter cases cover three active samples plus an absent sample, zero or
two initial cache rows, three events, and ragged fiber boundaries. Regular samples
use one packed group; ragged samples use two; native unbatched frontier uses three.
The recorded max batch is respectively 3/2/1, max sequence is three **events**, and
score elements are exactly `sum_b heads*source_rows_b*(cache_b+source_rows_b)`.
No cross-sample quadratic matrix or padding-induced event occurs. Ordered log-bias
values are bitwise equal to scalar recurrence in these cases. Final K/V, log-bias
and state values have compact tensor storage; inference replay count is zero.

All prior scalar-profile tests pass with packed execution: source-slot permutations,
analytic formulas, initial/idle caches, selected clear, cycles, independent
specializations, SettleGraph embedding, cuts, cursor, detach, sharing, AdamW and
checkpoint restore. New isolated first-output tests exclude future inputs, other
samples, later source scales, and decay when no old cache exists. All cache-slot
roots and disabled-prefill execution also agree in value and first-order VJP.

## Unmodified original LH oracle

Snapshot identity remains
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`; original
runtime assertions stay enabled and no reference files were changed.
Each dtype passed **264 configurations, 6336 ticks, 17688 candidate updates**.
This includes the prior five modes plus CROSSBATCH through its actual batch-owned
K/V/log-bias cache and per-sample indices. CROSSBATCH requires original multi-batch
mode; the forbidden single-batch combination is not claimed.

Compare eager original ticks with Tide tick/whole windows across serial, packed
streaming and packed frontier, including initial state, idle ticks, clear and
cache capacity growth. Width/head/bias domains and norm precision policy are those
in `lh-attention.md`. Original Selector, Add and Full also passed again in both
dtypes during this clean qualification. Source/library/CMake/binary hashes and
per-component logs are retained; `build/lh-oracle` is a reusable cache.

Before commit, **246 targeted tests** plus the six-mode original Attention oracle
passed in `artifacts/fiber-pack-dev-20260921-1633/`; its source tar/hash is retained.
Two incorporated draft files and one Python import cache were removed only after
a dry-run body comparison with installed source. No other artifacts were removed.

No throughput, latency or memory-scale claim is established. Dense per-sample
scores, repeated cache concatenation, tick-repeat bias loops and training replay
remain disclosed costs. This only qualifies sum Confluence; weighted pooling,
whole IOCortexNet/Pronounce and higher-order AD remain outside this gate. Prior
ill-conditioned FP32 RMS failures remain retained, with unchanged strict acceptance
tolerances on the bounded cases described in `lh-attention.md`.
