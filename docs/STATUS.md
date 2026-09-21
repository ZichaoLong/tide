# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: `graph-execution-foundation`.

## Current qualification

**2728 tests passed** on clean implementation
`dfc2e5b622946614831962fb44683990fa609e8d`; `evidence/lh-attention.md`.
Original LH Selector/Add/Full/Attention passed in FP64/FP32. Attention per dtype:
240 configurations, 5760 ticks, 16080 candidate updates across five original modes.
This qualifies same-fiber sum attention's scalar baseline and complete cache/clock
projection, not whole-model behavior or joint packed attention.

Unit `tide-foundation-fiber-20260921-1618` is inactive, MainPID 0, exit 0.
`artifacts/fiber-20260921-1618/{status.json,verification/result.json,oracle/result.json}`
all passed at the same clean source. No active job. Evidence is committed separately.

Earlier failures `fiber-dev-20260921-1601` and `fiber-dev-20260921-1608` remain
retained and failed. Corrections, FP32 conditioning limits and Read precision
comparison policy: `fiber-attention.md` and the qualification report.

## Current increment and next qualification

Real packed source/event attention is ready to commit: `fiber_packing.py` and
`fiber_packing.{h,cpp}`. Scalar step remains independent. The data/visibility,
storage, work-counter and replay contracts are in `fiber-packing.md`.

Development unit `tide-foundation-fiber-pack-dev-20260921-1633` is inactive,
MainPID 0, exit 0. Output `artifacts/fiber-pack-dev-20260921-1633/` records **246
passing targeted tests**, then original attention passing both dtypes across all
six modes, including CROSSBATCH: each 264 cases, 6336 ticks, 17688 candidates.
The dirty source snapshot/hash and oracle manifests are retained in that output.
Two incorporated draft sources and their one Python import cache were removed
only after a dry-run comparison confirmed their bodies are preserved in source.
No other artifacts or reference files were removed.

After implementation commit, submit `tide-foundation-fiber-pack-20260921-1638`
through scripts/job.py with output `artifacts/fiber-pack-20260921-1638`, running
`python scripts/qualify.py --output-dir artifacts/fiber-pack-20260921-1638 --jobs 2
--lh-snapshot artifacts/lh-source-20260921-1428`. Freeze until terminal; inspect
unit/status.json/verification/result.json/oracle/result.json. Save evidence in a
separate commit only after all pass. No performance claim follows from counters.

Then continue `lh-attention-plan.md`: post-attention normalized/learned Confluence,
token-clock Pronounce and IOCortexNet adapter. Normalized
pooling must not multiply coefficients into pre-attention Q/K/V input rows.
ROADMAP retains optimizer/backward/cache/history-patch and scale/performance work.

## Immutable original source and execution policy

Snapshot `artifacts/lh-source-20260921-1428`: 69 C++/JSON-header files, identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`; LH HEAD
`5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, including actual dirty source hashes.
Reference trees stay read-only. Do not remove this snapshot or cited artifacts.

Oracle builds reuse `build/lh-oracle`, reconfigured each time; result directories
remain unique and retain source/library/CMake/binary hashes. Cached binaries can
be replaced by later builds; source commits and recorded recipes identify runs.
Never concurrently mutate this checkout, core build or oracle cache during jobs.
No push or artifact deletion. STATUS is the sole current handoff; ROADMAP is backlog.

CPU aarch64; `/home/zlong/anaconda3/bin/python`, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. Disable backend autoload, set OMP/OpenBLAS to 1,
use two build jobs and background.slice. Current schemas: semantics.md (graph v11,
checkpoint v4); portability contract now also states v4. Packed replay only
promises tested first-order public-root VJPs; no replay occurs in inference.
