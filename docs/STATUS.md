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

## Next implementation

Implement real independent-batch and event-sequence
packed same-fiber attention, retaining the scalar path as a reference. Prototype
`artifacts/fiber-packed-draft.py` and `artifacts/fiber_packing_draft.cpp` are isolated
from installed code. Only the Python draft was
checked for values/caches on a small ragged case in both dtypes; it is not qualified.
Use event/source offsets, per-sample visibility, all current-fiber keys, per-event
pooling and ordered bias updates. Check batch/sequence work counters, storage
compaction, every public-root VJP, cuts, clear, sharing and original LH again.

Then continue `lh-attention-plan.md`: post-attention normalized/learned Confluence,
CROSSBATCH original mode, token-clock Pronounce and IOCortexNet adapter. Normalized
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
