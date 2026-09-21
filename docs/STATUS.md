# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: `graph-execution-foundation`.

## Current increment and qualification

Same-fiber sum attention baseline is ready to commit: Python per-query oracle,
native head-batched step, complete K/V/log-bias state, tick-repeat decay and cut
decoder; batch/sequence execution still uses explicit scalar fallbacks.
Contract: `fiber-attention.md`. No packed-prefill, CROSSBATCH, whole-LH or speed claim.

Development unit `tide-foundation-fiber-dev-20260921-1613` is inactive, MainPID 0,
exit 0. `artifacts/fiber-dev-20260921-1613/{status.json,task.log,oracle/result.json}`
records 210 targeted fiber/counter tests and original LH Attention passing both
dtypes: each 240 cases, 5760 ticks, 16080 candidate occurrences. Dirty-source tar
SHA256 `c162fa6be69adf0e42997c215b41959e77a3351deba6ea832ee03a1af9da1007`.

Earlier failed development records remain: `fiber-dev-20260921-1601` captures
unsupported dual-input specialization fixtures and ill-conditioned FP32 RMS
composition; `fiber-dev-20260921-1608` captures a promoted-descriptor tolerance
mistake. Corrections and remaining conditioning limits: `fiber-attention.md`.
Do not relabel those historical runs as passed.

Next command after committing: submit `tide-foundation-fiber-20260921-1618` through
scripts/job.py with output `artifacts/fiber-20260921-1618`, running
`python scripts/qualify.py --output-dir artifacts/fiber-20260921-1618 --jobs 2
--lh-snapshot artifacts/lh-source-20260921-1428`.
Freeze this checkout until terminal. Inspect the unit, outer status, verification
result and oracle result. Commit qualification evidence separately after all pass.
The prior clean qualification remains 2596 passed on `5df88e7b91985092870d0197888d8d95311ee7b1`
(`evidence/lh-full.md`), covering original Selector/Add/Full.

## Next implementation

After clean qualification, implement real independent-batch and event-sequence
packed same-fiber attention, retaining the scalar path as a reference. Prototype
`artifacts/fiber-packed-draft.py` is isolated from installed code and was only
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
