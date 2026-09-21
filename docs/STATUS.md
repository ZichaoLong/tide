# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: `graph-execution-foundation`.

## Qualification boundary

Last complete qualification: **2764 tests passed** on clean implementation
`eb9dc6c86ef46ab4db9e06029b0029412a890395`; `evidence/fiber-packing.md`.
Original LH Selector/Add/Full/Attention passed FP64/FP32 for that sum-only gate.
The pooling implementation below has passed its development gate, but has not
yet received complete clean-source qualification. Do not promote it prematurely.

## Pooling implementation and completed development gate

Mean, linear, active-softmax and all-softmax profiles are implemented in Python
and native scalar/packed attention; see `fiber-pooling.md`. Original sum is
unchanged. Graph-owned incoming domains validate vector `fiber_pool` parameters,
including a supplied C++ state kernel's policy. Analytic/VJP, missing/zero,
scheduler, embedding, cuts, optimizer/sharing and checkpoint tests are present.
A packed Python source-slot variable collision found during review was fixed.

Unit `tide-foundation-fiber-pool-dev-20260921-1715` is inactive, MainPID 0, exit 0.
`artifacts/fiber-pool-dev-20260921-1715/{status.json,oracle/result.json,
oracle-release/result.json}` are passed. **582 targeted tests passed in 53.27s**.
Both original runtime-assertion variants passed their declared coverage:

- Assertions-on: FP64 324 cases (12 explicitly unavailable), FP32 336 cases.
- Assertions-off: FP64/FP32 each 336 cases, 8064 ticks, 22512 candidate updates.

The 12 unavailable cases hit original `ActSoftmaxConfluence`'s diagnostic
FP32-default `SumCoe` multiplied by FP64 weights. A direct original exception
check preserves this limit; the separate build disables only the original
`ENABLE_RUNTIME_ASSERTION` flag, with plain C/C++ assertions still enabled.
The same immutable snapshot is used throughout, without source edits.
Initial failed run `fiber-pool-dev-20260921-1709` and its source tar/hash remain
retained: build + 582 tests passed, original FP64 failed as described above.

## Exact next action

Commit this coherent implementation, then launch clean qualification as unit
`tide-foundation-fiber-pool-20260921-1722` with:

`python scripts/job.py --output-dir artifacts/fiber-pool-20260921-1722 --
/home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir
artifacts/fiber-pool-20260921-1722 --jobs 2 --lh-snapshot
artifacts/lh-source-20260921-1428`.

Use the durable systemd policy below. Freeze source and both oracle caches while
active. No full result is asserted yet. Inspect the unit and all four results:
outer status, verification, oracle and oracle-release. Qualify now runs both
original assertion variants. After passing, commit evidence separately and update
ROADMAP/current compatibility links. Remove only the incorporated helper
`artifacts/fiber_pool_draft.py` after a dry-run comparison; keep failed reproducers.
Continue Pronounce token-clock/readout, then IOCortexNet mapping and ROADMAP's
remaining training/cache/history/performance work. No whole-LH equivalence or
performance result is claimed.

## Source, reference and execution policy

Snapshot `artifacts/lh-source-20260921-1428`: 69 C++/JSON-header files, identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`; LH HEAD
`5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, including actual dirty source hashes.
Reference trees stay read-only. Do not remove this snapshot or cited artifacts.
Old attention failures `fiber-dev-20260921-1601` and `fiber-dev-20260921-1608`
remain retained; conditioning and norm precision limits: `fiber-attention.md`.

Default original cache: `build/lh-oracle` (runtime assertions on). Separate
release-variant cache: `build/lh-oracle-release`. Reconfigure each run; unique
result directories retain source/library/CMake/binary hashes. Never concurrently
mutate source or shared caches during a job. No push. Delete only known obsolete
project artifacts after dry-run inspection. STATUS is the sole current handoff;
ROADMAP is backlog. Implementation and evidence use separate commits.

CPU aarch64; `/home/zlong/anaconda3/bin/python`, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. Disable backend autoload; OMP/OpenBLAS=1;
use two build jobs, Nice=10 and background.slice. Current schemas: semantics.md
(graph v11, checkpoint v4). Packed training promises tested first-order public
root VJPs with scalar semantic replay; inference does not replay.
