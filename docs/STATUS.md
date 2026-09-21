# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest complete foundation qualification: **2318 passed** on clean implementation
`805c74eab1fc8041dc83791e64700dcbc68e26af`; `evidence/lh-selector.md`.
The same job passed original LH heap/tensor Selector comparisons in FP64/FP32:
each dtype 12 cases, 288 ticks and 7368 candidate occurrences against Tide serial,
node-parallel packed streaming and frontier. This is component, not full-model,
equivalence. Norm VJP and payload/descriptor precision are Tide training choices.

Unit `tide-foundation-lh-selector-20260921-1442` is inactive, MainPID 0, exit 0.
Artifacts: `artifacts/lh-selector-20260921-1442/`; outer `status.json`,
`verification/result.json`, `oracle/result.json` all passed with clean source.
Selector evidence is committed separately from the tested implementation.

## Add implementation ready for clean qualification

`lh-add-repeat-v1` now has Python/LibTorch state programs, native equal-gap batch
buckets, physical cut decode and explicit causal time-loop accounting. Contracts
and fixed-parameter interpretation: `lazy-add.md`.
**278 focused tests passed in 10.79 seconds**. Original Selector and Add both
passed in FP64/FP32. Add per dtype: 54 cases, 1296 ticks, 5400 candidate occurrences,
with original single/cached/dual hidden against serial/packed/frontier and whole
versus tick windows. These are development results, not clean qualification.

Unit `tide-foundation-lh-add-dev-20260921-1507` is inactive, MainPID 0, exit 0;
outer and oracle records passed under `artifacts/lh-add-dev-20260921-1507/`.
The earlier `...-1503` job built successfully but stopped on a pending-root test
fixture at an exhausted cut. Cut 7 now guarantees that root; the retry passed.
Failure logs are retained. No active job remains before the prepared launch.

Prepared clean unit: `tide-foundation-lh-add-20260921-1511`, `background.slice`.
Output: `artifacts/lh-add-20260921-1511/`. Commit this implementation first, then
run `python scripts/qualify.py --output-dir artifacts/lh-add-20260921-1511
--jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428` via scripts/job.py.
It runs the full suite plus original Selector/Add in both dtypes. Launch has not
occurred at this edit. Freeze source until terminal; require passed outer,
verification and oracle results. Commit evidence separately after success.

## Immutable original source

Snapshot `artifacts/lh-source-20260921-1428` has 69 C++/JSON-header files and identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.
LH HEAD is `5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, with actual dirty source
captured and checksummed. The original tree was only read. Oracle details:
`lh-selector.md`; full-model mapping: `lh-compatibility.md`.

## Next action after qualification

Follow `lh-add-plan.md`: same-fiber attention, source signaling/activation/
normalization and token-window Pronounce. Do not treat the existing aggregated-event
attention as the LH profile. The fixed snapshot's Hidden/BatchHidden stores a
per-key log bias, subtracts decay every tick and resets appended keys to zero.
Attention sees every key in the complete current fiber. A dedicated profile must
preserve that visibility and distinguish event count from cache atom count.
No whole-LH parity is claimed. STATUS is the sole current handoff; ROADMAP is the
backlog. Current schema versions live in semantics.md; historical evidence is
unchanged. Artifact cleanup dry-run found nothing eligible; no files deleted.

ROADMAP retains broader losses/performance: optimized packed backward, cache
allocation, structured Delta chunks and large sparse workloads. Touched region
histories currently copy/validate their stored maps; avoiding scans of other
regions alone does not prove sparse large-region performance.

## Contracts and environment

Native graph identity v11, checkpoint v4. No implicit older-checkpoint migration.
Region controls are finite payload-dtype tensors; projection Emit/control-blend
require scalars. Descriptor precision is payload/FP64 by Read profile. Mixed
scores promote for softmax; built-ins explicitly cast controls/history updates.

CPU aarch64; `/home/zlong/anaconda3/bin/python`, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu. Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; native builds use two jobs.
No packages changed, no push. Reference repositories are read-only.

Packed replay preserves tested first-order public-root VJPs, including None versus
connected-zero, at a disclosed training cost; inference has no replay. Native
SettleGraph executes its exact TimedDAG encoding; Python is only its graph compiler.
Event attention is not LH same-fiber attention or general pretrained-model import.
No speed claims or artifact deletion occurred in this increment.
