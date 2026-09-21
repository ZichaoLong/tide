# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest complete qualification: **2436 tests passed** on clean implementation
`bb6590fe8a977e16590243094d0f645b41ccdcb0`; `evidence/lh-add.md`.
Original Selector and Add also passed in FP64/FP32. Each dtype: Selector 12 cases,
288 ticks, 7368 candidate occurrences; Add 54 cases, 1296 ticks, 5400 candidate
occurrences. These are component gates, not whole-model or performance claims.

Unit `tide-foundation-lh-add-20260921-1511` is inactive, MainPID 0, exit 0.
`artifacts/lh-add-20260921-1511/{status.json,verification/result.json,oracle/result.json}`
all report passed with the same clean source. No active job. Evidence is saved
separately from the tested implementation.

The Add profile, lazy/physical state distinction, native equal-gap batch buckets,
causal time-loop accounting and parameter-epoch limits are in `lazy-add.md`.
An earlier development fixture requested a pending root after the queue drained;
cut 7 guarantees that root. The corrected 278 focused tests and clean full suite
passed. The original failure logs are retained under `artifacts/lh-add-dev-20260921-1503/`.

## Immutable original source

Snapshot `artifacts/lh-source-20260921-1428` has 69 C++/JSON-header files and identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.
LH HEAD is `5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, with actual dirty source
captured and checksummed. The original tree was only read. Oracle details:
`lh-selector.md`; full-model mapping: `lh-compatibility.md`.

## Next action

First map LH's selected `activation -> normalization -> per-edge signaling` into
Full. Existing slot-affine Emit provides the projections, but existing backbones
include an extra FFN/residual. Add explicit ReLU/SiLU plus identity/RMS/LayerNorm
profiles using original default eps, then compare the actual ModuleUtils routines.
This closes the current Add-to-output gap before the whole-model adapter.

Next implement same-fiber attention and token-window Pronounce (`lh-add-plan.md`).
The snapshot's Hidden/BatchHidden stores per-key log bias, subtracts decay each
tick and resets appended keys to zero. Every query sees all current-fiber K/V;
cache atom count differs from observation count. Existing event attention is not
that profile. Keep source IDs and post-attention Confluence ordering explicit.

STATUS is the sole handoff; ROADMAP is the backlog. Current schemas live in
semantics.md; immutable evidence retains old versions. Artifact cleanup dry-run
found nothing eligible; no files deleted. Keep future jobs frozen, use two build
threads, and require terminal records before reporting success.

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
