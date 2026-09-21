# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest complete qualification: **2596 tests passed** on clean implementation
`5df88e7b91985092870d0197888d8d95311ee7b1`; `evidence/lh-full.md`.
Original Selector, Add and Full passed in FP64/FP32. Per dtype:
- Selector: 12 cases, 288 ticks, 7368 candidate occurrences.
- Add: 54 cases, 1296 ticks, 5400 candidate occurrences.
- Full: 72 configurations, 504 rows, scalar/packed activation/norm/signaling.
These are component gates, not whole-model or performance claims.

Unit `tide-foundation-lh-full-20260921-1535` is inactive, MainPID 0, exit 0.
`artifacts/lh-full-20260921-1535/{status.json,verification/result.json,oracle/result.json}`
all passed at the same clean source. No active job. Evidence is saved separately
from the tested implementation. Add semantics: `lazy-add.md`; the six Full
profiles, default epsilon/width domain and signaling mapping: `lh-full.md`.

Re-entry now reports all live and three recent terminal jobs; use
`python scripts/status.py --all-jobs` to audit historical failures. Current schema
versions live in semantics.md; historical evidence retains tested versions.
Artifact cleanup dry-run found nothing eligible; no reference/artifact files deleted.

## Immutable original source

Snapshot `artifacts/lh-source-20260921-1428` has 69 C++/JSON-header files and identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.
LH HEAD is `5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, with actual dirty source
captured and checksummed. The original tree was only read. Oracle details:
`lh-selector.md`; full-model mapping: `lh-compatibility.md`.

## Next action

Implement `lh-attention-plan.md`: a separately named same-fiber attention profile
with sum Confluence and tick-repeat log-bias decay first, then post-attention
normalized/learned Confluence and token-window Pronounce. The plan records the
original visibility, clock, cache, source-order and pooling constraints from the
actual C++ snapshot. Existing event attention is not that profile. Start from
AccumulateLocal/Hidden/BatchHidden, StateKernel and Content/SourceInput interfaces.
Full activation/normalization/signaling now has a component oracle; do not
conflate this with an executed IOCortexNet end-to-end comparison.

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
