# Current handoff

Updated: 2026-09-21. Branch: `graph-execution-foundation`.

## Current work

Latest complete foundation qualification: **2226 passed** on clean implementation
`eb328b36487d7acc4a11a21554c4959c12659d5a`; `evidence/region-programs.md`.
Unit `tide-foundation-region-20260921-1417` finished with exit 0; no live process.
Region/history/controls, cursor/checkpoint v4 and checked counters are qualified.

The next implementation, now ready to commit, adds FP64 norm Read, explicit
payload metadata/conversion, LH selected/affected-count selection and a standalone
original-C++ selector oracle. **400 focused tests passed**, and the original LH
heap/tensor paths matched Tide serial, packed/node-parallel streaming and frontier
in FP64/FP32. Each dtype: 12 cases, 288 ticks, 7368 candidate occurrences. These
are development results; full qualification of a clean commit is the next step.
Both development units finished with exit 0:
- `tide-foundation-lh-selector-dev-20260921-1431`
- `tide-foundation-lh-selector-assert-20260921-1436`
The latter explicitly enables original assertions (`-UNDEBUG`) and validates
snapshot inventory/CMake/library identities. Their paths are under `artifacts/`
with the same suffixes. No active job remains before the prepared submission below.

## Immutable original source

Snapshot `artifacts/lh-source-20260921-1428` has 69 C++/JSON-header files and identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.
LH HEAD is `5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, with actual dirty source
captured and checksummed. The original tree was only read. Oracle details:
`lh-selector.md`; full-model mapping: `lh-compatibility.md`.

## Prepared clean qualification

Commit this implementation, then submit:
- Unit `tide-foundation-lh-selector-20260921-1442`, `background.slice`.
- Job output `artifacts/lh-selector-20260921-1442`.
- `python scripts/qualify.py --output-dir artifacts/lh-selector-20260921-1442
  --jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428`, via scripts/job.py.

This job runs the full foundation suite, then rebuilds/runs the original selector
oracle in both dtypes. Freeze source until terminal. Inspect outer `status.json`,
`verification/result.json` and `oracle/result.json`, plus unit state/exit code.
Launch has not occurred at this handoff commit; the outer record identifies the
actual clean source revision. Archive evidence separately after success.

## Next action after qualification

Follow `lh-add-plan.md`: explicit lazy tick clocks and original Add comparison,
then same-fiber attention, source signaling/activation/normalization and Pronounce.
A repeat decay profile is the original numeric-order baseline; power regrouping
requires separate explicit numerical/route policy. No whole-LH parity is claimed.

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
