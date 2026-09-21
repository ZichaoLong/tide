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
No active job. Evidence is committed separately from the tested implementation.

## Immutable original source

Snapshot `artifacts/lh-source-20260921-1428` has 69 C++/JSON-header files and identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.
LH HEAD is `5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, with actual dirty source
captured and checksummed. The original tree was only read. Oracle details:
`lh-selector.md`; full-model mapping: `lh-compatibility.md`.

## Next action

Implement `lh-add-plan.md`: lazy tick clocks and original Add component comparison,
then same-fiber attention, source signaling/activation/normalization and Pronounce.
No Add profile code exists at this handoff. Start with `StateKernel`, Python
`memory.py`/`state_program.py` and snapshotted AccumulateLocal/Hidden/BatchHidden.
A repeat decay profile is the original numeric-order baseline; power regrouping
requires separate explicit numerical/route policy. No whole-LH parity is claimed.

Use focused tests, commit implementation, then launch clean `scripts/qualify.py`
under `scripts/job.py` in `background.slice` with `--jobs 2` and the LH snapshot.
Freeze source during jobs. Require terminal outer/test/oracle records before
recording success. STATUS is the sole current handoff; ROADMAP is the backlog.

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
