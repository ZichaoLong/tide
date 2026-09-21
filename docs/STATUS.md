# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: `graph-execution-foundation`.

## Current qualification

**2764 tests passed** on clean implementation
`eb9dc6c86ef46ab4db9e06029b0029412a890395`; `evidence/fiber-packing.md`.
Original LH Selector/Add/Full/Attention passed in FP64/FP32. Attention per dtype:
264 configurations, 6336 ticks, 17688 candidate updates across all six original
modes, including CROSSBATCH. This qualifies same-fiber sum attention's actual
batch/sequence packing and complete state/clock projection, not whole-model or speed.

Unit `tide-foundation-fiber-pack-20260921-1638` is inactive, MainPID 0, exit 0.
`artifacts/fiber-pack-20260921-1638/{status.json,verification/result.json,oracle/result.json}`
all passed at that clean source. No active job. Evidence is committed separately.
Scalar baseline: `evidence/lh-attention.md` on `dfc2e5b622946614831962fb44683990fa609e8d`.

Earlier failures `fiber-dev-20260921-1601` and `fiber-dev-20260921-1608` remain
retained and failed. Corrections, FP32 conditioning limits and Read precision
comparison policy: `fiber-attention.md` and the baseline qualification report.
Two incorporated packing drafts and one Python import cache were removed after
body comparison with installed source; no other artifacts/reference files removed.

## Next implementation

Continue `lh-attention-plan.md` with post-attention Confluence, then token-clock
Pronounce/IOCortexNet. Proposed bounded profiles reuse the existing memory-name
field: `lh-fiber-attention-{mean,linear,active-softmax,all-softmax}-repeat-v1`;
keep the sum profile unchanged. Coefficients act on attention output rows before
output projection, never on pre-attention Q/K/V inputs. Keep raw sum Aggregate as
independent content for Tide's Full/Emit contract.

Use a single learned `fiber_pool` vector indexed by graph-owned local input slots;
validate its shape against incoming degree. Mean/active softmax normalize only
present sources; all-source softmax uses the full domain including absent sources.
Absent coordinates of a used vector parameter have ordinary zero gradients under
active-only pooling; all-source normalization can give them nonzero gradients.
A cache-only root must have no pooling-parameter path. Preserve sum's old order.

An isolated helper draft is at `artifacts/fiber_pool_draft.py`; it is not installed
or tested. Wire input-slot count through state factories and built-in validation
(Python validation/native adapter, native configure_model). Do not add a graph
field or silently reinterpret the existing sum profile. Extend scalar/packed
pooling and the original oracle, with independent analytic/VJP/order tests,
slot permutations, missing/zero rows, parameter-domain sharing, checkpoints and
scheduler/embedding checks. Then commit, qualify frozen clean source, save evidence.
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
