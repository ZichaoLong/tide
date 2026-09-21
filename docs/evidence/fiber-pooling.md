# Post-attention pooling and original LH Confluence

2026-09-22 (Asia/Shanghai), clean implementation
`733983e648b18f3da0df215dfd7aedbef1d1d331`.
**3092 tests passed in 296.20 seconds**. Outer job, verification, original
runtime-assertions-on oracle and separate assertions-off oracle all passed at
that source with exit 0 and empty dirty status.
Unit `tide-foundation-fiber-pool-20260921-1722` is inactive, MainPID 0.
Records: `artifacts/fiber-pool-20260921-1722/{status.json,task.log,verification/,
oracle/,oracle-release/}`.

Command: `python scripts/qualify.py --output-dir artifacts/fiber-pool-20260921-1722
--jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428`, via scripts/job.py in
background.slice. CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu,
C++11 ABI; two build workers, OMP/OpenBLAS single-thread, backend autoload disabled.
Schemas at this source: native graph v11, checkpoint v4. Source, build/library,
CMake and binary hashes are retained in the individual result manifests.

## Qualified contract

Four explicitly named memory profiles add mean, linear, active-softmax and
all-softmax pooling **after attention and before output projection**. The sum
profile and raw sum Aggregate are unchanged. Learned pooling uses one vector
Parameter over the graph-owned local incoming domain. Missing sources participate
only in all-source normalization; numerical zeros and zero coefficients preserve
real query/K/V rows. Contract: `../fiber-pooling.md`.

Independent two-row formulas check values and analytic input/vector derivatives,
output bias once per event, source permutations and cache-only disconnected
roots. Active-only missing vector coordinates have connected-zero gradients;
all-source softmax can give them nonzero gradients. Scalar, packed streaming,
node parallel, frontier, cycles, independent specializations, SettleGraph slot
embedding, complete cuts/detach, shared AdamW and checkpoint restoration agree.
Direct native model tests bypass Python validation and reject missing, malformed,
wrong-dtype/nonfinite parameters, cached-kernel domain mismatch and wrong profile.

The packed Python source-slot variable collision discovered during review was
fixed before the development run. First-order public-root replay limits and the
known FP32 RMS conditioning limits remain unchanged. No higher-order AD, arbitrary
model import, or speed claim is established.

## Original-source comparison and its diagnostic limitation

The unchanged 69-file LH snapshot has identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.
New pooling cases use width 4, two heads, projection biases, decay .01,
clear on/off, PACKED single/multi projection and CROSSBATCH multi projection,
with Tide serial/packed/frontier and tick/whole windows. All prior 264 sum cases
remain included. These are finite bounded logits; extreme original active-softmax
underflow is not an equivalence claim.

| Original build / dtype | Numerical cases | Ticks | Candidate updates | Unavailable cases |
| --- | ---: | ---: | ---: | ---: |
| runtime assertions on / FP64 | 324 | 7776 | 21708 | 12 |
| runtime assertions on / FP32 | 336 | 8064 | 22512 | 0 |
| runtime assertions off / FP64 | 336 | 8064 | 22512 | 0 |
| runtime assertions off / FP32 | 336 | 8064 | 22512 | 0 |

The 12 cases are FP64 active-softmax multi-batch. In original `Confluence.cpp`,
the diagnostic comparison calls `SumCoe(num, indptr)` whose `Confluence.h`
default is explicitly FP32, then multiplies FP64 weights. It throws
`expected scalar type Float but found Double` before its assertion comparison.
The ordinary oracle checks this exact original exception and reports those cases
as unavailable, never as numerical passes. The separate executable turns off
only `ENABLE_RUNTIME_ASSERTION`; ordinary C/C++ assertions remain enabled in both
builds through `-UNDEBUG`. No LH source was patched. Both modes and precision
choices are recorded explicitly; this is not a silent fallback.

`scripts/qualify.py` now runs both variants; assertions-on also requalified
original Selector/Add/Full in both precisions. Separate caches are
`build/lh-oracle` and `build/lh-oracle-release`. Unique results survive cache
rebuilds. This gate does not establish whole IOCortexNet or Pronounce parity.

## Development and retention

`artifacts/fiber-pool-dev-20260921-1709/` remains failed: native build and
582 tests passed, then original FP64 hit the diagnostic defect above. Its source
tar/hash and error stack are preserved. The follow-up
`artifacts/fiber-pool-dev-20260921-1715/` passed 582 targeted checks in 53.27s and
both declared original-oracle variants, before the clean implementation commit.

After clean qualification, the obsolete 805-byte `artifacts/fiber_pool_draft.py`
was removed following a dry-run diff: installed source contains its entire helper
plus explicit unknown-kind rejection. No failure reproducer, source snapshot,
reference tree or cited job artifact was removed.
