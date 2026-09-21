# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: `graph-execution-foundation`.

## Current qualification

**3092 tests passed in 296.20s** on clean implementation
`733983e648b18f3da0df215dfd7aedbef1d1d331`; `evidence/fiber-pooling.md`.
Unit `tide-foundation-fiber-pool-20260921-1722` is inactive, MainPID 0, exit 0.
All four records in `artifacts/fiber-pool-20260921-1722/` passed at that clean
source: `status.json`, `verification/result.json`, `oracle/result.json` and
`oracle-release/result.json`. The completed gate is terminal; evidence is saved separately.

Four post-attention pooling profiles now qualify Python/native scalar/packed,
node parallel, frontier, cycles, specialization/embedding, analytic VJPs, cuts,
shared optimizer and checkpoint behavior. Original Selector/Add/Full also passed.
Original Attention runtime-assertions-on: FP64 324 cases with 12 explicitly
unavailable, FP32 336. Assertions-off: each precision 336 cases, 8064 ticks,
22512 candidate updates. The 12 are a hardcoded-FP32 diagnostic denominator in
original FP64 multi-batch active-softmax; the unchanged-source oracle checks this
exception, and an independent build covers the missing numerical cases. Plain
C/C++ assertions remain enabled in both builds. This is not whole-model parity.

Retain failed `fiber-pool-dev-20260921-1709` and successful follow-up
`fiber-pool-dev-20260921-1715` (582 targeted checks plus both original variants).
Only the incorporated 805-byte `artifacts/fiber_pool_draft.py` was removed, after
reviewing a dry-run diff against installed source including its added validation.

## Pronounce implementation and completed development gate

Norm-only Full profiles, independent Python/native sealed token-window conversion,
and the original Pronounce oracle are implemented. `token-window.md` owns the
contract. Actual Pronounce defaults to identity normalization, because ALConfig
has no norm option and `ModuleUtils.h:get_norm_type` falls back to identity.
Optional RMS/Layer norm-only Full profiles are separate supported components.
The vocabulary head stays outside the uniform-width graph; source coordinates
and both body/token clocks stay explicit. No whole IOCortexNet claim follows.

Unit `tide-foundation-pronounce-dev-20260921-175316` is inactive, MainPID 0, exit 0.
Its outer status and both Pronounce results passed. Corrected adapter/readout
checks: **132 passed in 6.45s**, including direct recurrence/VJPs, cuts/detach,
head/state checkpoint and momentum-SGD resume. Original assertions-on FP64:
180 cases/1080 tokens/2430 sample outputs, 24 explicitly unavailable. FP32 and
both assertions-off precisions: 204 cases/1224 tokens/2754 sample outputs each.

Retain failed `pronounce-dev-20260921-1744` (846 tests + Full passed; inherited
AnyModule accessor failed to compile) and `pronounce-dev-20260921-1751` (850 tests
+ Full passed; original default norm disproved the adapter's RMS assumption).
The source tar/hash/logs preserve both. The adapter now casts the stored Module
and checks the actual identity norm policy; no original source was edited.
Full's development oracle covered 108 cases/756 rows per precision, including
36 norm-only combinations in addition to the previous 72 activation cases.

## Clean qualification dispatch and next action

This implementation is being committed before the clean qualification dispatch
`tide-foundation-pronounce-20260921-175646`. Command:

`python scripts/job.py --output-dir artifacts/pronounce-20260921-175646 --
/home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir
artifacts/pronounce-20260921-175646 --jobs 2 --lh-snapshot
artifacts/lh-source-20260921-1428`.

Freeze source and both oracle caches while active. Inspect the unit and all five
terminal records: outer status, verification, oracle, oracle-release and
pronounce-release. No complete qualification result is asserted yet. Save the
result as a separate evidence commit and update current links after success.
Then continue the actual IOCortexNet bridge: four CSR/CSC blocks and exact port
ordering, original selection/caches, activation-to-pending-message projection,
body/token continuation and labeled logits. The single-PDG clock embedding and
large-sparse performance remain separate roadmap obligations.

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
