# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: `graph-execution-foundation`.

## Current qualification

**3092 tests passed in 296.20s** on clean implementation
`733983e648b18f3da0df215dfd7aedbef1d1d331`; `evidence/fiber-pooling.md`.
Unit `tide-foundation-fiber-pool-20260921-1722` is inactive, MainPID 0, exit 0.
All four records in `artifacts/fiber-pool-20260921-1722/` passed at that clean
source: `status.json`, `verification/result.json`, `oracle/result.json` and
`oracle-release/result.json`. No active job. Evidence is saved separately.

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

## Exact next action

Proceed with `lh-pronounce-plan.md`: norm-only Full, independent Python/native
sealed token-window conversion and actual original Pronounce oracle, then whole
IOCortexNet. The audit found token/phase clocks and contiguous per-port occurrence
positions must stay distinct. A globally empty original Pronounce window is an
invalid input domain; absent samples/phases must not be padded with zero messages.

First gate composes body-tick and token-clock graphs through an explicit adapter;
it does not prove a single-PDG whole-model encoding. The plan records the required
autoregressive sealing and phase/source-domain obligations for that later proof.
No next-stage code exists yet. Implement/test in bounded files, commit, qualify
frozen clean source and commit evidence separately. ROADMAP retains further
training/backward, persistent cache/history and measured-scale work. No speed or
whole-LH equivalence claim is established.

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
